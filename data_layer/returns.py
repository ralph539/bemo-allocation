import numpy as np
import pandas as pd


def repair_spikes(prices: pd.DataFrame, thresh: float = 0.15) -> tuple[pd.DataFrame, dict]:
    # Replace isolated bad prints (log-price deviating > thresh from the 3-point
    # median) with that median. Sustained moves like a real crash are untouched.
    out, counts = prices.copy(), {}
    for c in prices.columns:
        p = prices[c].dropna()
        lp = np.log(p)
        med = lp.rolling(3, center=True).median()
        bad = (lp - med).abs() > thresh
        counts[c] = int(bad.sum())
        if bad.any():
            out.loc[p.index[bad], c] = np.exp(med[bad])
    return out, counts


def _base_level(price: pd.Series, ccy: str, fx: pd.DataFrame, base: str) -> pd.Series:
    # Value of the holding in the portfolio base currency. ECB rates are foreign
    # per EUR. A sleeve already quoted in the base currency needs no conversion;
    # GBp (pence) keeps its 1/100 constant, which cancels in returns and rebased NAV.
    if ccy == base:
        return price
    if base == "EUR":
        pair = "EURUSD" if ccy == "USD" else "EURGBP"
        return price / fx[pair].reindex(price.index).ffill().bfill()
    if base == "USD":
        # base USD: EUR-quoted sleeve to USD is price * EURUSD; GBP via EUR cross.
        eurusd = fx["EURUSD"].reindex(price.index).ffill().bfill()
        if ccy == "EUR":
            return price * eurusd
        if ccy == "GBP":
            return price * eurusd / fx["EURGBP"].reindex(price.index).ffill().bfill()
    raise ValueError(f"no FX path from {ccy} to base {base}")


def build_returns(prices: pd.DataFrame, fx: pd.DataFrame, sleeves: list[dict],
                  base: str = "EUR"):
    ret, nav = {}, {}
    for s in sleeves:
        p = prices[s["proxy"]].dropna()
        lvl = _base_level(p, s["quote_ccy"], fx, base)
        ret[s["name"]] = lvl.pct_change()
        nav[s["name"]] = 100.0 * lvl / lvl.iloc[0]
    ret_df = pd.DataFrame(ret).sort_index()
    nav_df = pd.DataFrame(nav).sort_index()
    ret_df.index.name = nav_df.index.name = "date"
    ret_long = ret_df.reset_index().melt("date", var_name="sleeve", value_name="ret").dropna()
    nav_long = nav_df.reset_index().melt("date", var_name="sleeve", value_name="nav").dropna()
    return ret_df, nav_df, ret_long, nav_long
