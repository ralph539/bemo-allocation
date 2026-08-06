import os

import duckdb
import pandas as pd

from engine.io import load_config, DB

_BUCKET = None
REF = "-"
CCY = os.environ.get("BEMO_CCY", "EUR")


def _con():
    return duckdb.connect(str(DB), read_only=True)


def options():
    con = _con()
    try:
        return con.execute("select distinct universe, variant, tier, method "
                           "from backtest_metrics").df()
    finally:
        con.close()


def load_run(universe: str, tier: str, method: str, variant: str = "house") -> dict:
    key = [universe, variant, tier, method]
    where = "universe=? and variant=? and tier=? and method=?"
    con = _con()
    try:
        metrics = con.execute(f"select * from backtest_metrics where {where}", key).df()
        if metrics.empty:
            raise KeyError(f"no run for {universe}/{variant}/{tier}/{method}")
        value = con.execute(f"select date, value_eur, daily_ret, drawdown from value_log "
                            f"where {where} order by date", key).df()
        rebal = con.execute(f"select date, sleeve, target_w, w_before, w_after, trade_pct, "
                            f"trade_eur, cost_eur, breached from rebalance_log where {where} "
                            f"order by date, sleeve", key).df()
        attrib = con.execute(f"select sleeve, pnl_eur, avg_weight, contrib_return from "
                             f"attribution where {where} order by pnl_eur desc", key).df()
    finally:
        con.close()
    return {"metrics": metrics.iloc[0], "value": value, "rebal": rebal, "attrib": attrib}


def run_label(variant: str, tier: str, method: str) -> str:
    return method if variant == REF else f"{variant}/{tier}/{method}"


def scoreboard() -> pd.DataFrame:
    con = _con()
    try:
        df = con.execute("select * from backtest_metrics").df()
    finally:
        con.close()
    df["run"] = [run_label(v, t, m) for v, t, m in zip(df.variant, df.tier, df.method)]
    df["gate"] = ["pass" if d >= 0.95 else "FAIL" for d in df.dsr]
    return df


def curves(universe: str) -> pd.DataFrame:
    # every equity curve for a universe, in EUR, as columns keyed by run label
    con = _con()
    try:
        df = con.execute("select variant, tier, method, date, equity from backtest_curves "
                         "where universe=? order by date", [universe]).df()
    finally:
        con.close()
    df["run"] = [run_label(v, t, m) for v, t, m in zip(df.variant, df.tier, df.method)]
    return df.pivot(index="date", columns="run", values="equity") * 1_000_000.0


def _lab_available() -> bool:
    con = _con()
    try:
        con.execute("select 1 from robustness_curves limit 1")
        return True
    except Exception:
        return False
    finally:
        con.close()


def all_curves() -> pd.DataFrame:
    """Every equity curve from both stores, in one long frame.

    The live backtest carries the weight variants but only spans the funded book's
    history. The robustness lab carries the long-history universes on house weights.
    Both are walk-forward curves, so they slice the same way.
    """
    con = _con()
    try:
        live = con.execute("select universe, variant, tier, method, date, equity "
                           "from backtest_curves").df()
        live["source"] = "live"
        frames = [live]
        try:
            lab = con.execute("select universe, tier, method, date, equity "
                              "from robustness_curves").df()
            # match the live convention: tier-free runs (benchmark, risk-structure
            # optimisers) carry no variant either, so run labels stay comparable
            lab["variant"] = [REF if t == REF else "house" for t in lab.tier]
            lab["source"] = "lab"
            frames.append(lab)
        except Exception:
            pass
    finally:
        con.close()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


_LONG_UNIVERSES = ("us_long", "us_long_stocks")


def risk_free(universe: str) -> pd.Series:
    """Daily risk-free return for a universe, so Sharpe is an excess-return Sharpe.

    The EUR and US books use their own money-market sleeve. The long-history lab
    universes use the 13-week T-bill, which is what robustness.py walked them with.
    Returns an empty series when nothing is available, and the caller then reports a
    raw-return Sharpe.
    """
    from engine.io import CASH_NAME
    if universe in _LONG_UNIVERSES:
        p = DB.parent / "robustness_prices.parquet"
        if not p.exists():
            return pd.Series(dtype=float)
        irx = pd.read_parquet(p)["^IRX"].dropna()
        return irx / 100.0 / 252.0
    con = _con()
    try:
        df = con.execute("select date, ret from returns_eur where sleeve=? order by date",
                         [CASH_NAME]).df()
    except Exception:
        return pd.Series(dtype=float)
    finally:
        con.close()
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index(pd.to_datetime(df["date"]))["ret"]


def universe_dates(curves_df: pd.DataFrame, universe: str) -> pd.DatetimeIndex:
    d = curves_df.loc[curves_df.universe == universe, "date"]
    return pd.DatetimeIndex(sorted(d.unique()))


_REBAL_COLS = {"date": "Date", "sleeve": "Sleeve", "target_w": "Target %", "w_before": "Before %",
               "w_after": "After %", "trade_pct": "Trade %", "trade_eur": "Trade EUR",
               "cost_eur": "Cost EUR", "breached": "Traded"}


def pretty_sleeve(s: str) -> str:
    """Display name for a sleeve.

    Strips the '(home)' marker, a config note rather than a fund name. The sleeve keys
    were written for the EUR book, so on a non-EUR book the three currency-specific
    names are relabelled: the USD book holds GOVT, LQD and BIL, and calling those
    'EUR Govt', 'EUR IG credit' and 'EUR money market' on screen is simply wrong.
    """
    s = s.replace(" (home)", "")
    if CCY != "EUR":
        for a in ("Govt", "IG credit", "money market"):
            s = s.replace(f"EUR {a}", f"{CCY} {a}")
    return s


def format_rebal(rebal, eps: float = 5e-7) -> pd.DataFrame:
    # solver residuals (order 1e-10) are not weights: clip them before they reach the screen
    df = rebal.copy()
    for c in ["target_w", "w_before", "w_after", "trade_pct"]:
        df[c] = (df[c] * 100).where(df[c].abs() > eps, 0.0)
    for c in ["trade_eur", "cost_eur"]:
        df[c] = df[c].where(df[c].abs() > 0.5, 0.0)
    df["sleeve"] = df["sleeve"].map(pretty_sleeve)
    return df.rename(columns=_REBAL_COLS)[list(_REBAL_COLS.values())]


def bucket_map() -> dict:
    global _BUCKET
    if _BUCKET is None:
        _BUCKET = {s["name"]: s["bucket"] for s in load_config().sleeves}
    return _BUCKET


def proxy_map() -> dict:
    return {s["name"]: s["proxy"] for s in load_config().sleeves}


def latest_weights(rebal):
    last = rebal[rebal["date"] == rebal["date"].max()]
    return last.set_index("sleeve")["w_after"]


def holdings(rebal, eps: float = 5e-5) -> pd.DataFrame:
    # the actual book on the last rebalance: one row per held sleeve, biggest first
    w = latest_weights(rebal)
    bkt, prx = bucket_map(), proxy_map()
    df = pd.DataFrame({
        "Sleeve": [pretty_sleeve(s) for s in w.index],
        "Bucket": [bkt.get(s, "") for s in w.index],
        "ETF": [prx.get(s, "") for s in w.index],
        "Weight %": (w.values * 100),
    })
    df = df[df["Weight %"] > eps * 100].sort_values("Weight %", ascending=False)
    return df.reset_index(drop=True)


def fmt_eur(x: float) -> str:
    return f"{CCY} {x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"
