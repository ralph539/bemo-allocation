"""Robustness lab: the house engine across universes, decades and crises.

Answers one question: does the 4-tier mean-CVaR engine hold up outside the
window it was built on? Each universe is walked forward once per tier and
method over its full history, then the out-of-sample equity curve is sliced
into calendar blocks and crisis windows. No re-fitting per window, so every
sub-period number is honest OOS.

Universes:
  eur_book       the live EUR 14-sleeve book (short history, AI sleeve limits it)
  eur_book_noAI  the same book without the AI sleeve (history back to 2008)
  us_book_noAI   the US-proxy book without AI (2008+)
  us_long        long-history US proxies (index mutual funds, gold futures,
                 QQQ standing in for the thematic sleeve) back to ~2000
  us_long_stocks us_long with the thematic sleeve as an equal-weight mega-cap
                 stock basket (AAPL, MSFT, AMZN): the ETF + single-stock mix

Methods per tier: the house mean-CVaR engine (full caps), the strategic weights,
and the 60/40 benchmark. Run with .venv/bin/python -m backtest.robustness
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from engine.io import load_config, load_returns, TIERS, AI_NAME, CASH_NAME, DB, ROOT
from engine.allocate import allocate
from engine.views import NAME
from backtest.benchmark import bench_weights, BENCH_NAME
from backtest.universe import clean_panel, reduce_universe
from backtest.walkforward import walk_forward
from backtest.metrics import perf_metrics, window_metrics, annual_turnover, TRADING_DAYS

PARQUET = Path(os.environ.get("BEMO_PARQUET", DB.parent / "parquet"))
OUT_DIR = ROOT / "reports" / "robustness"
CACHE = DB.parent / "robustness_prices.parquet"
US_DB = DB.parent / "bemo_us.duckdb"
COST_BPS = 10.0
TOL_BAND = 0.05
ENGINE = "mean_cvar"

# long-history US proxies per sleeve key (lists are equal-weight baskets)
LONG_PROXY = {
    "EU": "VEURX", "US": "VTSMX", "AP": "VPACX", "EM": "VEIEX", "AI": "QQQ",
    "GOV": "VFITX", "IG": "VFICX", "IL": "VIPSX", "HY": "VWEHX", "EMD": "PREMX",
    "GLD": "GC=F", "ALT": "MERFX", "RE": "VGSIX",
}
STOCK_BASKET = ["AAPL", "MSFT", "AMZN"]

BLOCKS = [("2003-2007", "2003-01-01", "2007-12-31"),
          ("2008-2012", "2008-01-01", "2012-12-31"),
          ("2013-2017", "2013-01-01", "2017-12-31"),
          ("2018-2022", "2018-01-01", "2022-12-31"),
          ("2023-2026", "2023-01-01", "2026-12-31")]
CRISES = [("GFC 2008", "2007-10-01", "2009-03-09"),
          ("Euro crisis 2011", "2011-05-02", "2011-10-04"),
          ("COVID 2020", "2020-02-19", "2020-04-30"),
          ("Rate shock 2022", "2022-01-03", "2022-10-14")]
MIN_OBS = 120


def _fetch(tickers: list, start="1996-01-01") -> pd.DataFrame:
    import yfinance as yf
    have = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame()
    missing = [t for t in tickers if t not in have.columns]
    if missing:
        px = yf.download(missing, start=start, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame(missing[0])
        have = px if have.empty else have.join(px, how="outer")
        have.to_parquet(CACHE)
    return have[tickers]


def _cash_daily(index: pd.DatetimeIndex) -> pd.Series:
    # 13-week T-bill yield as the money-market sleeve / risk-free rate
    irx = _fetch(["^IRX"])["^IRX"].reindex(index).ffill()
    return (irx / 100.0 / TRADING_DAYS).fillna(0.0)


def _long_returns(cfg, basket_ai=False) -> pd.DataFrame:
    tickers = sorted({t for t in LONG_PROXY.values()})
    if basket_ai:
        tickers = sorted(set(tickers) | set(STOCK_BASKET))
    px = _fetch(tickers)
    cols = {}
    for key, name in NAME.items():
        if basket_ai and key == "AI":
            cols[name] = px[STOCK_BASKET].pct_change().mean(axis=1, skipna=False)
        else:
            cols[name] = px[LONG_PROXY[key]].pct_change()
    ret = pd.DataFrame(cols)
    ret[CASH_NAME] = _cash_daily(ret.index)
    ret = clean_panel(ret.reindex(columns=cfg.funded))
    missing = set(cfg.funded) - set(ret.columns)
    if missing:
        raise ValueError(f"unmapped sleeves: {missing}")
    return ret


def build_universes() -> dict:
    cfg = load_config()
    out = {}
    eur_raw = load_returns(cfg)
    out["eur_book"] = (cfg, clean_panel(eur_raw))
    out["eur_book_noAI"] = reduce_universe(cfg, eur_raw, AI_NAME)
    us_raw = load_returns(cfg, db=US_DB)
    out["us_book_noAI"] = reduce_universe(cfg, us_raw, AI_NAME)
    out["us_long"] = (cfg, _long_returns(cfg))
    out["us_long_stocks"] = (cfg, _long_returns(cfg, basket_ai=True))
    return out


def make_target(method, tier, cfg):
    if method == BENCH_NAME:
        w = bench_weights(cfg)
        return lambda r: w
    if method == "strategic":
        w = pd.Series(cfg.tier_w[tier], index=cfg.funded)
        return lambda r: w
    return lambda r: allocate(r, tier, cfg, ENGINE, False, "full")


def run_universe(uname, cfg, ret, rows, curves) -> None:
    min_hist = cfg.params["window"]
    rf = ret[CASH_NAME]
    oos = ret.index[min_hist] if len(ret) > min_hist else None
    if oos is None:
        print(f"[{uname}] skipped: only {len(ret)} days, window {min_hist}")
        return
    print(f"[{uname}] data {ret.index[0].date()} .. {ret.index[-1].date()}, "
          f"OOS from {oos.date()}")
    eqs = {}
    for method in [BENCH_NAME, "strategic", ENGINE]:
        for tier in (["-"] if method == BENCH_NAME else TIERS):
            fn = make_target(method, tier, cfg)
            eq, tos, *_ = walk_forward(ret, fn, min_hist, COST_BPS, TOL_BAND)
            eqs[(tier, method)] = eq
            years = len(eq) / TRADING_DAYS
            m = perf_metrics(eq, rf)
            rows.append(dict(universe=uname, tier=tier, method=method, period="full",
                             kind="full", start=str(eq.index[0].date()),
                             end=str(eq.index[-1].date()),
                             ret=float(eq.iloc[-1] / eq.iloc[0] - 1),
                             ann_return=m["ann_return"], ann_vol=m["ann_vol"],
                             sharpe=m["sharpe"], max_dd=m["max_dd"],
                             cvar95=m["cvar95"],
                             turnover=annual_turnover(tos, years)))
            for d, v in eq.items():
                curves.append((uname, tier, method, d, float(v)))
            for pname, s, e in BLOCKS + CRISES:
                sub = eq.loc[s:e]
                kind = "crisis" if (pname, s, e) in CRISES else "block"
                if len(sub) < (30 if kind == "crisis" else MIN_OBS):
                    continue
                wm = window_metrics(eq, s, e, rf)
                rows.append(dict(universe=uname, tier=tier, method=method,
                                 period=pname, kind=kind,
                                 start=str(sub.index[0].date()),
                                 end=str(sub.index[-1].date()),
                                 ret=wm["ret"], ann_return=np.nan, ann_vol=np.nan,
                                 sharpe=wm["sharpe"], max_dd=wm["max_dd"],
                                 cvar95=np.nan, turnover=np.nan))
        print(f"  {method}: done")
    return eqs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for uname, (cfg, ret) in build_universes().items():
        run_universe(uname, cfg, ret, rows, curves)
    df = pd.DataFrame(rows)
    # benchmark-relative excess per (universe, period)
    bm = df[df.method == BENCH_NAME].set_index(["universe", "period"])
    key = list(zip(df.universe, df.period))
    df["bench_ret"] = [bm.ret.get(k, np.nan) for k in key]
    df["bench_dd"] = [bm.max_dd.get(k, np.nan) for k in key]
    df["bench_sharpe"] = [bm.sharpe.get(k, np.nan) for k in key]
    df["excess_ret"] = df.ret - df.bench_ret
    df["beats_bench"] = df.excess_ret > 0
    cdf = pd.DataFrame(curves, columns=["universe", "tier", "method", "date", "equity"])
    PARQUET.mkdir(parents=True, exist_ok=True)
    # one matrix, both stores: the EUR db and the US db that feeds the hosted dashboard
    for db_path in [DB] + ([US_DB] if US_DB.exists() else []):
        con = duckdb.connect(str(db_path))
        try:
            for name, d in [("robustness_metrics", df), ("robustness_curves", cdf)]:
                con.register("tmp", d)
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp")
                if db_path == DB:
                    con.execute(f"COPY {name} TO "
                                f"'{PARQUET / (name + '.parquet')}' (FORMAT PARQUET)")
                con.unregister("tmp")
        finally:
            con.close()
    df.to_csv(OUT_DIR / "robustness_metrics.csv", index=False)
    _summary(df)


def _summary(df: pd.DataFrame) -> None:
    eng = df[(df.method == ENGINE)]
    print("\n=== engine (mean-CVaR, house caps) vs 60/40, return by period ===")
    for kind in ["full", "block", "crisis"]:
        sub = eng[eng.kind == kind]
        if sub.empty:
            continue
        piv = sub.pivot_table(index=["universe", "period"], columns="tier",
                              values="excess_ret", observed=True)
        print(f"\n[{kind}] excess return vs 60/40 (positive = engine wins)")
        print((piv * 100).round(1).to_string())
    n = len(eng[eng.kind != "full"])
    w = int(eng[eng.kind != "full"].beats_bench.sum())
    print(f"\nengine beats 60/40 in {w}/{n} sub-period cells "
          f"({w / n * 100:.0f}%)" if n else "")


if __name__ == "__main__":
    main()
