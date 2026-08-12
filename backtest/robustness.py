"""Robustness lab: the house engine across universes, decades and crises.

Answers one question: does the 4-tier mean-CVaR engine hold up outside the
window it was built on? Each universe is walked forward once per tier and
method over its full history, then the out-of-sample equity curve is sliced
into calendar blocks and crisis windows. No re-fitting per window, so every
sub-period number is honest OOS.

Universes:
  eur_book       the live EUR 14-sleeve book (short history, AI sleeve limits it)
  eur_book_noAI  the same book without the AI sleeve
  us_book_noAI   the US-proxy book without AI
  us_long        long-history US proxies (index mutual funds, gold futures,
                 QQQ standing in for the thematic sleeve), out-of-sample from 2003
  us_long_stocks us_long with the thematic sleeve as an equal-weight mega-cap
                 stock basket (AAPL, MSFT, AMZN): the ETF + single-stock mix

Methods per tier: the house mean-CVaR engine and the other tier-dependent optimisers,
the strategic weights, the tier-free risk-structure optimisers, and the 60/40
benchmark. Run with .venv/bin/python -m backtest.robustness
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from engine.io import load_config, load_returns, TIERS, AI_NAME, CASH_NAME, DB, ROOT
from engine.allocate import allocate, _window
from engine.estimators import ledoit_wolf_cov, bayes_stein_mu
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

# methods that depend on the tier and its strategic weights, so they run once per tier
TIER_METHODS = ["strategic", "mean_cvar", "mean_variance", "max_ret_cvarcap",
                "trend_tilt", "black_litterman_mom", "mean_cvar_anchored",
                "vol_target", "regime_breaker", "dual_momentum"]
# risk-structure optimisers: they ignore the tier, so one run per universe
REF_METHODS = ["risk_parity", "hrp", "min_variance", "max_sharpe",
               "max_diversification", "equal_weight", "inverse_vol"]

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
LEDGER_UNIVERSES = ("us_long", "us_long_stocks")
# Cap regimes and combinations, walked across the long histories too. Two ladders:
# the incumbent (mean_cvar) and the attacker (max_ret_cvarcap). band10 is omitted
# on purpose: the engine's default band is already 10 points, so it duplicates the
# base run. Note the attacker's "uncapped" rung is not truly uncapped: its CVaR
# budget survives, because that constraint is the method itself.
EXTRA_SPECS = {
    "mean_cvar_no_band":          dict(base="mean_cvar", regime="no_band"),
    "mean_cvar_no_sleeve_caps":   dict(base="mean_cvar", regime="no_sleeve_caps"),
    "mean_cvar_equity_band_only": dict(base="mean_cvar", regime="equity_band_only"),
    "mean_cvar_uncapped":         dict(base="mean_cvar", regime="uncapped"),
    "mean_cvar_band15":           dict(base="mean_cvar", band=0.15),
    "mean_cvar_relaxed":          dict(base="mean_cvar", regime="relaxed"),
    "trend_tilt_relaxed":         dict(base="trend_tilt", regime="relaxed"),
    "cvarcap_relaxed":            dict(base="max_ret_cvarcap", regime="relaxed"),
    "cvarcap_breaker":            dict(base="max_ret_cvarcap", overlay="regime_breaker"),
    "cvarcap_dualmom":            dict(base="max_ret_cvarcap", overlay="dual_momentum"),
    "trend_breaker":              dict(base="trend_tilt", overlay="regime_breaker"),
    "trend_dualmom":              dict(base="trend_tilt", overlay="dual_momentum"),
    "cvarcap_no_band":            dict(base="max_ret_cvarcap", regime="no_band"),
    "cvarcap_no_sleeve_caps":     dict(base="max_ret_cvarcap", regime="no_sleeve_caps"),
    "cvarcap_equity_band_only":   dict(base="max_ret_cvarcap", regime="equity_band_only"),
    "cvarcap_uncapped":           dict(base="max_ret_cvarcap", regime="uncapped"),
}

# Input experiments: same caps, same loop, one estimator changed at a time.
# ewma: scenario weights lambda=0.97, the RiskMetrics monthly parameter, fixed before
#       any result was seen. Recent risk counts more; 2022's correlation flip enters
#       the tail within weeks instead of being averaged over three years.
# w5y / w10y: the whole engine (mu and scenarios) on a longer lookback.
# yield: mu = current cash rate + each sleeve's expanding-window excess premium,
#        shrunk toward the cross-sleeve mean. Point-in-time only. Levels move with
#        rates; history sets only the premia.
EWMA_LAM = 0.97
EXPERIMENTS = ("mean_cvar_ewma", "cvarcap_ewma", "mean_cvar_w5y", "mean_cvar_w10y",
               "mean_cvar_yield", "cvarcap_yield")
EXP_WINDOW = {"mean_cvar_w5y": 1260, "mean_cvar_w10y": 2520}


def _ewma_probs(S: int) -> np.ndarray:
    w = EWMA_LAM ** np.arange(S - 1, -1, -1)
    return w / w.sum()


def _weighted_cvar(losses: np.ndarray, p: np.ndarray, beta: float) -> float:
    order = np.argsort(losses)[::-1]
    lp, pp = losses[order], p[order]
    tail = 1.0 - beta
    prior = np.concatenate([[0.0], np.cumsum(pp)[:-1]])
    take = np.clip(tail - prior, 0.0, pp)
    return float((lp * take).sum() / tail)


def _shrunk(vec: np.ndarray, Sigma: np.ndarray, T: int) -> np.ndarray:
    g = vec.mean()
    d = vec - g
    n = len(vec)
    phi = (n + 2) / ((n + 2) + (T / 252.0) * float(d @ np.linalg.pinv(Sigma) @ d))
    phi = min(max(phi, 0.0), 1.0)
    return (1 - phi) * vec + phi * g


def _mu_yield(upto, cfg) -> np.ndarray:
    # expanding excess premia over the cash sleeve, shrunk; level from today's cash rate
    rf = upto[CASH_NAME]
    prem = (upto.sub(rf, axis=0)).mean().values * 252.0
    W = _window(upto, cfg.params["window"]).values
    prem = _shrunk(prem, ledoit_wolf_cov(W), len(upto))
    rf_now = float(rf.tail(21).mean()) * 252.0
    mu = rf_now + prem
    mu[cfg.idx(CASH_NAME)] = rf_now
    return mu


def make_experiment(method, tier, cfg):
    from engine.optimizers import mean_cvar, max_return_cvar_cap
    band = cfg.params["band"]

    def solve(upto):
        W = EXP_WINDOW.get(method, cfg.params["window"])
        R = _window(upto, W)
        Rv = R.values
        if method in ("mean_cvar_yield", "cvarcap_yield"):
            mu = _mu_yield(upto, cfg)
        else:
            mu = bayes_stein_mu(Rv, ledoit_wolf_cov(Rv))
        p = _ewma_probs(len(Rv)) if method.endswith("_ewma") else None
        ws = cfg.tier_w[tier]
        if method.startswith("cvarcap"):
            losses = -(Rv @ ws)
            beta = cfg.params["cvar_alpha"]
            ceil = (_weighted_cvar(losses, p, beta) if p is not None
                    else float(np.sort(losses)[int(beta * len(losses)):].mean()))
            w, _ = max_return_cvar_cap(Rv, mu, ceil, tier, cfg, band, "full", probs=p)
        else:
            target = float(mu @ ws)
            w, _ = mean_cvar(Rv, mu, target, tier, cfg, band, "full", 0.0, probs=p)
        return pd.Series(w, index=cfg.funded)

    return solve


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
    if method in EXPERIMENTS:
        return make_experiment(method, tier, cfg)
    if method in EXTRA_SPECS:
        s = EXTRA_SPECS[method]
        return lambda r: allocate(r, tier, cfg, s["base"], False,
                                  s.get("regime", "full"), band=s.get("band"),
                                  overlay=s.get("overlay"))
    # reference optimisers ignore the tier; allocate() resolves the rest by name
    t = TIERS[0] if method in REF_METHODS else tier
    return lambda r: allocate(r, t, cfg, method, False, "full")


def run_universe(uname, cfg, ret, rows, curves, rebals=None, attribs=None) -> None:
    min_hist = cfg.params["window"]
    rf = ret[CASH_NAME]
    oos = ret.index[min_hist] if len(ret) > min_hist else None
    if oos is None:
        print(f"[{uname}] skipped: only {len(ret)} days, window {min_hist}")
        return
    print(f"[{uname}] data {ret.index[0].date()} .. {ret.index[-1].date()}, "
          f"OOS from {oos.date()}")
    eqs = {}
    # the diagnostic ladders run only where the dashboard shows them
    extras = (list(EXTRA_SPECS) + list(EXPERIMENTS)) if uname in LEDGER_UNIVERSES else []
    for method in [BENCH_NAME] + TIER_METHODS + extras + REF_METHODS:
        # bench and the risk-structure optimisers do not depend on the tier
        tiers = ["-"] if method in [BENCH_NAME] + REF_METHODS else TIERS
        for tier in tiers:
            fn = make_target(method, tier, cfg)
            mh = EXP_WINDOW.get(method, min_hist)
            try:
                eq, tos, rbl, _vl, att = walk_forward(ret, fn, mh, COST_BPS,
                                                      TOL_BAND)
            except Exception as e:                  # a solver can fail on a short universe
                print(f"  {method}/{tier}: skipped ({type(e).__name__})")
                continue
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
            if rebals is not None and uname in LEDGER_UNIVERSES:
                for r in rbl:
                    rebals.append({"universe": uname, "tier": tier, "method": method, **r})
                for a in att:
                    attribs.append({"universe": uname, "tier": tier, "method": method, **a})
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
    rows, curves, rebals, attribs = [], [], [], []
    for uname, (cfg, ret) in build_universes().items():
        run_universe(uname, cfg, ret, rows, curves, rebals, attribs)
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
    rdf = pd.DataFrame(rebals)
    adf = pd.DataFrame(attribs)
    PARQUET.mkdir(parents=True, exist_ok=True)
    # one matrix, both stores: the EUR db and the US db that feeds the hosted dashboard
    for db_path in [DB] + ([US_DB] if US_DB.exists() else []):
        con = duckdb.connect(str(db_path))
        try:
            for name, d in [("robustness_metrics", df), ("robustness_curves", cdf),
                            ("robustness_rebalance", rdf),
                            ("robustness_attribution", adf)]:
                if d.empty:
                    continue
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
