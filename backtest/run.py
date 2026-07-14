import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from engine.io import load_config, load_returns, TIERS, AI_NAME, CASH_NAME, DB

PARQUET = Path(os.environ.get("BEMO_PARQUET", DB.parent / "parquet"))
from engine.allocate import allocate
from backtest.benchmark import bench_weights, BENCH_NAME
from backtest.universe import clean_panel, reduce_universe
from backtest.walkforward import walk_forward, START_EUR
from backtest.metrics import (perf_metrics, annual_turnover, deflated_sharpe,
                              benchmark_metrics, window_metrics, TRADING_DAYS)

STRESS = ("2022-01-01", "2022-12-31")           # the bond-and-equity drawdown year

VARIANTS = ["house", "peer_mid", "peer_low", "peer_high", "us_tilt"]
CAP_HOST = "house"                              # cap regimes only run on the house weights
# the research lab (extra optimisers / overlays / relaxed caps) only runs when BEMO_LAB=1,
# which the US-book pipeline sets. The default EUR run is byte-identical to before.
LAB = os.environ.get("BEMO_LAB") == "1"
# mean_cvar under each constraint layer, from the house policy down to long-only.
# This measures what each cap costs; it is not a menu to pick the winner from.
CAP_METHODS = {"mean_cvar": "full",
               "mean_cvar_no_band": "no_band",
               "mean_cvar_no_sleeve_caps": "no_sleeve_caps",
               "mean_cvar_equity_band_only": "equity_band_only",
               "mean_cvar_uncapped": "uncapped"}
WEIGHT_METHODS = ["strategic", "mean_cvar"]     # depend on the strategic weights
# covariance only: no tier, no variant. min_variance and max_sharpe are classical Markowitz,
# long-only and uncapped, kept as baselines to show why the shrinkage methods exist.
REF_METHODS = ["risk_parity", "hrp", "min_variance", "max_sharpe"]
LAB_REF_METHODS = ["max_diversification", "equal_weight", "inverse_vol"]
# extra house-book candidates: capped Markowitz twin, honest BL momentum views, the CVaR
# dual, ridge-anchored CVaR, the tactical overlays, band sensitivity, relaxed-cap diagnostics.
LAB_TIER_METHODS = ["mean_variance", "black_litterman_mom", "max_ret_cvarcap",
                    "mean_cvar_anchored", "trend_tilt", "vol_target",
                    "regime_breaker", "dual_momentum",
                    "mean_cvar_band10", "mean_cvar_band15",
                    "mean_cvar_relaxed", "trend_tilt_relaxed",
                    "cvarcap_relaxed", "cvarcap_breaker", "cvarcap_dualmom",
                    "trend_breaker", "trend_dualmom"]
BAND_METHODS = {"mean_cvar_band10": 0.10, "mean_cvar_band15": 0.15}
RELAXED_METHODS = {"mean_cvar_relaxed": "mean_cvar", "trend_tilt_relaxed": "trend_tilt"}
# offense optimiser (max return under the strategic CVaR cap, or trend-boosted mu) paired
# with a relaxed cap regime or a risk-off overlay: tilt harder, then de-risk on top.
COMBO_METHODS = {"cvarcap_relaxed": {"method": "max_ret_cvarcap", "regime": "relaxed"},
                 "cvarcap_breaker": {"method": "max_ret_cvarcap", "overlay": "regime_breaker"},
                 "cvarcap_dualmom": {"method": "max_ret_cvarcap", "overlay": "dual_momentum"},
                 "trend_breaker": {"method": "trend_tilt", "overlay": "regime_breaker"},
                 "trend_dualmom": {"method": "trend_tilt", "overlay": "dual_momentum"}}
METHODS = (REF_METHODS + LAB_REF_METHODS + ["strategic"] + list(CAP_METHODS)
           + LAB_TIER_METHODS + [BENCH_NAME])
COST_BPS = 10.0
TOL_BAND = 0.05
REF = "-"                                       # tier/variant placeholder for the references


def ref_methods() -> list:
    return REF_METHODS + (LAB_REF_METHODS if LAB else [])


def methods_for(variant: str) -> list:
    if variant != CAP_HOST:
        return WEIGHT_METHODS
    return ["strategic"] + list(CAP_METHODS) + (LAB_TIER_METHODS if LAB else [])


def make_target(method, tier, config):
    if method == BENCH_NAME:
        w = bench_weights(config)
        return lambda r: w
    if method == "strategic":
        w = pd.Series(config.tier_w[tier], index=config.funded)
        return lambda r: w
    if method in CAP_METHODS:
        regime = CAP_METHODS[method]
        return lambda r: allocate(r, tier, config, "mean_cvar", False, regime)
    if method in BAND_METHODS:
        band = BAND_METHODS[method]
        return lambda r: allocate(r, tier, config, "mean_cvar", False, "full", band)
    if method in RELAXED_METHODS:
        base = RELAXED_METHODS[method]
        return lambda r: allocate(r, tier, config, base, False, "relaxed")
    if method in COMBO_METHODS:
        kw = COMBO_METHODS[method]
        return lambda r: allocate(r, tier, config, kw["method"], False,
                                  kw.get("regime", "full"), None, kw.get("overlay"))
    return lambda r: allocate(r, tier, config, method, use_views=False)


def _run(ret, target_fn, min_hist, rf=None):
    eq, tos, rr, *_ = walk_forward(ret, target_fn, min_hist, COST_BPS, TOL_BAND)
    m = perf_metrics(eq, rf)
    m["turnover"] = annual_turnover(tos, len(eq) / TRADING_DAYS)
    m["end_eur"] = eq.iloc[-1] * START_EUR
    m["profit_eur"] = m["end_eur"] - START_EUR
    m["profit_pct"] = m["profit_eur"] / START_EUR
    m["cost_eur"] = sum(row["cost_eur"] for row in rr)
    s = window_metrics(eq, *STRESS, rf)
    m["s2022_ret"], m["s2022_sharpe"], m["s2022_dd"] = s["ret"], s["sharpe"], s["max_dd"]
    return eq, m


def main() -> None:
    runs, bench, rfs = {}, {}, {}
    for variant in VARIANTS:
        base = load_config(variant=variant)
        raw = load_returns(base)
        cfg_noai, ret_noai = reduce_universe(base, raw, AI_NAME)
        universes = {f"full_{len(base.funded)}": (base, clean_panel(raw)),
                     f"no_AI_{len(cfg_noai.funded)}": (cfg_noai, ret_noai)}
        for uname, (cfg, ret) in universes.items():
            min_hist = cfg.params["window"]
            # the EUR money-market sleeve is the risk-free rate; without it, Sharpe rewards cash
            rf = ret[CASH_NAME]
            rfs[uname] = rf
            if variant == VARIANTS[0]:      # references are variant-independent: run once
                print(f"[{uname}] {ret.index[0].date()} .. {ret.index[-1].date()}  "
                      f"rebalance from index {min_hist}")
                key = (uname, REF, REF, BENCH_NAME)
                runs[key] = _run(ret, make_target(BENCH_NAME, TIERS[0], cfg), min_hist, rf)
                bench[uname] = runs[key][0]
                for method in ref_methods():
                    runs[(uname, REF, REF, method)] = _run(
                        ret, make_target(method, TIERS[0], cfg), min_hist, rf)
            for tier in TIERS:
                for method in methods_for(variant):
                    runs[(uname, variant, tier, method)] = _run(
                        ret, make_target(method, tier, cfg), min_hist, rf)
        print(f"  variant {variant}: done")

    for (u, _, _, meth), (eq, m) in runs.items():
        m.update(benchmark_metrics(eq, bench[u], rfs[u]))

    # the benchmark is not a strategy we searched over, so it does not inflate the DSR trial count
    strat = {k: v for k, v in runs.items() if k[3] != BENCH_NAME}
    all_sh = [m["sharpe"] for _, m in strat.values()]
    n_trials = len(strat)
    for key, (eq, m) in runs.items():
        m["dsr"] = deflated_sharpe(eq, all_sh, n_trials, rfs[key[0]])

    _scoreboard(runs, n_trials)
    _save(runs, n_trials)


def _scoreboard(runs, n_trials) -> None:
    rows = []
    for (u, v, t, meth), (_, m) in runs.items():
        rows.append({"variant": v, "tier": t, "method": meth, "universe": u,
                     "start_EUR": START_EUR, "end_EUR": round(m["end_eur"]),
                     "profit_EUR": round(m["profit_eur"]),
                     "profit_%": round(m["profit_pct"] * 100, 1),
                     "CAGR_%": round(m["ann_return"] * 100, 1),
                     "vol_%": round(m["ann_vol"] * 100, 1),
                     "Sharpe": round(m["sharpe"], 2), "DSR": round(m["dsr"], 3),
                     "maxDD_%": round(m["max_dd"] * 100, 1),
                     "2022_%": round(m["s2022_ret"] * 100, 1),
                     "2022DD_%": round(m["s2022_dd"] * 100, 1),
                     "CVaR95_%": round(m["cvar95"] * 100, 2),
                     "alpha_%": round(m["alpha"] * 100, 1), "beta": round(m["beta"], 2),
                     "TE_%": round(m["tracking_error"] * 100, 1),
                     "IR": round(m["info_ratio"], 2),
                     "turnover": round(m["turnover"], 2),
                     "cost_EUR": round(m["cost_eur"])})
    df = pd.DataFrame(rows)
    df["tier"] = pd.Categorical(df["tier"], TIERS + [REF], ordered=True)
    df["method"] = pd.Categorical(df["method"], METHODS, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], VARIANTS + [REF], ordered=True)
    df = df.sort_values(["tier", "variant", "method", "universe"]).reset_index(drop=True)
    print(f"\n=== SCOREBOARD: {len(df)} runs, DSR trials = {n_trials}, gate = 0.95 ===")
    print(f"({REF} = covariance-only reference: independent of tier and of the strategic weights)")
    print(f"alpha / beta / TE / IR are measured against {BENCH_NAME} in the same universe.")
    with pd.option_context("display.width", 300, "display.max_columns", None,
                           "display.max_rows", None):
        print(df.to_string(index=False))


def _save(runs, n_trials) -> None:
    curves, metrics = [], []
    for (u, v, t, meth), (eq, m) in runs.items():
        for d, val in eq.items():
            curves.append((u, v, t, meth, d, float(val)))
        metrics.append((u, v, t, meth, n_trials, COST_BPS, TOL_BAND, *[m[c] for c in
                        ["ann_return", "ann_vol", "sharpe", "dsr", "max_dd", "cvar95", "turnover",
                         "beta", "alpha", "tracking_error", "info_ratio",
                         "s2022_ret", "s2022_sharpe", "s2022_dd"]]))
    cdf = pd.DataFrame(curves, columns=["universe", "variant", "tier", "method", "date", "equity"])
    mdf = pd.DataFrame(metrics, columns=["universe", "variant", "tier", "method", "n_trials",
                       "cost_bps", "tol_band", "ann_return", "ann_vol", "sharpe",
                       "dsr", "max_dd", "cvar95", "turnover",
                       "beta", "alpha", "tracking_error", "info_ratio",
                       "s2022_ret", "s2022_sharpe", "s2022_dd"])
    bm = mdf[mdf.method == BENCH_NAME].set_index("universe")
    mdf["excess_return"] = mdf.ann_return - mdf.universe.map(bm.ann_return)
    mdf["excess_sharpe"] = mdf.sharpe - mdf.universe.map(bm.sharpe)
    mdf["beats_bench"] = (mdf.excess_return > 0) & (mdf.excess_sharpe > 0)
    con = duckdb.connect(str(DB))
    try:
        for name, df in [("backtest_curves", cdf), ("backtest_metrics", mdf)]:
            con.register("tmp", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp")
            con.execute(f"COPY {name} TO '{PARQUET / (name + '.parquet')}' (FORMAT PARQUET)")
            con.unregister("tmp")
    finally:
        con.close()
    print(f"\nSaved backtest_curves ({len(cdf)} rows) and backtest_metrics ({len(mdf)} rows).")


if __name__ == "__main__":
    main()
