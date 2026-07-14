import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from engine.io import load_config, load_returns, load_tilts, TIERS, DB
from engine.estimators import ledoit_wolf_cov, bayes_stein_mu
from engine.allocate import allocate, _window
from engine.constraints import binding

TRADING_DAYS = 252


def _full_index(config, series: pd.Series) -> pd.Series:
    names = [s["name"] for s in config.sleeves]
    return series.reindex(names).fillna(0.0)


def main() -> None:
    config = replace(load_config(), tilts=load_tilts())
    cfg = {b: replace(config, params={**config.params, "band": b}) for b in (0.05, 0.10)}
    returns = load_returns(config)
    T = returns.index[-1]
    R = _window(returns, config.params["window"])
    Sigma = ledoit_wolf_cov(R.values)
    mu = bayes_stein_mu(R.values, Sigma)
    print(f"Point-in-time T = {T.date()}  window = {len(R)} days  sleeves = {len(config.funded)}\n")

    records = []
    for tier in TIERS:
        cols = {"strategic": pd.Series(config.tier_w[tier], index=config.funded)}
        cols["risk_parity"] = allocate(returns, tier, config, "risk_parity")
        cols["hrp"] = allocate(returns, tier, config, "hrp")
        cols["mean_cvar@5"] = allocate(returns, tier, cfg[0.05], "mean_cvar")
        cols["mean_cvar@10"] = allocate(returns, tier, cfg[0.10], "mean_cvar")
        cols["mean_cvar@10+V"] = allocate(returns, tier, cfg[0.10], "mean_cvar", use_views=True)

        tbl = pd.DataFrame({k: _full_index(config, v) for k, v in cols.items()})
        print(f"=== {tier.upper()} (weights %) ===")
        print((tbl * 100).round(1).to_string())
        print(f"{'':22}{'ret':>7}{'vol':>7}")
        for k, w in cols.items():
            v = w.reindex(config.funded).values
            ret, vol = float(mu @ v) * 100, float(np.sqrt(v @ Sigma @ v)) * 100
            bnd = ""
            if k.startswith("mean_cvar"):
                band = 0.05 if "@5" in k else 0.10
                bnd = "  binds: " + (", ".join(binding(v, tier, cfg[band], band)) or "none")
            print(f"{k:22}{ret:6.2f}%{vol:6.2f}%{bnd}")
            for sl, wt in w.items():
                records.append((tier, k, sl, float(wt)))
        print()

    _save(records)
    print(f"Saved engine_weights ({len(records)} rows) to {DB} and data/parquet/")


def _save(records: list) -> None:
    df = pd.DataFrame(records, columns=["tier", "method", "sleeve", "weight"])
    con = duckdb.connect(str(DB))
    try:
        con.register("tmp", df)
        con.execute("CREATE OR REPLACE TABLE engine_weights AS SELECT * FROM tmp")
        con.execute("COPY engine_weights TO 'data/parquet/engine_weights.parquet' (FORMAT PARQUET)")
    finally:
        con.close()


if __name__ == "__main__":
    main()
