import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from engine.io import load_config, load_returns, TIERS, AI_NAME, DB

PARQUET = Path(os.environ.get("BEMO_PARQUET", DB.parent / "parquet"))
from backtest.universe import clean_panel, reduce_universe
from backtest.walkforward import walk_forward, START_EUR
from backtest.benchmark import BENCH_NAME
from backtest.run import (make_target, methods_for, ref_methods, VARIANTS, REF,
                          COST_BPS, TOL_BAND)


def main() -> None:
    rebal, value, attrib = [], [], []
    max_gap = 0.0
    for variant in VARIANTS:
        base = load_config(variant=variant)
        raw = load_returns(base)
        cfg_noai, ret_noai = reduce_universe(base, raw, AI_NAME)
        universes = {f"full_{len(base.funded)}": (base, clean_panel(raw)),
                     f"no_AI_{len(cfg_noai.funded)}": (cfg_noai, ret_noai)}
        for uname, (cfg, ret) in universes.items():
            jobs = ([(REF, REF, m) for m in ref_methods() + [BENCH_NAME]]
                    if variant == VARIANTS[0] else [])
            jobs += [(variant, t, m) for t in TIERS for m in methods_for(variant)]
            for vtag, tier, method in jobs:
                eq, _, rr, vr, at = walk_forward(
                    ret, make_target(method, tier if tier != REF else TIERS[0], cfg),
                    cfg.params["window"], COST_BPS, TOL_BAND)
                tag = {"universe": uname, "variant": vtag, "tier": tier, "method": method}
                rebal += [{**tag, **row} for row in rr]
                value += [{**tag, **row} for row in vr]
                attrib += [{**tag, **row} for row in at]
                max_gap = max(max_gap, _check(uname, vtag, tier, method, eq))

    _save({"rebalance_log": pd.DataFrame(rebal), "value_log": pd.DataFrame(value),
           "attribution": pd.DataFrame(attrib)})
    print(f"Ledger saved. Max equity mismatch vs saved backtest_curves: {max_gap:.2e} "
          f"(0 = math unchanged). Start capital EUR {START_EUR:,.0f}.")


def _check(uname, variant, tier, method, eq) -> float:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        saved = con.execute("select date, equity from backtest_curves where universe=? and "
                            "variant=? and tier=? and method=? order by date",
                            [uname, variant, tier, method]).df()
    finally:
        con.close()
    if len(saved) != len(eq):
        raise ValueError(f"curve length mismatch for {uname}/{variant}/{tier}/{method}: "
                         f"{len(saved)} saved vs {len(eq)} recomputed")
    return float(np.abs(saved["equity"].values - eq.values).max())


def _save(tables: dict) -> None:
    con = duckdb.connect(str(DB))
    try:
        for name, df in tables.items():
            con.register("tmp", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp")
            con.execute(f"COPY {name} TO '{PARQUET / (name + '.parquet')}' (FORMAT PARQUET)")
            con.unregister("tmp")
            print(f"  {name}: {len(df)} rows")
    finally:
        con.close()


if __name__ == "__main__":
    main()
