import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_layer.universe import load_universe, funded_sleeves, TIERS
from data_layer.sources import fetch_fx, fetch_prices
from data_layer.returns import build_returns, repair_spikes
from data_layer.quality import run_quality
from data_layer.store import write_store

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB = Path(os.environ.get("BEMO_DB", DATA / "bemo.duckdb"))
PARQUET = Path(os.environ.get("BEMO_PARQUET", DATA / "parquet"))


def sleeves_table(u: dict) -> pd.DataFrame:
    rows = []
    for s in u["sleeves"]:
        row = {"sleeve": s["name"], "bucket": s["bucket"], "proxy": s["proxy"],
               "quote_ccy": s["quote_ccy"], "fx": s["fx"]}
        row.update({f"w_{t}": s["weights"][t] for t in TIERS})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    u = load_universe()
    sleeves = funded_sleeves(u)
    tickers = [s["proxy"] for s in sleeves]
    print(f"Funded sleeves: {len(sleeves)} | tickers: {tickers}")

    base = u.get("base_currency", "EUR")
    fx = fetch_fx(u["start"])
    prices = fetch_prices(tickers, u["start"])
    prices, repairs = repair_spikes(prices)
    ret_df, nav_df, ret_long, nav_long = build_returns(prices, fx, sleeves, base)

    report, issues = run_quality(prices, fx, ret_df, sleeves, repairs)

    prices_long = (prices.reset_index()
                   .melt("date", var_name="ticker", value_name="close").dropna())
    tables = {
        "sleeves": sleeves_table(u),
        "fx": fx.reset_index(),
        "prices_native": prices_long,
        "returns_eur": ret_long,
        "nav_eur": nav_long,
    }
    write_store(DB, PARQUET, tables)

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "quality_report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nStored: {DB} + {PARQUET}/ ({', '.join(tables)})")


if __name__ == "__main__":
    main()
