# Data layer (Phase 1)

Daily EUR total-return series for the 14 funded Bemo sleeves, plus a data-quality check.

## Run
    python3 -m venv .venv && .venv/bin/pip install pandas numpy yfinance duckdb pyarrow pyyaml
    .venv/bin/python -m data_layer.build

Writes `data/bemo.duckdb`, `data/parquet/*.parquet` and `data/quality_report.txt`.

## Source of truth
`config/bemo_universe.yaml` is the bridge from the Bemo 4-Tier sheet: per sleeve the
bucket, ETF proxy, quote currency, FX policy and the four tier weights, plus the caps
section. Tier weights are asserted to sum to 100 at load.

## FX
Base EUR. ECB daily reference rates (foreign per 1 EUR): EUR value = native price / rate.
- EUR-quoted lines (native / traded / hedged): used directly.
- USD line (DBMF): converted via EURUSD, unhedged.
- GBp lines (SJPA, SGLN, INFR): converted via EURGBP, unhedged.
- EM debt uses the EUR-hedged share class (EMBE.L), so no separate hedge is modelled.

## Quality
Coverage, business-day gaps, stale runs and extreme moves per sleeve. Isolated Yahoo
bad prints are repaired (log-price vs 3-point median) and counted; a windowed check
(`extrW`) confirms the common window (2019-05, bounded by XAIX and DBMF) is clean.

## Tables
`sleeves`, `fx`, `prices_native`, `returns_eur`, `nav_eur` (DuckDB and Parquet).
