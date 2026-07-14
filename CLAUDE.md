# Bemo Europe - Asset Allocation Engine

## What this project is
A dynamic, rules-based multi-asset allocation engine for Bemo Europe (EUR base,
Luxembourg, UCITS / redeemable client mandates). The bank is moving from a static
60/40 toward a risk-profiled, quantitative advisory framework. Built by a quant
intern (applied maths + AI background, ENSIMAG M2 Ingenierie Financiere) as a
CV-grade quant-analyst / quant-developer project.

## Current state (read before building)
The research and specification layer is DONE:
- `docs/Bemo_Institutional_Allocations_v2.pdf` and the source PDFs = how major
  institutions allocate (the survey that justifies the design).
- `docs/Bemo_Portfolio_Construction_Handbook.pdf` = theory and method.
- `docs/Bemo_Strategy_Landscape_and_Toolkit.pdf` = strategy survey plus open-source toolkit.
- `models/Bemo_Allocation_Models_4Tiers.xlsx` = THE SPEC. 8 sheets: institutional spine
  sheets, House-View Tilts, Consensus, and the Bemo 4-Tier model with weights,
  caps, a doughnut and a ladder chart. This file is the single source of truth for
  the target weights, the tilts, and the constraints. Do not contradict it; read it.
- `reference/starter_allocation.py` = a library-usage skeleton ONLY. It uses a generic
  9-ETF universe (VTI, VEA, VWO, IEF, TLT, LQD, GLD, DBC, VNQ) to show how to call
  PyPortfolioOpt and Riskfolio-Lib. It is NOT the Bemo universe and NOT the Bemo
  weights. Use it as a reference for how the libraries work, nothing more.

## Source of truth
The xlsx is the spec. The real universe is the 15 sleeves in the Bemo 4-Tier sheet:
Equity = Europe, US, Dev Asia-Pac, EM, AI theme. Fixed income = EUR Govt, IG credit,
Infl-linked, High yield, EM debt. Alternatives = Gold, Liquid alts / HF, Real assets,
Private mkts. Cash = Cash. The per-tier target weights and the caps section also live
in that sheet. Always pull the sleeves, weights and caps from the xlsx. Never use the
starter's 9-ETF universe as the investable set. Each sleeve needs an ETF proxy mapping
(EUR-denominated or EUR-hedged where it exists), which does not exist yet and is the
first thing to build.
- `docs/CV_Project_Summary.md` = discipline-grouped scope and tool list.

We are now turning the spec into a working, validated engine.

## Repo layout
- `config/`, `data_layer/`, `data/` = the working pipeline (do not clutter). Run it with
  `.venv/bin/python -m data_layer.build`.
- `docs/` = the source PDFs, README.md and CV_Project_Summary.md.
- `models/` = the 4-Tier xlsx spec and its backup.
- `reference/` = vendor library zips and `starter_allocation.py` (example only).
- `scratch/` = throwaway analysis scripts.
- `sources/` = LaTeX source for the institutional-allocations PDF.
- Root keeps `CLAUDE.md`, `requirements.txt`, `.gitignore`, `.venv`.

## Build plan (6 phases, build in order)
1. Data layer: daily total-return series for the sleeves (ETF proxies via yfinance,
   FX from ECB), stored as Parquet via DuckDB or SQLite, plus a data-quality module.
   Handle EUR / Luxembourg FX (hedged vs unhedged) explicitly.
2. Portfolio-construction engine: code the 4 Bemo tiers. Ledoit-Wolf shrinkage,
   HRP and risk parity (Riskfolio-Lib / skfolio), mean-CVaR, Black-Litterman with
   views from the House-View Tilts sheet and the caps column as hard constraints (cvxpy).
3. Backtesting and validation: walk-forward with transaction costs and tolerance-band
   rebalancing (bt / vectorbt), regime filter, purged CV, Deflated Sharpe as the
   go / no-go gate, QuantStats tearsheets per tier.
4. Productionize: pytest, config per tier (YAML), a Streamlit dashboard, CI.
5. Agentic layer: multi-agent research committee (house-view agent, risk agent,
   reporting agent) with human-in-the-loop and an evaluation harness scored against
   the hand-verified xlsx.
6. Package: sanitized public repo (public ETF data only), README, architecture
   diagram, quantified results.

Do not skip ahead to the agentic layer. Solid data and validation come first.

## Bemo house allocation rules (must hold in code)
- Risk-profiled 4-tier ladder: Conservative, Balanced, Growth, Aggressive. Each
  tier sums to 100 percent.
- Four buckets: Equity, Fixed income, Alternatives plus Real assets, Cash.
- Home EUR and Europe tilt inside equity. EM capped at 10 to 20 percent of equity.
  High yield and EM debt capped. Gold sleeve 3 to 5 percent. Private markets only
  for large, liquidity-aware mandates (ELTIF or discretionary mandate), so 0 in the
  standard redeemable model.
- Tactical band of plus or minus 5 to 10 percent around strategic weights.
- UCITS constraints apply: 5 / 10 / 40 diversification, eligible-asset rules
  (gold via ETC, no physical commodities), daily liquidity and redeemability.
- The exact target weights and caps live in the xlsx. Pull them from there, do not
  re-invent them.

## Tech stack (use these)
- Python 3 only. The command is `python3`, never `python`.
- Data: pandas, numpy, yfinance, ECB (FX). Store: Parquet via DuckDB or SQLite.
- Optimise: PyPortfolioOpt then Riskfolio-Lib, cvxpy as the solver.
- Validate and backtest: skfolio, bt.
- Report: QuantStats, openpyxl for Excel.

## Coding conventions (hard rules)
- Keep code clean and minimal. No big comment blocks. Comment only where the logic
  is not self-evident, one short line.
- No decorative banners, no ASCII-art headers, no verbose docstrings on trivial
  functions.
- Small, focused functions. Type hints where they help, not everywhere.
- Do not add features, config or abstractions that were not asked for.
- Fail at boundaries (bad data, missing files); trust internal code.

## Writing style (hard rules, applies to code, comments, docs, and chat)
- No em dashes and no en dashes. Use a plain hyphen or reword.
- No arrows, no middots, no curly quotes or apostrophes, no ellipsis characters.
  Use plain ASCII: straight quotes, "to" or "then" instead of an arrow, a comma or
  full stop instead of a middot.
- Neutral, factual, professional tone. This is supervisor-facing. No salesy or
  AI-flavored phrasing, no superlatives, no first-person marketing.
- Keep quant notation where it is real maths (<=, >=, +/-, ~).

## Working directory
`/home/khairallah/Work_Environment/asset_allocation`
