# Engine (Phase 2)

Turns the cleaned EUR returns from Phase 1 into optimized portfolios for the four
Bemo tiers, inside the house caps, and compares them to the hand-set xlsx ladder.

## Run
    .venv/bin/pip install cvxpy scikit-learn scipy
    .venv/bin/python -m engine.run

Prints a per-tier weights table (strategic, risk_parity, hrp, mean_cvar at 5% and
10% bands, and mean_cvar+views) with a return/vol/binding-caps summary, and saves
the `engine_weights` table to the store.

## Core rule
`allocate(returns_upto_T, tier, config, method, use_views)` is pure point-in-time:
it uses only rows on or before T, no clock, no globals. The same function serves the
live allocation and the Phase 3 walk-forward backtest.

## Pieces
- `io.py` loads returns from the store and the config/tilts. `EngineConfig` is frozen.
- `estimators.py` Ledoit-Wolf covariance, Bayes-Stein mean (views off), equilibrium prior.
- `constraints.py` the tier hard-cap set as cvxpy constraints, plus binding diagnosis.
- `optimizers.py` risk_parity (ERC), hrp, mean_cvar (Rockafellar-Uryasev 95% CVaR).
- `views.py` maps the clean directional Tilts rows to Black-Litterman (P, Q, Omega).
- `allocate.py` wires estimators + optimizer + constraints (+ views).

## Notes
- risk_parity and hrp are pure references (long-only, sum 1, no caps). With the cash
  sleeve (near-zero vol) in the set, both concentrate in cash (ERC ~42%, HRP ~99%); that
  is textbook behaviour for risk-based methods when a near-riskless asset is included.
  To use them as investable cores, exclude cash from the risk allocation.
- mean_cvar holds the ladder's expected return at lower vol, within the tactical band.
  Its return equals the strategic return by construction (the CVaR objective binds the
  return target); the value is the vol/CVaR reduction and the interpretable binding caps.
- The yaml `tail_risk` CVaR ceiling has no target number yet, so v1 minimizes CVaR
  rather than capping it. A hard ceiling is a one-line constraint once the number is set.
