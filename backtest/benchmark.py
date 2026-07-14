import pandas as pd

# Passive EUR 60/40, the thing Bemo is moving away from. Held constant, same costs and
# tolerance band as every other run, so the comparison is like for like.
# Equity leg (60): developed regions at roughly MSCI World weights, US 70 / Europe 18 / Japan 12.
# Bond leg (40): EUR aggregate approximated as 60 govt / 40 investment-grade credit.
# It holds no EM, no thematic, no gold, no cash. Assumption, not a fitted choice.
BENCH_60_40 = {
    "Equity - US": 42.0,
    "Equity - Europe (home)": 11.0,
    "Equity - Developed Asia-Pacific / Japan": 7.0,
    "Fixed income - EUR Govt / core": 24.0,
    "Fixed income - EUR IG credit": 16.0,
}
BENCH_NAME = "bench_60_40"


def bench_weights(config) -> pd.Series:
    missing = [s for s in BENCH_60_40 if s not in config.funded]
    if missing:
        raise ValueError(f"benchmark sleeves absent from universe: {missing}")
    w = pd.Series(0.0, index=config.funded)
    for sleeve, pct in BENCH_60_40.items():
        w[sleeve] = pct / 100.0
    if abs(w.sum() - 1.0) > 1e-9:
        raise ValueError(f"benchmark weights sum to {w.sum()}, expected 1")
    return w
