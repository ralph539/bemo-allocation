"""Calendar windows and windowed performance, computed from an equity curve.

Every sub-period number in the dashboard is a slice of one walk-forward curve,
never a re-fit. That keeps sub-period results honest out-of-sample: the engine
never saw the window boundary when it chose weights.
"""
import numpy as np
import pandas as pd

from backtest.metrics import perf_metrics

FULL = "Full history"

BLOCKS = [("2003-2007", "2003-01-01", "2007-12-31"),
          ("2008-2012", "2008-01-01", "2012-12-31"),
          ("2013-2017", "2013-01-01", "2017-12-31"),
          ("2018-2022", "2018-01-01", "2022-12-31"),
          ("2023-2026", "2023-01-01", "2026-12-31")]
CRISES = [("GFC 2008", "2007-10-01", "2009-03-09"),
          ("Euro crisis 2011", "2011-05-02", "2011-10-04"),
          ("COVID 2020", "2020-02-19", "2020-04-30"),
          ("Rate shock 2022", "2022-01-03", "2022-10-14")]

KIND = {**{n: "block" for n, _, _ in BLOCKS}, **{n: "crisis" for n, _, _ in CRISES}}
BOUNDS = {n: (s, e) for n, s, e in BLOCKS + CRISES}
MIN_DAYS = {"block": 120, "crisis": 30}

NOTE = {
    "GFC 2008": "Peak to trough of the global financial crisis.",
    "Euro crisis 2011": "The euro sovereign debt sell-off.",
    "COVID 2020": "The pandemic crash, February to April 2020.",
    "Rate shock 2022": "Stocks and bonds fell together as rates rose.",
}


def bounds(period, dates):
    """Start and end timestamps for a period, clipped to the data actually held."""
    if period == FULL:
        return dates.min(), dates.max()
    s, e = BOUNDS[period]
    return max(pd.Timestamp(s), dates.min()), min(pd.Timestamp(e), dates.max())


MIN_ANNUALISE = 120     # below this, a CAGR or an annualised vol is noise, not a rate
MIN_COVER = 0.5         # a window shown under its calendar name must be at least half there


def coverage(period, dates) -> float:
    """Fraction of the period's calendar span the universe actually holds."""
    if period == FULL:
        return 1.0
    s, e = (pd.Timestamp(x) for x in BOUNDS[period])
    have = dates[(dates >= s) & (dates <= e)]
    if not len(have):
        return 0.0
    return float((have.max() - have.min()).days + 1) / ((e - s).days + 1)


def available(dates) -> list:
    """Periods with enough observations, and enough of their span, to be labelled."""
    out = [FULL]
    for name, s, e in BLOCKS + CRISES:
        n = ((dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))).sum()
        if n >= MIN_DAYS[KIND[name]] and coverage(name, dates) >= MIN_COVER:
            out.append(name)
    return out


def slice_curve(equity: pd.Series, period, rebase=True) -> pd.Series:
    """The curve over one window, restarted at 1.0 so windows are comparable."""
    if period == FULL:
        sub = equity
    else:
        s, e = BOUNDS[period]
        sub = equity.loc[s:e]
    if len(sub) < 3:
        return pd.Series(dtype=float)
    return sub / sub.iloc[0] if rebase else sub


def metrics(equity: pd.Series, period, rf=None) -> dict:
    """Performance over one window. Returns NaNs when the slice is too short.

    On a window shorter than MIN_ANNUALISE, CAGR and annualised volatility are set to
    NaN: compounding a two-month crash to a yearly rate produces a number that looks
    like a rate and is not one. Total return and drawdown stay valid at any length.
    """
    sub = slice_curve(equity, period, rebase=False)
    if len(sub) < 3:
        return {k: np.nan for k in
                ("ret", "ann_return", "ann_vol", "sharpe", "max_dd", "cvar95", "n_days")}
    m = perf_metrics(sub, rf)
    m["ret"] = float(sub.iloc[-1] / sub.iloc[0] - 1)
    m["n_days"] = len(sub)
    if len(sub) < MIN_ANNUALISE:
        m["ann_return"] = np.nan
        m["ann_vol"] = np.nan
    return m


def label(period, dates=None) -> str:
    if period == FULL:
        if dates is None or not len(dates):
            return FULL
        return f"{FULL} ({dates.min():%Y} to {dates.max():%Y})"
    return period
