import numpy as np
from scipy.stats import skew, kurtosis, norm

TRADING_DAYS = 252
EULER = 0.5772156649015329


def _excess(equity, rf):
    # rf is a daily risk-free return series (the EUR money-market sleeve)
    r = equity.pct_change().dropna()
    if rf is None:
        return r.values, r.values
    return r.values, (r - rf.reindex(r.index).fillna(0.0)).values


def perf_metrics(equity, rf=None) -> dict:
    r, ex = _excess(equity, rf)
    n = len(r)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (TRADING_DAYS / n) - 1
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sde = ex.std(ddof=1)
    sharpe = ex.mean() / sde * np.sqrt(TRADING_DAYS) if sde > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())
    cut = np.quantile(r, 0.05)
    cvar95 = float(r[r <= cut].mean())
    return {"ann_return": cagr, "ann_vol": vol, "sharpe": sharpe,
            "max_dd": max_dd, "cvar95": cvar95}


def window_metrics(equity, start: str, end: str, rf=None) -> dict:
    # performance over a calendar sub-window (e.g. the 2022 stress year), from the
    # already-computed equity curve. Empty or too-short slices return NaNs.
    sub = equity.loc[start:end]
    if len(sub) < 3:
        return {"ret": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    m = perf_metrics(sub, rf)
    return {"ret": float(sub.iloc[-1] / sub.iloc[0] - 1),
            "sharpe": m["sharpe"], "max_dd": m["max_dd"]}


def annual_turnover(turnovers: list, years: float) -> float:
    return sum(to for _, to in turnovers) / years if years > 0 else 0.0


def benchmark_metrics(equity, bench, rf=None) -> dict:
    # beta and Jensen alpha on returns in excess of the risk-free rate; TE and IR are
    # benchmark-relative and need no rf, it cancels
    rp = equity.pct_change().dropna()
    rb = bench.pct_change().dropna()
    idx = rp.index.intersection(rb.index)
    rp, rb = rp.loc[idx], rb.loc[idx]
    if len(idx) < 3:
        return {"beta": np.nan, "alpha": np.nan, "tracking_error": np.nan, "info_ratio": np.nan}
    f = rf.reindex(idx).fillna(0.0) if rf is not None else 0.0
    xp, xb = (rp - f).values, (rb - f).values
    var_b = xb.var(ddof=1)
    beta = float(np.cov(xp, xb, ddof=1)[0, 1] / var_b) if var_b > 0 else float("nan")
    alpha = float((xp.mean() - beta * xb.mean()) * TRADING_DAYS)
    ex = (rp - rb).values
    sd = ex.std(ddof=1)
    return {"beta": beta, "alpha": alpha,
            "tracking_error": float(sd * np.sqrt(TRADING_DAYS)),
            "info_ratio": float(ex.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0}


def deflated_sharpe(equity, all_annual_sharpes: list, n_trials: int, rf=None) -> float:
    # Bailey and Lopez de Prado: probability the true Sharpe > 0 after correcting
    # for skew, kurtosis and the number of configurations tried. Uses excess returns.
    _, r = _excess(equity, rf)
    n = len(r)
    sd = r.std(ddof=1)
    if sd == 0 or n < 3:
        return float("nan")
    sr = r.mean() / sd                      # per-period observed Sharpe
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    trials = np.array(all_annual_sharpes) / np.sqrt(TRADING_DAYS)
    var_sr = np.var(trials, ddof=1) if len(trials) > 1 else 0.0
    N = max(n_trials, 2)
    sr0 = np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1 / N)
                             + EULER * norm.ppf(1 - 1 / (N * np.e)))
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / den))
