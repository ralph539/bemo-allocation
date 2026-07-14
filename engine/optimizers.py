import cvxpy as cp
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from engine.constraints import tier_constraints, binding


def risk_parity(Sigma: np.ndarray) -> np.ndarray:
    # equal risk contribution via the convex log-barrier form, then normalise
    n = len(Sigma)
    w = cp.Variable(n, pos=True)
    obj = 0.5 * cp.quad_form(w, cp.psd_wrap(Sigma)) - cp.sum(cp.log(w)) / n
    cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
    x = np.asarray(w.value).clip(min=0)
    return x / x.sum()


def min_variance(Sigma: np.ndarray) -> np.ndarray:
    # classical Markowitz global minimum-variance, long-only, no caps
    w = cp.Variable(len(Sigma))
    cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))),
               [cp.sum(w) == 1, w >= 0]).solve(solver=cp.CLARABEL)
    x = np.asarray(w.value).clip(min=0)
    return x / x.sum()


def max_sharpe(Sigma: np.ndarray, mu: np.ndarray, rf: float = 0.0) -> np.ndarray:
    # classical Markowitz tangency portfolio, long-only, no caps.
    # convex reformulation: min y'Sy s.t. (mu-rf)'y = 1, y >= 0; then w = y / sum(y)
    ex = mu - rf
    if np.max(ex) <= 0:
        return min_variance(Sigma)          # no positive excess return: tangency undefined
    y = cp.Variable(len(mu))
    cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))),
               [ex @ y == 1, y >= 0]).solve(solver=cp.CLARABEL)
    x = np.asarray(y.value).clip(min=0)
    return x / x.sum()


def hrp(R: np.ndarray) -> np.ndarray:
    # Lopez de Prado hierarchical risk parity
    cov = np.cov(R, rowvar=False)
    corr = np.corrcoef(R, rowvar=False)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    order = leaves_list(linkage(squareform(dist, checks=False), method="single"))
    w = np.ones(len(cov))
    clusters = [list(order)]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
            a = 1 - vl / (vl + vr)
            w[left] *= a
            w[right] *= 1 - a
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


def max_diversification(Sigma: np.ndarray) -> np.ndarray:
    # Choueifaty diversification ratio: maximise sigma'w / sqrt(w'Sigma w), long-only.
    # convex form: min y'Sy s.t. sigma'y = 1, y >= 0; then w = y / sum(y)
    sig = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    y = cp.Variable(len(Sigma))
    cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))),
               [sig @ y == 1, y >= 0]).solve(solver=cp.CLARABEL)
    x = np.asarray(y.value).clip(min=0)
    return x / x.sum()


def inverse_vol(Sigma: np.ndarray) -> np.ndarray:
    # naive risk parity: weight inversely to each asset's own volatility
    iv = 1.0 / np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    return iv / iv.sum()


def mean_variance(R: np.ndarray, mu: np.ndarray, target: float, tier: str, config,
                  band: float, regime: str = "full"):
    # Markowitz twin of mean_cvar: minimise variance s.t. the same return target and
    # the same tier hard caps. Same frame, different risk measure (vol vs tail).
    Sigma = np.cov(R, rowvar=False)
    n = R.shape[1]
    w = cp.Variable(n)
    cons = tier_constraints(w, tier, config, band, regime)
    cons += [mu @ w >= target]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return None, {"status": prob.status, "binding": ["infeasible"]}
    x = np.asarray(w.value).clip(min=0)
    x = x / x.sum()
    return x, {"status": prob.status, "binding": binding(x, tier, config, band)}


def mean_cvar(R: np.ndarray, mu: np.ndarray, target: float, tier: str, config,
              band: float, regime: str = "full", anchor: float = 0.0):
    # minimise 95% CVaR of loss s.t. return target and the tier hard caps.
    # anchor > 0 adds a ridge pull toward the strategic weights (lowers turnover).
    S, n = R.shape
    beta = config.params["cvar_alpha"]
    w = cp.Variable(n)
    var = cp.Variable()
    z = cp.Variable(S, nonneg=True)
    cvar = var + cp.sum(z) / ((1 - beta) * S)
    obj = cvar
    if anchor > 0:
        obj = cvar + anchor * cp.sum_squares(w - config.tier_w[tier])
    cons = tier_constraints(w, tier, config, band, regime)
    cons += [z >= -(R @ w) - var, mu @ w >= target]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return None, {"status": prob.status, "binding": ["infeasible"]}
    x = np.asarray(w.value).clip(min=0)
    x = x / x.sum()
    return x, {"status": prob.status, "cvar": float(cvar.value),
               "binding": binding(x, tier, config, band)}


def max_return_cvar_cap(R: np.ndarray, mu: np.ndarray, cvar_ceiling: float, tier: str,
                        config, band: float, regime: str = "full"):
    # dual of mean_cvar: maximise expected return s.t. a binding 95% CVaR ceiling and
    # the tier hard caps. The ceiling is the strategic portfolio's own CVaR, so this
    # chases return without taking more tail risk than the policy allows.
    S, n = R.shape
    beta = config.params["cvar_alpha"]
    w = cp.Variable(n)
    var = cp.Variable()
    z = cp.Variable(S, nonneg=True)
    cvar = var + cp.sum(z) / ((1 - beta) * S)
    cons = tier_constraints(w, tier, config, band, regime)
    cons += [z >= -(R @ w) - var, cvar <= cvar_ceiling]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return None, {"status": prob.status, "binding": ["infeasible"]}
    x = np.asarray(w.value).clip(min=0)
    x = x / x.sum()
    return x, {"status": prob.status, "cvar": float(cvar.value),
               "binding": binding(x, tier, config, band)}


def _cluster_var(cov: np.ndarray, idx: list) -> float:
    sub = cov[np.ix_(idx, idx)]
    iv = 1 / np.diag(sub)
    w = iv / iv.sum()
    return float(w @ sub @ w)
