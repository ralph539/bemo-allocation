import numpy as np
import pandas as pd

from engine.estimators import ledoit_wolf_cov, bayes_stein_mu, equilibrium_pi
from engine.optimizers import (risk_parity, hrp, mean_cvar, min_variance, max_sharpe,
                               max_diversification, inverse_vol, mean_variance,
                               max_return_cvar_cap)
from engine.overlays import trend_mu, vol_target, regime_breaker, dual_momentum
from engine.views import net_stance, build_views, bl_posterior, momentum_stance
from engine.io import CASH_NAME


def _window(returns_upto_T: pd.DataFrame, window: int) -> pd.DataFrame:
    R = returns_upto_T.dropna(how="any")
    return R.iloc[-window:] if len(R) > window else R


def _risk_core(config, R: pd.DataFrame, solve) -> np.ndarray:
    # RP/HRP over the risky sleeves only; cash (near-zero vol) re-inserted at 0
    keep = [i for i, n in enumerate(config.funded) if n != CASH_NAME]
    w = np.zeros(len(config.funded))
    w[keep] = solve(R.iloc[:, keep].values)
    return w


def _tangency(r: np.ndarray) -> np.ndarray:
    Sigma = ledoit_wolf_cov(r)
    return max_sharpe(Sigma, bayes_stein_mu(r, Sigma))


def _bl_mu(R, Sigma, tier, config, stance) -> np.ndarray:
    Pi = equilibrium_pi(Sigma, config.tier_w[tier], config.params["delta"])
    P, Q, Omega = build_views(stance, Pi, Sigma, config)
    return bl_posterior(Pi, Sigma, P, Q, Omega, config.params["tau"])


def expected_returns(R: np.ndarray, Sigma: np.ndarray, tier: str, config,
                     use_views: bool) -> np.ndarray:
    if not use_views:
        return bayes_stein_mu(R, Sigma)
    return _bl_mu(R, Sigma, tier, config, net_stance(config.tilts, config))


def _check(res, tier: str, method: str, regime: str) -> np.ndarray:
    w, info = res
    if w is None:
        raise ValueError(f"{tier} {method} infeasible ({regime}): {info['binding']}")
    x = np.asarray(w).clip(min=0)
    return x / x.sum()


def _strategic_cvar(R: np.ndarray, ws: np.ndarray, beta: float) -> float:
    losses = -(R @ ws)
    var = np.quantile(losses, beta)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


OVERLAYS = ("vol_target", "regime_breaker", "dual_momentum")


def _cvar_family(R, tier, config, band, regime, method, use_views) -> np.ndarray:
    # mean_cvar and its variants: trend offense, ridge anchoring, BL-momentum views.
    Rv = R.values
    Sigma = ledoit_wolf_cov(Rv)
    if method == "black_litterman_mom":
        mu = _bl_mu(Rv, Sigma, tier, config, momentum_stance(Rv, config))
    elif method == "mean_cvar" and use_views:
        mu = expected_returns(Rv, Sigma, tier, config, True)
    else:
        mu = bayes_stein_mu(Rv, Sigma)
    if method == "trend_tilt":
        mu = trend_mu(mu, Rv, config.params.get("trend_strength", 0.02))
    anchor = config.params.get("anchor", 0.005) if method == "mean_cvar_anchored" else 0.0
    target = float(mu @ config.tier_w[tier])
    return _check(mean_cvar(Rv, mu, target, tier, config, band, regime, anchor),
                  tier, method, regime)


def _apply_overlay(w: np.ndarray, Rv: np.ndarray, config, overlay: str) -> np.ndarray:
    cash = config.idx(CASH_NAME)
    if overlay == "vol_target":
        return vol_target(w, Rv, cash, config.params.get("target_vol", 0.10))
    if overlay == "regime_breaker":
        return regime_breaker(w, Rv, config.eq_idx, cash, config.gold_idx,
                              config.params.get("exit_frac", 0.5))
    if overlay == "dual_momentum":
        return dual_momentum(w, Rv, cash)
    raise ValueError(f"unknown overlay {overlay}")


def _base_weights(R, tier, config, b, regime, method, use_views) -> np.ndarray:
    if method == "risk_parity":
        return _risk_core(config, R, lambda r: risk_parity(ledoit_wolf_cov(r)))
    if method == "hrp":
        return _risk_core(config, R, hrp)
    if method == "min_variance":
        return _risk_core(config, R, lambda r: min_variance(ledoit_wolf_cov(r)))
    if method == "max_sharpe":
        return _risk_core(config, R, _tangency)
    if method == "max_diversification":
        return _risk_core(config, R, lambda r: max_diversification(ledoit_wolf_cov(r)))
    if method == "inverse_vol":
        return _risk_core(config, R, lambda r: inverse_vol(ledoit_wolf_cov(r)))
    if method == "equal_weight":
        return np.full(len(config.funded), 1.0 / len(config.funded))
    if method == "mean_variance":
        Rv = R.values
        mu = expected_returns(Rv, ledoit_wolf_cov(Rv), tier, config, use_views)
        target = float(mu @ config.tier_w[tier])
        return _check(mean_variance(Rv, mu, target, tier, config, b, regime),
                      tier, method, regime)
    if method == "max_ret_cvarcap":
        Rv = R.values
        mu = expected_returns(Rv, ledoit_wolf_cov(Rv), tier, config, use_views)
        ceiling = _strategic_cvar(Rv, config.tier_w[tier], config.params["cvar_alpha"])
        return _check(max_return_cvar_cap(Rv, mu, ceiling, tier, config, b, regime),
                      tier, method, regime)
    if method in ("mean_cvar", "mean_cvar_anchored", "black_litterman_mom", "trend_tilt"):
        return _cvar_family(R, tier, config, b, regime, method, use_views)
    raise ValueError(f"unknown method {method}")


def allocate(returns_upto_T: pd.DataFrame, tier: str, config, method: str = "mean_cvar",
             use_views: bool = False, regime: str = "full", band: float = None,
             overlay: str = None) -> pd.Series:
    # method builds the base weights; overlay (optional) de-risks on top of any base.
    # the standalone overlay method names are shorthand for a mean_cvar base + that overlay.
    R = _window(returns_upto_T, config.params["window"])
    b = config.params["band"] if band is None else band
    base, ov = ("mean_cvar", method) if method in OVERLAYS else (method, overlay)
    w = _base_weights(R, tier, config, b, regime, base, use_views)
    if ov:
        w = _apply_overlay(w, R.values, config, ov)
    return pd.Series(w, index=config.funded, name=method)
