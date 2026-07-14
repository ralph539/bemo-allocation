import numpy as np
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def ledoit_wolf_cov(R: np.ndarray) -> np.ndarray:
    # annualized shrunk covariance from daily returns
    return LedoitWolf().fit(R).covariance_ * TRADING_DAYS


def bayes_stein_mu(R: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    # Jorion shrinkage of the sample mean toward the grand (cross-sectional) mean
    m = R.mean(axis=0) * TRADING_DAYS
    n, T = len(m), len(R)
    g = m.mean()
    inv = np.linalg.pinv(Sigma)
    d = m - g
    phi = (n + 2) / ((n + 2) + (T / TRADING_DAYS) * float(d @ inv @ d))
    phi = min(max(phi, 0.0), 1.0)
    return (1 - phi) * m + phi * g


def equilibrium_pi(Sigma: np.ndarray, w_strat: np.ndarray, delta: float) -> np.ndarray:
    return delta * Sigma @ w_strat
