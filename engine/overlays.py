import numpy as np

from engine.estimators import ledoit_wolf_cov

# 12-1 month momentum: compound over the last 252 days, skipping the most recent 21
# (the classic Jegadeesh-Titman skip that avoids the short-term reversal month).
MOM_LOOKBACK = 252
MOM_SKIP = 21
MA_WINDOW = 200


def momentum_score(R: np.ndarray) -> np.ndarray:
    r = R[-MOM_LOOKBACK:-MOM_SKIP] if len(R) >= MOM_LOOKBACK else R[:-MOM_SKIP]
    return np.prod(1 + r, axis=0) - 1


def trend_mu(mu: np.ndarray, R: np.ndarray, strength: float = 0.02) -> np.ndarray:
    # offense: nudge expected returns toward recent winners, z-scored and capped.
    # the optimiser still solves inside the same tier caps, so no cap is breached.
    m = momentum_score(R)
    z = (m - m.mean()) / (m.std() + 1e-9)
    return mu + strength * np.clip(z, -2.0, 2.0)


def vol_target(w: np.ndarray, R: np.ndarray, cash_idx: int,
               target_vol: float = 0.10) -> np.ndarray:
    # scale exposure to hit an ex-ante annual vol target; park the remainder in cash.
    Sigma = ledoit_wolf_cov(R)
    vol = float(np.sqrt(w @ Sigma @ w))
    if vol <= 0:
        return w
    scale = min(1.0, target_vol / vol)
    out = w * scale
    out[cash_idx] += 1.0 - out.sum()
    return out


def regime_breaker(w: np.ndarray, R: np.ndarray, eq_idx: np.ndarray, cash_idx: int,
                   gold_idx: int, exit_frac: float = 0.5) -> np.ndarray:
    # circuit breaker: when the equity composite sits below its 200d moving average,
    # move a fraction of equity weight into cash and gold (risk-off).
    comp = np.cumprod(1 + R[:, eq_idx].mean(axis=1))
    if len(comp) < MA_WINDOW or comp[-1] >= comp[-MA_WINDOW:].mean():
        return w
    out = w.copy()
    cut = w[eq_idx] * exit_frac
    out[eq_idx] -= cut
    moved = cut.sum()
    out[gold_idx] += moved * 0.5
    out[cash_idx] += moved * 0.5
    return out


def dual_momentum(w: np.ndarray, R: np.ndarray, cash_idx: int) -> np.ndarray:
    # absolute-momentum filter: any sleeve whose 12-1m momentum is below cash's own
    # gets sold to cash (get out of losing trades, stay in the winners).
    m = momentum_score(R)
    out = w.copy()
    losers = m < m[cash_idx]
    losers[cash_idx] = False
    out[cash_idx] += out[losers].sum()
    out[losers] = 0.0
    return out
