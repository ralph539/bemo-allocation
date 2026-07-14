from dataclasses import replace

import numpy as np

from engine.io import EM_NAME, AI_NAME, GOLD_NAME, HY_NAME, EMD_NAME


def common_start(returns):
    return returns.dropna(how="any").index[0]


def clean_panel(returns):
    # common trading days where every sleeve in the universe has a return
    return returns.dropna(how="any")


def reduce_universe(config, returns, drop: str):
    keep = [n for n in config.funded if n != drop]
    ret = clean_panel(returns[keep])
    return _reduce_config(config, keep), ret


def _reduce_config(config, keep: list):
    idx = [config.funded.index(n) for n in keep]
    tier_w = {t: _renorm(w[idx]) for t, w in config.tier_w.items()}
    eqset = {s["name"] for s in config.sleeves if s["bucket"] == "Equity"}
    eq_idx = np.array([i for i, n in enumerate(keep) if n in eqset])
    gi = lambda name: keep.index(name) if name in keep else -1
    caps = {**config.caps, "equity_band": _shift_band(config, tier_w, eq_idx)}
    return replace(config, funded=keep, tier_w=tier_w, eq_idx=eq_idx, caps=caps,
                   em_idx=gi(EM_NAME), ai_idx=gi(AI_NAME), gold_idx=gi(GOLD_NAME),
                   hyemd_idx=np.array([gi(HY_NAME), gi(EMD_NAME)]))


def _shift_band(config, tier_w: dict, eq_idx: np.ndarray) -> dict:
    # renormalising the weights moves the equity total, so the band must move with it
    out = {}
    for t, w in tier_w.items():
        shift = (w[eq_idx].sum() - config.tier_w[t][config.eq_idx].sum()) * 100
        lo, hi = config.caps["equity_band"][t]
        out[t] = [max(0.0, lo + shift), min(100.0, hi + shift)]
    return out


def _renorm(w: np.ndarray) -> np.ndarray:
    return w / w.sum()
