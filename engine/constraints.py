import cvxpy as cp
import numpy as np


# constraint layers, from fully capped to unconstrained. "full" is the house policy.
# "relaxed" keeps the risk-profile equity band and the UCITS tail caps (gold, HY+EM debt)
# but drops the tactical band and widens the EM / AI thematic caps, so the optimiser can
# tilt harder into US, EM and AI when the signal supports it.
REGIMES = ("full", "no_band", "no_sleeve_caps", "equity_band_only", "relaxed", "uncapped")


def tier_constraints(w, tier: str, config, band: float, regime: str = "full") -> list:
    # hard cvxpy constraints for one tier from the caps block
    if regime not in REGIMES:
        raise ValueError(f"unknown cap regime {regime}")
    caps, ws = config.caps, config.tier_w[tier]
    eq = config.eq_idx
    cons = [cp.sum(w) == 1, w >= 0]          # long-only and fully invested, always
    if regime == "uncapped":
        return cons

    lo, hi = caps["equity_band"][tier]        # the client's risk profile
    cons += [cp.sum(w[eq]) >= lo / 100, cp.sum(w[eq]) <= hi / 100]
    if regime == "equity_band_only":
        return cons

    if regime == "relaxed":
        hyemd_cap = 0.0 if tier == "aggressive" else caps["hy_plus_em_debt"]["lower_tiers_max"] / 100
        glo, ghi = caps["gold_pct"]
        cons += [
            w[config.em_idx] <= 0.40 * cp.sum(w[eq]),      # EM cap doubled vs house 20%
            w[config.gold_idx] >= glo / 100, w[config.gold_idx] <= ghi / 100,
            cp.sum(w[config.hyemd_idx]) <= hyemd_cap,
        ]
        if config.ai_idx >= 0:
            cons.append(w[config.ai_idx] <= 0.25 * cp.sum(w[eq]))   # AI cap widened vs 10%
        return cons

    if regime != "no_sleeve_caps":
        hyemd_cap = 0.0 if tier == "aggressive" else caps["hy_plus_em_debt"]["lower_tiers_max"] / 100
        glo, ghi = caps["gold_pct"]
        cons += [
            w[config.em_idx] <= caps["em_equity_max_pct_of_equity"][1] / 100 * cp.sum(w[eq]),
            w[config.gold_idx] >= glo / 100, w[config.gold_idx] <= ghi / 100,
            cp.sum(w[config.hyemd_idx]) <= hyemd_cap,
        ]
        if config.ai_idx >= 0:   # AI sleeve absent in the AI-removed robustness universe
            cons.append(w[config.ai_idx]
                        <= caps["thematic_ai_max_pct_of_equity"][1] / 100 * cp.sum(w[eq]))

    if regime != "no_band":
        cons += [w <= ws + band, w >= np.maximum(0.0, ws - band)]
    return cons


def binding(w_val: np.ndarray, tier: str, config, band: float, tol: float = 1e-4) -> list:
    caps, ws = config.caps, config.tier_w[tier]
    eq = w_val[config.eq_idx].sum()
    lo, hi = caps["equity_band"][tier]
    hyemd_cap = 0.0 if tier == "aggressive" else caps["hy_plus_em_debt"]["lower_tiers_max"] / 100
    out = []
    if abs(eq - lo / 100) < tol: out.append(f"equity floor {lo}%")
    if abs(eq - hi / 100) < tol: out.append(f"equity cap {hi}%")
    if abs(w_val[config.em_idx] - 0.20 * eq) < tol: out.append("EM = 20% of equity")
    if abs(w_val[config.ai_idx] - 0.10 * eq) < tol: out.append("AI = 10% of equity")
    if abs(w_val[config.gold_idx] - caps["gold_pct"][1] / 100) < tol: out.append("gold cap 5%")
    if abs(w_val[config.gold_idx] - caps["gold_pct"][0] / 100) < tol: out.append("gold floor 3%")
    if abs(w_val[config.hyemd_idx].sum() - hyemd_cap) < tol: out.append("HY+EM debt cap")
    lb, ub = np.maximum(0.0, ws - band), ws + band
    for i in np.where((np.abs(w_val - lb) < tol) | (np.abs(w_val - ub) < tol))[0]:
        out.append(f"band: {config.funded[i][:18]}")
    return out
