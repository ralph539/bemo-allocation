import numpy as np

# short key -> funded sleeve name
NAME = {
    "US": "Equity - US", "EU": "Equity - Europe (home)",
    "AP": "Equity - Developed Asia-Pacific / Japan", "EM": "Equity - Emerging / Asia",
    "AI": "Equity - Thematic AI / automation", "GOV": "Fixed income - EUR Govt / core",
    "IG": "Fixed income - EUR IG credit", "IL": "Fixed income - Inflation-linked",
    "HY": "Fixed income - High yield", "EMD": "Fixed income - EM debt",
    "GLD": "Gold", "ALT": "Liquid alternatives / hedge funds",
    "RE": "Real assets / REITs / infrastructure",
}

# (firm substring, asset-class substring) -> [(sleeve key, sign)], clean directional rows only
ROW_RULES = {
    ("morgan stanley", "equities"): [("US", 1), ("AI", 1)],
    ("morgan stanley", "core fixed income"): [("GOV", -1)],
    ("morgan stanley", "alternatives"): [("ALT", 1), ("RE", 1)],
    ("hsbc", "global equities"): [("US", 1)],
    ("hsbc", "asia equities"): [("EM", 1), ("AP", 1)],
    ("hsbc", "fixed income"): [("IG", 1), ("EMD", 1), ("HY", -1)],
    ("hsbc", "gold"): [("GLD", 1), ("ALT", 1), ("RE", 1)],
    ("ubs", "equities"): [("US", 1), ("AI", 1), ("GLD", 1)],
    ("julius baer", "equities"): [("US", 1), ("EM", 1)],
    ("julius baer", "fixed income"): [("IG", 1), ("EMD", 1), ("GLD", 1),
                                      ("ALT", 1), ("RE", 1), ("HY", -1)],
    ("barclays", "fixed income"): [("GOV", 1), ("HY", -1), ("GLD", 1)],
    ("lombard odier", "equities"): [("EM", 1), ("EU", 1)],
    ("blackrock", "equities"): [("US", 1), ("EM", 1), ("GOV", -1), ("IL", 1)],
    ("société générale", "equities"): [("US", 1), ("AI", 1), ("GOV", -1)],
    ("societe generale", "equities"): [("US", 1), ("AI", 1), ("GOV", -1)],
}


def momentum_stance(R: np.ndarray, config) -> np.ndarray:
    # point-in-time BL views: cross-sectional 12-1m momentum z-scores, computed only
    # from returns up to T (no hindsight, unlike the House-View Tilts sheet). Feeds the
    # same build_views/bl_posterior machinery as the analyst stance.
    from engine.overlays import momentum_score
    m = momentum_score(R)
    z = (m - m.mean()) / (m.std() + 1e-9)
    return np.clip(z, -2.0, 2.0)


def net_stance(tilts, config) -> np.ndarray:
    s = np.zeros(len(config.funded))
    for _, row in tilts.iterrows():
        firm, ac = str(row["firm"]).lower(), str(row["asset_class"]).lower()
        for (fk, ak), rules in ROW_RULES.items():
            if fk in firm and ak in ac:
                for key, sign in rules:
                    s[config.idx(NAME[key])] += sign
                break
    return s


def build_views(s: np.ndarray, Pi: np.ndarray, Sigma: np.ndarray, config):
    active = np.where(s != 0)[0]
    P = np.eye(len(s))[active]
    kappa, tau = config.params["kappa"], config.params["tau"]
    Q = Pi[active] + np.clip(kappa * s[active], -0.03, 0.03)
    Omega = np.diag([(tau / abs(s[i])) * Sigma[i, i] for i in active])
    return P, Q, Omega


def bl_posterior(Pi, Sigma, P, Q, Omega, tau: float) -> np.ndarray:
    if len(P) == 0:
        return Pi
    tauS = tau * Sigma
    mid = P @ tauS @ P.T + Omega
    return Pi + tauS @ P.T @ np.linalg.solve(mid, Q - P @ Pi)
