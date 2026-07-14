import os
from pathlib import Path
import yaml

TIERS = ["conservative", "balanced", "growth", "aggressive"]
CONFIG = Path(os.environ.get(
    "BEMO_CONFIG",
    Path(__file__).resolve().parent.parent / "config" / "bemo_universe.yaml"))


def load_universe(path: Path = CONFIG) -> dict:
    with open(path) as f:
        u = yaml.safe_load(f)
    _check_tier_sums(u["sleeves"])
    return u


def funded_sleeves(u: dict) -> list[dict]:
    # sleeves with a proxy and a non-zero weight in at least one tier
    out = []
    for s in u["sleeves"]:
        if s["proxy"] and any(s["weights"][t] for t in TIERS):
            out.append(s)
    return out


def _check_tier_sums(sleeves: list[dict]) -> None:
    for t in TIERS:
        total = sum(s["weights"][t] for s in sleeves)
        if total != 100:
            raise ValueError(f"tier {t} weights sum to {total}, expected 100")
