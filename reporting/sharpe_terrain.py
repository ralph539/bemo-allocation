"""3D Sharpe terrain over two engine hyperparameters.

Sweeps the mean-CVaR policy across (estimation window, tactical band), runs a full
walk-forward at every grid point, and plots the out-of-sample Sharpe as a surface.

The point is not to find the single best cell. It is to see the SHAPE. A broad,
flat, high region is a robust choice: performance survives being wrong about the
exact parameter. A lone spike surrounded by lower ground is overfitting: it only
works at one setting, which will not repeat out of sample. The colouring marks the
robust plateau in blue and flags isolated peaks in amber.

Run:
    .venv/bin/python -m reporting.sharpe_terrain --tier balanced
Outputs reports/robustness/sharpe_terrain_<tier>.png and a small CSV of the grid.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from engine.allocate import allocate
from engine.io import load_config, load_returns
from backtest.walkforward import walk_forward
from backtest.metrics import perf_metrics

COST_BPS, TOL_BAND = 10.0, 0.05
OUT = Path(__file__).resolve().parent.parent / "reports" / "robustness"

# The two axes. Window is the estimation lookback in trading days; band is the
# tactical deviation allowed from the strategic weights. Both are genuine choices
# a user makes, and both are exactly where a backtest can be talked into overfitting.
WINDOWS = [252, 378, 504, 630, 756, 882, 1008]         # 1.0 to 4.0 years
BANDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]


def sharpe_at(ret, tier, window, band, regime="full"):
    """One walk-forward, returning out-of-sample Sharpe (or nan if it cannot run)."""
    cfg = load_config(variant="house", window=window)
    fn = lambda r: allocate(r, tier, cfg, "mean_cvar", False, regime, band)  # noqa: E731
    try:
        eq, *_ = walk_forward(ret, fn, window, COST_BPS, TOL_BAND)
        return perf_metrics(eq)["sharpe"]
    except Exception:
        return np.nan


def robustness_mask(Z, frac=0.90):
    """Split the grid into the robust plateau and isolated peaks.

    A cell is 'plateau' if it is within frac of the global best AND at least half of
    its neighbours are also strong. A cell that is high but stands alone is a peak,
    the overfitting signature.
    """
    thresh = np.nanmax(Z) * frac
    strong = Z >= thresh
    plateau = np.zeros_like(strong)
    rows, cols = Z.shape
    for i in range(rows):
        for j in range(cols):
            if not strong[i, j]:
                continue
            nb = strong[max(0, i - 1):i + 2, max(0, j - 1):j + 2]
            if nb.sum() - 1 >= 3:            # >=3 strong neighbours -> genuine plateau
                plateau[i, j] = True
    peaks = strong & ~plateau
    return plateau, peaks


def build(tier="balanced"):
    ret = load_returns(load_config()).dropna(how="any")
    Z = np.full((len(BANDS), len(WINDOWS)), np.nan)
    for i, b in enumerate(BANDS):
        for j, w in enumerate(WINDOWS):
            Z[i, j] = sharpe_at(ret, tier, w, b)
        print(f"  band {b:.2f}: " + " ".join(f"{z:4.2f}" for z in Z[i]))

    X, Y = np.meshgrid([w / 252 for w in WINDOWS], BANDS)   # window in years
    plateau, peaks = robustness_mask(Z)

    OUT.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(Z, index=[f"band_{b}" for b in BANDS],
                        columns=[f"win_{w}" for w in WINDOWS])
    grid.to_csv(OUT / f"sharpe_terrain_{tier}.csv")

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.82, linewidth=0.3,
                    edgecolor="#2b2b2b", rstride=1, cstride=1, antialiased=True)
    ax.contourf(X, Y, Z, zdir="z", offset=np.nanmin(Z) - 0.05,
                cmap=cm.viridis, alpha=0.5)

    # mark the two regimes on the surface itself
    for mask, colour, lab, size in ((plateau, "#1f6feb", "robust plateau", 55),
                                    (peaks, "#e3a008", "isolated peak (overfit)", 70)):
        pts = np.argwhere(mask)
        if len(pts):
            ax.scatter([WINDOWS[j] / 252 for i, j in pts],
                       [BANDS[i] for i, j in pts],
                       [Z[i, j] + 0.02 for i, j in pts],
                       c=colour, s=size, edgecolor="white", linewidth=0.6,
                       depthshade=False, label=lab)

    best = np.unravel_index(np.nanargmax(Z), Z.shape)
    ax.set_xlabel("estimation window (years)", labelpad=12, fontsize=10)
    ax.set_ylabel("tactical band", labelpad=12, fontsize=10)
    ax.set_zlabel("out-of-sample Sharpe", labelpad=8, fontsize=10)
    ax.set_title(f"Sharpe terrain, {tier} tier, mean-CVaR\n"
                 f"best {Z[best]:.2f} at {WINDOWS[best[1]]/252:.1f}y window, "
                 f"band {BANDS[best[0]]:.2f}", fontsize=12, pad=18)
    ax.view_init(elev=28, azim=-58)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.text(0.5, 0.02,
             "A broad blue region means the strategy is not sensitive to the exact "
             "parameter, which is what you want. An amber spike alone is overfitting.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    png = OUT / f"sharpe_terrain_{tier}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    span = np.nanmax(Z) - np.nanmin(Z)
    print(f"\nwrote {png.name}: best Sharpe {Z[best]:.2f}, spread across grid {span:.2f}, "
          f"{int(plateau.sum())} plateau cells, {int(peaks.sum())} isolated peaks")
    return Z


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="balanced",
                    choices=["conservative", "balanced", "growth", "aggressive"])
    build(ap.parse_args().tier)
