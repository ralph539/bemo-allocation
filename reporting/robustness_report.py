"""Robustness report: the engine across universes, decades and crises, as one PDF.

Reads the tables written by backtest.robustness and renders reports/robustness/
robustness.pdf. Run with .venv/bin/python -m reporting.robustness_report
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from engine.io import DB, ROOT, TIERS

PARQUET = Path(os.environ.get("BEMO_PARQUET", DB.parent / "parquet"))
OUT = ROOT / "reports" / "robustness" / "robustness.pdf"
NAVY = "#1f3a5f"
GREEN = "#2E7D5B"
RED = "#C0504D"
ENGINE = "mean_cvar"
BENCH = "bench_60_40"
A4 = (8.27, 11.69)

UNIVERSE_DESC = {
    "eur_book": "Live EUR UCITS book, 14 sleeves",
    "eur_book_noAI": "EUR book without the AI sleeve (history to 2008)",
    "us_book_noAI": "US ETF proxies without AI (history to 2008)",
    "us_long": "Long-history US proxies (index funds, gold futures) to 2000",
    "us_long_stocks": "Long-history mix: thematic sleeve as AAPL/MSFT/AMZN basket",
}


def _load():
    m = pd.read_parquet(PARQUET / "robustness_metrics.parquet")
    c = pd.read_parquet(PARQUET / "robustness_curves.parquet")
    return m, c


def _pct(x):
    return "-" if pd.isna(x) else f"{x * 100:.1f}%"


def _title_page(pdf, m):
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.90, "Bemo Allocation Engine", ha="center", size=24,
             weight="bold", color=NAVY)
    fig.text(0.5, 0.865, "Robustness report: universes, decades and crises",
             ha="center", size=13, color="#333333")
    eng = m[(m.method == ENGINE) & (m.kind != "full")]
    n, w = len(eng), int(eng.beats_bench.sum())
    full = m[(m.method == ENGINE) & (m.kind == "full")]
    pos = int((full.ret > 0).sum())
    fig.text(0.5, 0.79, f"{w} of {n}", ha="center", size=40, weight="bold",
             color=GREEN if w / max(n, 1) >= 0.5 else RED)
    fig.text(0.5, 0.765, "sub-period cells where the engine beats 60/40 "
             "(same universe, same window)", ha="center", size=10, color="#555555")
    fig.text(0.5, 0.72, f"{pos} of {len(full)} full-history runs end in profit",
             ha="center", size=11, color="#333333")
    y = 0.64
    fig.text(0.08, y, "Universes tested", size=13, weight="bold", color=NAVY)
    y -= 0.03
    for u, d in UNIVERSE_DESC.items():
        sub = m[(m.universe == u) & (m.kind == "full") & (m.method == ENGINE)]
        span = (f"OOS {sub.start.iloc[0][:7]} to {sub.end.iloc[0][:7]}"
                if len(sub) else "no data")
        fig.text(0.10, y, f"{u}", size=10, weight="bold", color="#222222")
        fig.text(0.32, y, d, size=9, color="#444444")
        fig.text(0.80, y, span, size=8.5, color="#666666")
        y -= 0.025
    y -= 0.02
    fig.text(0.08, y, "Method", size=13, weight="bold", color=NAVY)
    fig.text(0.08, y - 0.02,
             "Each universe is walked forward once per tier with the house engine\n"
             "(mean-CVaR, Ledoit-Wolf shrinkage, all caps and bands), the static\n"
             "strategic weights, and a 60/40 benchmark. Monthly rebalancing, 10 bps\n"
             "cost, 5% tolerance band, 3-year estimation window. Sub-period and\n"
             "crisis numbers are slices of the out-of-sample equity curve, never\n"
             "re-fitted. The thematic sleeve uses QQQ before the AI ETF era, and an\n"
             "equal-weight AAPL/MSFT/AMZN basket in the ETF-plus-stocks variant.",
             size=9.5, color="#333333", va="top")
    fig.text(0.5, 0.05, "Educational reference, not investment advice.",
             ha="center", size=9, style="italic", color="#777777")
    pdf.savefig(fig)
    plt.close(fig)


def _cell_table(pdf, m, tier):
    eng = m[(m.method == ENGINE) & (m.tier == tier) & (m.kind != "full")]
    if eng.empty:
        return
    piv_e = eng.pivot_table(index="universe", columns="period", values="excess_ret",
                            observed=True)
    piv_r = eng.pivot_table(index="universe", columns="period", values="ret",
                            observed=True)
    order = [p for p in ["2003-2007", "2008-2012", "2013-2017", "2018-2022",
                         "2023-2026", "GFC 2008", "Euro crisis 2011", "COVID 2020",
                         "Rate shock 2022"] if p in piv_e.columns]
    piv_e, piv_r = piv_e[order], piv_r[order]
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(f"Engine vs 60/40 by period, {tier} tier\n"
                 "cell = engine total return (excess vs 60/40 underneath)",
                 size=14, color=NAVY, weight="bold", pad=18)
    n_r, n_c = piv_e.shape
    for j, p in enumerate(piv_e.columns):
        ax.text(j + 0.5, n_r + 0.25, p.replace(" ", "\n", 1), ha="center",
                va="bottom", size=9, weight="bold", color=NAVY)
    for i, u in enumerate(piv_e.index):
        yy = n_r - 1 - i
        ax.text(-0.15, yy + 0.5, u, ha="right", va="center", size=9,
                weight="bold", color="#222222")
        for j, p in enumerate(piv_e.columns):
            e, r = piv_e.loc[u, p], piv_r.loc[u, p]
            if pd.isna(e):
                ax.add_patch(plt.Rectangle((j, yy), 0.96, 0.96, color="#F0F2F4"))
                continue
            col = "#DCEBE2" if e > 0 else "#F3D9D7"
            ax.add_patch(plt.Rectangle((j, yy), 0.96, 0.96, color=col))
            ax.text(j + 0.48, yy + 0.60, _pct(r), ha="center", va="center",
                    size=9, weight="bold", color="#1B2A38")
            ax.text(j + 0.48, yy + 0.28, f"{'+' if e > 0 else ''}{_pct(e)}",
                    ha="center", va="center", size=8,
                    color=GREEN if e > 0 else RED)
    ax.set_xlim(-2.6, n_c)
    ax.set_ylim(-0.2, n_r + 1.1)
    fig.text(0.5, 0.04, "Green cell = engine beat 60/40 in that window. "
             "Crisis windows are peak-to-trough slices.", ha="center", size=9,
             color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def _crisis_page(pdf, m):
    cr = m[(m.kind == "crisis") & (m.method == ENGINE)]
    if cr.empty:
        return
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.94, "Crisis behavior", ha="center", size=16, weight="bold",
             color=NAVY)
    fig.text(0.5, 0.915, "engine return and max drawdown vs 60/40, per tier",
             ha="center", size=10, color="#555555")
    y = 0.86
    for period in cr.period.unique():
        sub = cr[cr.period == period]
        fig.text(0.08, y, period, size=12, weight="bold", color=NAVY)
        fig.text(0.30, y, f"({sub.start.iloc[0]} to {sub.end.iloc[0]})",
                 size=9, color="#666666")
        y -= 0.022
        fig.text(0.10, y, "universe", size=8.5, weight="bold", color="#555555")
        for k, t in enumerate(TIERS):
            fig.text(0.34 + k * 0.13, y, t[:4], size=8.5, weight="bold",
                     color="#555555")
        fig.text(0.86, y, "60/40", size=8.5, weight="bold", color="#555555")
        y -= 0.018
        for u in sub.universe.unique():
            us = sub[sub.universe == u]
            fig.text(0.10, y, u, size=8.5, color="#222222")
            for k, t in enumerate(TIERS):
                r = us[us.tier == t]
                if len(r):
                    v = r.ret.iloc[0]
                    fig.text(0.34 + k * 0.13, y, _pct(v), size=8.5,
                             color=GREEN if v > r.bench_ret.iloc[0] else RED)
            b = us.bench_ret.iloc[0] if len(us) else np.nan
            fig.text(0.86, y, _pct(b), size=8.5, color="#444444")
            y -= 0.016
        y -= 0.018
    fig.text(0.5, 0.05, "Green = better than 60/40 in the same window.",
             ha="center", size=9, color="#666666")
    pdf.savefig(fig)
    plt.close(fig)


def _curves_page(pdf, m, c):
    unis = [u for u in UNIVERSE_DESC if u in set(c.universe)]
    fig, axes = plt.subplots(len(unis), 1, figsize=A4, sharex=False)
    if len(unis) == 1:
        axes = [axes]
    fig.suptitle("Out-of-sample equity curves, balanced tier (log scale)",
                 size=13, weight="bold", color=NAVY, y=0.985)
    for ax, u in zip(np.atleast_1d(axes), unis):
        for meth, col, lab in [(ENGINE, NAVY, "engine"),
                               ("strategic", "#B8862B", "strategic"),
                               (BENCH, "#888888", "60/40")]:
            tier = "-" if meth == BENCH else "balanced"
            sub = c[(c.universe == u) & (c.method == meth) & (c.tier == tier)]
            if sub.empty:
                continue
            ax.plot(pd.to_datetime(sub.date), sub.equity, lw=1.1, color=col,
                    label=lab, ls="--" if meth == BENCH else "-")
        ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, NullFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}x"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(u, size=9.5, color="#222222", loc="left")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    pdf.savefig(fig)
    plt.close(fig)


def _full_table(pdf, m):
    full = m[(m.kind == "full") & (m.method.isin([ENGINE, "strategic", BENCH]))]
    cols = ["universe", "tier", "method", "start", "end", "ann_return", "ann_vol",
            "sharpe", "max_dd", "turnover", "excess_ret"]
    d = full[cols].copy().sort_values(["universe", "method", "tier"])
    for cc in ["ann_return", "ann_vol", "max_dd", "excess_ret"]:
        d[cc] = d[cc].map(_pct)
    d["sharpe"] = d.sharpe.round(2)
    d["turnover"] = d.turnover.round(2)
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    fig.text(0.5, 0.96, "Full-history results per universe (whole OOS span)",
             ha="center", size=13, weight="bold", color=NAVY)
    tbl = ax.table(cellText=d.values, colLabels=[c.replace("_", " ") for c in cols],
                   cellLoc="center", bbox=[0.0, 0.0, 1.0, 0.94])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.8)
    for (r, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDE3E8")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    m, c = _load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        _title_page(pdf, m)
        for tier in ["balanced", "aggressive"]:
            _cell_table(pdf, m, tier)
        _crisis_page(pdf, m)
        _curves_page(pdf, m, c)
        _full_table(pdf, m)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
