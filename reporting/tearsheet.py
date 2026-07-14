import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reporting import data, charts

REPORTS = Path(os.environ.get(
    "BEMO_REPORTS", Path(__file__).resolve().parent.parent / "reports"))
LEDGERS = REPORTS / "ledgers"


REF = "-"


def _rel(variant: str, tier: str, method: str, universe: str) -> Path:
    # universe / variant / tier_method ; references get their own folder, they have no tier
    if variant == REF:
        return Path(universe) / "reference" / method
    return Path(universe) / variant / f"{tier}_{method}"


def _subtitle(variant: str, tier: str, method: str, universe: str) -> str:
    if variant == REF:
        return f"{method}, {universe} universe. Covariance-only reference: tier-independent."
    return f"{tier.title()} tier, {method}, {variant} weights, {universe} universe."


def generate(tier: str, method: str, universe: str, variant: str = "house") -> Path:
    run = data.load_run(universe, tier, method, variant)
    m, value, rebal, attrib = run["metrics"], run["value"], run["rebal"], run["attrib"]
    start, end = value["value_eur"].iloc[0], value["value_eur"].iloc[-1]
    profit, cost = end - start, rebal["cost_eur"].sum()

    fig = plt.figure(figsize=(8.27, 11.69))
    gs = fig.add_gridspec(100, 12, left=0.09, right=0.95, top=0.96, bottom=0.04)

    axt = fig.add_subplot(gs[0:6, :]); axt.axis("off")
    axt.text(0, 0.7, "Bemo Europe allocation backtest", fontsize=15, fontweight="bold")
    axt.text(0, 0.15, f"{_subtitle(variant, tier, method, universe)} "
             f"{value['date'].iloc[0].date()} to {value['date'].iloc[-1].date()}. "
             f"Walk-forward, monthly rebalance, {data.CCY} 1,000,000 start, "
             "10 bps cost, 5% band.",
             fontsize=8.5, color="#333333")

    axm = fig.add_subplot(gs[7:20, :]); axm.axis("off")
    rows = [("Start value", data.fmt_eur(start), "Volatility", data.fmt_pct(m["ann_vol"])),
            ("End value", data.fmt_eur(end), "Sharpe", f"{m['sharpe']:.2f}"),
            ("Total profit", f"{data.fmt_eur(profit)} ({profit/start*100:.1f}%)",
             "Deflated Sharpe", f"{m['dsr']:.3f}"),
            ("CAGR", data.fmt_pct(m["ann_return"]), "Max drawdown", data.fmt_pct(m["max_dd"])),
            ("95% CVaR (daily)", data.fmt_pct(m["cvar95"]), "Total cost", data.fmt_eur(cost)),
            ("Annual turnover", f"{m['turnover']:.2f}x", "", "")]
    tbl = axm.table(cellText=rows, cellLoc="left", loc="center",
                    colWidths=[0.20, 0.30, 0.20, 0.30])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if cc in (0, 2):
            cell.set_text_props(color="#555555")

    axe = fig.add_subplot(gs[24:46, :])
    axd = fig.add_subplot(gs[46:54, :], sharex=axe)
    charts.draw_equity_drawdown(axe, axd, value)

    axo = fig.add_subplot(gs[58:84, 0:5]); charts.draw_donut(axo, data.latest_weights(rebal), data.bucket_map())
    axo.set_title("Latest allocation", fontsize=9)
    axp = fig.add_subplot(gs[58:84, 8:12]); charts.draw_pnl(axp, attrib)
    axp.set_title("Profit and loss by sleeve", fontsize=9)

    axs = fig.add_subplot(gs[87:100, :]); axs.axis("off")
    trade_dates = int(rebal[rebal["breached"]]["date"].nunique())
    reb_dates = int(rebal["date"].nunique())
    axs.text(0, 0.8, "Rebalance summary", fontsize=9, fontweight="bold")
    axs.text(0, 0.45, f"Rebalance checks: {reb_dates}.  Trades executed: {trade_dates}.  "
             f"Total traded cost: {data.fmt_eur(cost)}.  "
             f"Annual turnover: {m['turnover']:.2f}x.", fontsize=8.5, color="#333333")
    axs.text(0, 0.15, "Educational reference, not investment advice.", fontsize=7.5,
             color="#888888")

    out = (REPORTS / _rel(variant, tier, method, universe)).with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def export_csv(tier: str, method: str, universe: str, variant: str = "house") -> Path:
    out = (LEDGERS / _rel(variant, tier, method, universe)).with_suffix(".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    data.load_run(universe, tier, method, variant)["rebal"].to_csv(out, index=False)
    return out


def main() -> None:
    combos = data.options().sort_values(["universe", "variant", "method", "tier"])
    for _, r in combos.iterrows():
        generate(r["tier"], r["method"], r["universe"], r["variant"])
        export_csv(r["tier"], r["method"], r["universe"], r["variant"])
    print(f"Wrote {len(combos)} PDFs under {REPORTS} and {len(combos)} CSVs under {LEDGERS}")


if __name__ == "__main__":
    main()
