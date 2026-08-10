import os

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

BUCKET_COLORS = {"Equity": "#2b6cb0", "Fixed income": "#2c7a7b",
                 "Alternatives": "#b7791f", "Cash": "#718096"}
GAIN, LOSS, EQUITY_LINE, BAR = "#2f855a", "#c53030", "#2b6cb0", "#4a5568"
CCY = os.environ.get("BEMO_CCY", "EUR")

SHORT = {"Equity - Europe (home)": "Europe", "Equity - US": "US",
         "Equity - Developed Asia-Pacific / Japan": "Dev Asia-Pac",
         "Equity - Emerging / Asia": "EM Asia", "Equity - Thematic AI / automation": "AI theme",
         "Fixed income - EUR Govt / core": "EUR Govt", "Fixed income - EUR IG credit": "IG credit",
         "Fixed income - Inflation-linked": "Infl-linked", "Fixed income - High yield": "High yield",
         "Fixed income - EM debt": "EM debt", "Gold": "Gold",
         "Liquid alternatives / hedge funds": "Liquid alts",
         "Real assets / REITs / infrastructure": "Real assets", "Cash / EUR money market": "Cash"}
def _short(s: str) -> str:
    # chart label: short name, no "(home)" marker, and the EUR-flavoured sleeve keys
    # renamed on a non-EUR book, where they hold that book's own instruments
    out = SHORT.get(s, s.split(" - ")[-1]).replace(" (home)", "")
    if CCY != "EUR":
        for a in ("Govt", "IG credit", "money market"):
            out = out.replace(f"EUR {a}", f"{CCY} {a}")
    return out
_millions = FuncFormatter(lambda v, _: f"{v/1e6:.2f}M")
_thousands = FuncFormatter(lambda v, _: f"{v/1e3:.0f}k")


def period_heatmap(long, value_col, period_order, method_order, fmt=".1f",
                   centre_zero=True, height=None, title=None):
    """Method by period grid. One cell per (method, period), coloured by value_col.

    long needs columns: method_label, period, <value_col>, plus 'tip' for the tooltip.
    """
    scheme = "redyellowgreen" if centre_zero else "viridis"
    vals = long[value_col].dropna()
    lim = float(max(abs(vals.min()), abs(vals.max()))) if len(vals) else 1.0
    scale = (alt.Scale(scheme=scheme, domain=[-lim, lim])
             if centre_zero else alt.Scale(scheme=scheme))
    h = height or max(300, 30 * long.method_label.nunique())
    base = alt.Chart(long).encode(
        x=alt.X("period:N", title=None, sort=period_order,
                axis=alt.Axis(labelAngle=-35, labelFontSize=12, orient="top",
                              domain=False, ticks=False, labelPadding=4)),
        y=alt.Y("method_label:N", title=None, sort=method_order,
                axis=alt.Axis(labelFontSize=12.5, labelLimit=0, domain=False,
                              ticks=False, labelPadding=8)))
    cells = base.mark_rect(stroke="#fcfcfa", strokeWidth=2).encode(
        color=alt.Color(f"{value_col}:Q", scale=scale,
                        legend=alt.Legend(title=title, orient="right",
                                          gradientLength=170, gradientThickness=9)),
        tooltip=[alt.Tooltip("method_label:N", title="Method"),
                 alt.Tooltip("period:N", title="Period"),
                 alt.Tooltip("tip:N", title="Result")])
    text = base.mark_text(font=FONT, fontSize=11, fontWeight=600).encode(
        text=alt.Text(f"{value_col}:Q", format=fmt),
        color=alt.condition(f"abs(datum.{value_col}) > {lim * 0.55}",
                            alt.value("white"), alt.value("#1a1a1a")))
    return _polish((cells + text).properties(height=h))


def winners_chart(df, height=None):
    """Horizontal bar of how many periods each method won, best at the top."""
    h = height or max(200, 26 * len(df))
    tt = [alt.Tooltip("method_label:N", title="Method"),
          alt.Tooltip("wins:Q", title="Periods won"),
          alt.Tooltip("periods:N", title="Which")]
    y = alt.Y("method_label:N", title=None, sort="-x",
              axis=alt.Axis(labelFontSize=12.5, labelLimit=0, grid=False,
                            domain=False, ticks=False, labelPadding=8))
    bars = alt.Chart(df).mark_bar(color=EQUITY_LINE, size=13).encode(
        x=alt.X("wins:Q", title=None,
                axis=alt.Axis(tickMinStep=1, format="d", grid=True,
                              domain=False, ticks=False)),
        y=y, tooltip=tt)
    labs = alt.Chart(df).mark_text(align="left", dx=5, font=FONT, fontSize=11.5,
                                   color="#565b61").encode(
        x=alt.X("wins:Q", axis=None), y=y, text=alt.Text("wins:Q", format="d"),
        tooltip=tt)
    return _polish((bars + labs).properties(height=h))


def draw_equity_drawdown(ax_eq, ax_dd, value, bench=None) -> None:
    d = value["date"]
    ax_eq.plot(d, value["value_eur"], color=EQUITY_LINE, lw=1.4, label="This run", zorder=3)
    if bench is not None:
        b = bench.set_index("date").reindex(d)
        ax_eq.plot(d, b["value_eur"], color=BAR, lw=1.1, ls="--", label="60/40 benchmark")
        ax_dd.plot(d, b["drawdown"] * 100, color=BAR, lw=0.9, ls="--", zorder=3)
        ax_eq.legend(loc="upper left", fontsize=8, frameon=False)
    ax_eq.set_ylabel(f"Portfolio value {CCY}")
    ax_eq.yaxis.set_major_formatter(_millions)
    ax_eq.grid(alpha=0.25)
    ax_dd.fill_between(d, value["drawdown"] * 100, 0, color=LOSS, alpha=0.35)
    ax_dd.set_ylabel("Drawdown %")
    ax_dd.grid(alpha=0.25)


def draw_donut(ax, weights, buckets, threshold=0.004, scale=1.0) -> None:
    # label every held wedge on a leader line, de-collided per side. each label's text and
    # its leader line are coloured to match the slice, so the eye maps label to wedge directly.
    # the four bucket totals sit in the hole. scale enlarges the screen donut, not the A4 tearsheet
    big = scale > 1.3
    R = 1.08 if big else 0.80
    w = weights[weights > 1e-4].sort_values(ascending=False)
    colors = [BUCKET_COLORS.get(buckets.get(s, ""), "#999999") for s in w.index]
    wedges, _ = ax.pie(w.values, colors=colors, startangle=90, counterclock=False,
                       radius=R, wedgeprops=dict(width=0.30 * R / 0.80, edgecolor="white",
                                                 linewidth=1.3))
    # anchor each label at its own wedge angle so it sits next to the slice, all around the ring
    sides = {1: [], -1: []}
    for i, (s, v) in enumerate(w.items()):
        if v < threshold:
            continue
        ang = (wedges[i].theta1 + wedges[i].theta2) / 2
        x, y = R * np.cos(np.radians(ang)), R * np.sin(np.radians(ang))
        sides[1 if x >= 0 else -1].append([y, x, y, ang, f"{_short(s)} {v*100:.0f}%", colors[i]])

    # de-collide each side into a vertical stack, then lay the labels on an ellipse around the
    # ring so each sits out from its own wedge instead of in one straight column
    n_lab = sum(len(v) for v in sides.values())
    eff = scale if n_lab <= 9 else scale * (9.0 / n_lab) ** 0.34
    gap = (0.195 if big else 0.155) * min(eff, 1.6)
    ax_x = (1.85 if big else 1.42)
    extreme = R
    for side, items in sides.items():
        items.sort(key=lambda t: t[0], reverse=True)
        span = (len(items) - 1) * gap
        top_y = min(max((t[0] for t in items), default=0.0), span / 2 + 0.25)
        for k, it in enumerate(items):
            it[0] = top_y - k * gap
            extreme = max(extreme, abs(it[0]))
    b_rad = extreme + 0.05
    for side, items in sides.items():
        for ly, x, y, ang, txt, col in items:
            # x rides an ellipse of vertical half-height b_rad: labels near the top sit above the
            # donut, labels near a side sit out to that side. all wrap around the ring, not stacked.
            lx = side * max(ax_x * np.sqrt(max(0.0, 1.0 - (ly / b_rad) ** 2)), 0.5)
            ax.annotate(txt, xy=(x, y), xytext=(lx, ly), fontsize=7.4 * eff,
                        va="center", ha="left" if side > 0 else "right",
                        color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.9, alpha=0.7,
                                        connectionstyle="arc3,rad=0.0", shrinkA=2, shrinkB=3))

    # bucket totals as a coloured key under the donut, biggest bucket first
    tot = {}
    for s, v in w.items():
        tot[buckets.get(s, "")] = tot.get(buckets.get(s, ""), 0.0) + v
    order = [b for b in BUCKET_COLORS if tot.get(b, 0) > 0]
    handles = [Patch(color=BUCKET_COLORS[b]) for b in order]
    labels = [f"{b}  {tot[b]*100:.0f}%" for b in order]
    ax.set_aspect("equal")
    top = extreme + 0.2
    if big:
        # two rows keeps the key wide enough to read big while the ring stays large
        ncol = 2 if len(order) >= 3 else len(order)
        rows = (len(order) + ncol - 1) // ncol
        ax.set_xlim(-(ax_x + 0.95), ax_x + 0.95)
        ax.set_ylim(-(top + 0.35 + 0.42 * rows), top)
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                  ncol=ncol, fontsize=13.0, frameon=False,
                  handlelength=1.2, handletextpad=0.5, columnspacing=2.2, labelspacing=0.7)
    else:
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-(top + 0.32), max(1.15, top))
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
                  ncol=len(order), fontsize=5.0 * scale, frameon=False,
                  handlelength=0.9, handletextpad=0.35, columnspacing=1.0)


def draw_pnl(ax, attrib, scale=1.0) -> None:
    a = attrib.sort_values("pnl_eur")
    ax.barh([_short(s)[:22] for s in a["sleeve"]], a["pnl_eur"],
            color=[GAIN if v >= 0 else LOSS for v in a["pnl_eur"]])
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel(f"P&L {CCY}", fontsize=8 * scale)
    ax.xaxis.set_major_formatter(_thousands)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=7 * scale)


def draw_turnover(ax, rebal) -> None:
    to = rebal.groupby("date")["trade_pct"].apply(lambda x: np.abs(x).sum())
    hit = to[to > 1e-9]
    if hit.empty:
        ax.text(0.5, 0.5, "No trades executed", ha="center", va="center",
                fontsize=9, color="#666666", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.bar(hit.index, hit.values * 100, width=18, color=BAR)
    # pin the axis to the whole run, or a single bar autoscales into a two-week window
    span = pd.Timedelta(days=30)
    ax.set_xlim(to.index.min() - span, to.index.max() + span)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_ylabel("Turnover %")
    ax.grid(axis="y", alpha=0.25)


def equity_drawdown_fig(value, bench=None, figsize=(9, 4.5)):
    fig, (a, b) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    draw_equity_drawdown(a, b, value, bench)
    fig.tight_layout()
    return fig


def multi_equity_drawdown_fig(wide, dashed=None, figsize=(9, 4.7)):
    """The same two-panel layout as a single run, with one line per run.

    wide is a date-indexed frame of portfolio values, one column per run. The column
    named in dashed is drawn as a grey dashed line so the benchmark reads as a
    reference rather than as another candidate.
    """
    fig, (a, b) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    others = [c for c in wide.columns if c != dashed]
    palette = plt.get_cmap("tab10").colors
    for i, run in enumerate(others):
        s = wide[run].dropna()
        col = palette[i % len(palette)]
        a.plot(s.index, s.values, color=col, lw=1.3, label=run, zorder=3)
        b.plot(s.index, (s / s.cummax() - 1).values * 100, color=col, lw=0.9, zorder=3)
    if dashed in wide.columns:
        s = wide[dashed].dropna()
        a.plot(s.index, s.values, color=BAR, lw=1.1, ls="--", label=dashed, zorder=4)
        b.plot(s.index, (s / s.cummax() - 1).values * 100, color=BAR, lw=0.9, ls="--",
               zorder=4)
    a.set_ylabel(f"Portfolio value {CCY}")
    a.yaxis.set_major_formatter(_millions)
    a.grid(alpha=0.25)
    a.legend(loc="upper left", fontsize=7.5, frameon=False,
             ncol=2 if len(wide.columns) > 4 else 1)
    b.set_ylabel("Drawdown %")
    b.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def donut_fig(weights, buckets, figsize=(5, 5), scale=1.0):
    fig, ax = plt.subplots(figsize=figsize)
    draw_donut(ax, weights, buckets, scale=scale)
    fig.tight_layout()
    return fig


def pnl_fig(attrib, figsize=(7, 5), scale=1.0):
    fig, ax = plt.subplots(figsize=figsize)
    draw_pnl(ax, attrib, scale=scale)
    fig.tight_layout()
    return fig


def turnover_fig(rebal, figsize=(9, 2.6)):
    fig, ax = plt.subplots(figsize=figsize)
    draw_turnover(ax, rebal)
    fig.tight_layout()
    return fig


# ---------------- interactive charts (Altair) ----------------
# Altair twins of the matplotlib figures above: same inputs, same numbers, plus a
# hover that reads out the exact date and value. The matplotlib versions stay for
# anything that needs a static figure.

alt.data_transformers.disable_max_rows()   # the long tests hold ~6k daily rows

FONT = "-apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
_GRID, _AXTX, _DOM = "#e9e9e3", "#6b7280", "#cfcfc8"


def _polish(chart):
    """House axis style: horizontal grid only, no plot box, quiet grey axes."""
    return (chart
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont=FONT, titleFont=FONT, labelFontSize=12.5,
                            titleFontSize=12, titleFontWeight=500, labelColor=_AXTX,
                            titleColor=_AXTX, gridColor=_GRID, gridWidth=0.8,
                            domainColor=_DOM, tickColor=_DOM)
            .configure_legend(labelFont=FONT, titleFont=FONT, labelFontSize=12.5))


def _nearest_date():
    return alt.selection_point(fields=["date"], nearest=True, on="pointerover",
                               empty=False, clear="pointerout")


def equity_drawdown_alt(value, bench=None, eq_height=330, dd_height=120):
    """Interactive twin of equity_drawdown_fig. Same frames in, same numbers out.

    Returns (equity_chart, drawdown_chart): the caller stacks them, and each panel
    carries its own nearest-date hover with the full readout. Nothing is resampled.
    """
    df = pd.DataFrame({"date": pd.to_datetime(value["date"]),
                       "run": value["value_eur"].values,
                       "dd": value["drawdown"].values * 100.0})
    if bench is not None:
        b = bench.set_index("date").reindex(pd.DatetimeIndex(df["date"]))
        df["bench"] = b["value_eur"].values
        df["bench_dd"] = b["drawdown"].values * 100.0

    tt = [alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
          alt.Tooltip("run:Q", format=",.0f", title=f"This run, {CCY}")]
    if bench is not None:
        tt.append(alt.Tooltip("bench:Q", format=",.0f", title=f"60/40, {CCY}"))
    tt.append(alt.Tooltip("dd:Q", format=".1f", title="Drawdown %"))

    base = alt.Chart(df)
    x_top = alt.X("date:T", axis=None)
    x_bot = alt.X("date:T", axis=alt.Axis(grid=False, title=None, labelPadding=6,
                                          format="%b %Y", tickCount=6))
    y_eq = alt.Axis(title=None, grid=True, tickCount=5,
                    labelExpr="format(datum.value / 1e6, '.2f') + 'M'")
    eq_scale = alt.Scale(zero=False)

    hov = _nearest_date()
    layers = []
    if bench is not None:
        # Vega-Lite drops a merged axis if any layer sets axis:null, so the axis is
        # declared once on the first layer and the rest inherit the shared scale
        layers.append(base.mark_line(color=BAR, strokeWidth=1.1, strokeDash=[5, 3])
                      .encode(x=x_top, y=alt.Y("bench:Q", scale=eq_scale, axis=y_eq)))
        layers.append(base.mark_line(color=EQUITY_LINE, strokeWidth=1.8)
                      .encode(x=x_top, y=alt.Y("run:Q", scale=eq_scale)))
    else:
        layers.append(base.mark_line(color=EQUITY_LINE, strokeWidth=1.8)
                      .encode(x=x_top, y=alt.Y("run:Q", scale=eq_scale, axis=y_eq)))
    layers.append(base.mark_rule(color="#b6b9be", strokeWidth=1)
                  .encode(x=x_top).transform_filter(hov))
    layers.append(base.mark_point(size=64, filled=True, color=EQUITY_LINE)
                  .encode(x=x_top, y=alt.Y("run:Q", scale=eq_scale),
                          opacity=alt.condition(hov, alt.value(1), alt.value(0)),
                          tooltip=tt)
                  .add_params(hov))
    eq = alt.layer(*layers).properties(height=eq_height)

    hov2 = _nearest_date()
    y_dd = alt.Axis(title=None, grid=True, tickCount=4,
                    labelExpr="format(datum.value, '.0f') + '%'")
    dd_layers = [base.mark_area(color=LOSS, opacity=0.28)
                 .encode(x=x_bot, y=alt.Y("dd:Q", axis=y_dd)),
                 base.mark_line(color=LOSS, strokeWidth=0.9)
                 .encode(x=x_bot, y=alt.Y("dd:Q"))]
    if bench is not None:
        dd_layers.append(base.mark_line(color=BAR, strokeWidth=0.9, strokeDash=[5, 3])
                         .encode(x=x_bot, y=alt.Y("bench_dd:Q")))
    dd_layers.append(base.mark_rule(color="#b6b9be", strokeWidth=1)
                     .encode(x=x_bot).transform_filter(hov2))
    dd_layers.append(base.mark_point(size=54, filled=True, color=LOSS)
                     .encode(x=x_bot, y=alt.Y("dd:Q"),
                             opacity=alt.condition(hov2, alt.value(1), alt.value(0)),
                             tooltip=tt)
                     .add_params(hov2))
    dd = alt.layer(*dd_layers).properties(height=dd_height)
    return _polish(eq), _polish(dd)


def donut_alt(weights, buckets, height=330):
    """Interactive twin of donut_fig: hover a wedge for the exact weight.

    Wedges are coloured by bucket; on-ring labels are shown for wedges of 2% and
    up, and the holdings table below the chart carries every position exactly.
    """
    w = weights[weights > 1e-4].sort_values(ascending=False)
    df = pd.DataFrame({"sleeve": [_short(s) for s in w.index],
                       "bucket": [buckets.get(s, "") for s in w.index],
                       "w": w.values * 100.0})
    df["i"] = np.arange(len(df))
    df["label"] = np.where(df["w"] >= 2.0,
                           df["sleeve"] + " " +
                           df["w"].round(0).astype(int).astype(str) + "%", "")
    dom = [b for b in BUCKET_COLORS if b in set(df["bucket"])]
    rng = [BUCKET_COLORS[b] for b in dom]
    base = alt.Chart(df).encode(
        theta=alt.Theta("w:Q", stack=True),
        order=alt.Order("i:Q", sort="ascending"),
        color=alt.Color("bucket:N", scale=alt.Scale(domain=dom, range=rng),
                        legend=None))
    arcs = base.mark_arc(innerRadius=72, outerRadius=112, stroke="#ffffff",
                         strokeWidth=1.5).encode(
        tooltip=[alt.Tooltip("sleeve:N", title="Sleeve"),
                 alt.Tooltip("bucket:N", title="Bucket"),
                 alt.Tooltip("w:Q", format=".2f", title="Weight %")])
    text = base.mark_text(radius=128, font=FONT, fontSize=12,
                          fontWeight=600).encode(text="label:N")
    return _polish((arcs + text).properties(height=height))


def pnl_alt(attrib, height=None):
    """Interactive twin of pnl_fig: P&L per sleeve, gains green, losses red."""
    a = attrib.copy()
    a["sleeve_s"] = [_short(s) for s in a["sleeve"]]
    a["lab"] = [f"{v / 1e3:+,.0f}k" for v in a["pnl_eur"]]
    a = a.sort_values("pnl_eur", ascending=False)
    order = list(a["sleeve_s"])
    h = height or max(300, 27 * len(a))
    tt = [alt.Tooltip("sleeve_s:N", title="Sleeve"),
          alt.Tooltip("pnl_eur:Q", format=",.0f", title=f"P&L {CCY}")]
    y = alt.Y("sleeve_s:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=0, grid=False, domain=False, ticks=False,
                            labelPadding=8))
    x = alt.X("pnl_eur:Q", title=None,
              axis=alt.Axis(grid=True, domain=False, ticks=False,
                            labelExpr="format(datum.value / 1e3, ',.0f') + 'k'"))
    bars = alt.Chart(a).mark_bar(size=13).encode(
        y=y, x=x,
        color=alt.condition("datum.pnl_eur >= 0", alt.value(GAIN), alt.value(LOSS)),
        tooltip=tt)
    lab_p = (alt.Chart(a[a["pnl_eur"] >= 0])
             .mark_text(align="left", dx=5, font=FONT, fontSize=11.5, color="#565b61")
             .encode(y=y, x=alt.X("pnl_eur:Q", axis=None), text="lab:N"))
    lab_n = (alt.Chart(a[a["pnl_eur"] < 0])
             .mark_text(align="right", dx=-5, font=FONT, fontSize=11.5, color="#565b61")
             .encode(y=y, x=alt.X("pnl_eur:Q", axis=None), text="lab:N"))
    zero = (alt.Chart(pd.DataFrame({"z": [0.0]}))
            .mark_rule(color="#3f434a", strokeWidth=1)
            .encode(x=alt.X("z:Q", axis=None)))
    return _polish(alt.layer(bars, zero, lab_p, lab_n).properties(height=h))


def turnover_alt(rebal, height=160):
    """Interactive twin of turnover_fig. Returns None when nothing traded."""
    to = rebal.groupby("date")["trade_pct"].apply(lambda x: np.abs(x).sum()) * 100.0
    hit = to[to > 1e-7]
    if hit.empty:
        return None
    df = pd.DataFrame({"date": pd.to_datetime(hit.index), "to": hit.values})
    lo = (pd.to_datetime(to.index.min()) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    hi = (pd.to_datetime(to.index.max()) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    return _polish(alt.Chart(df).mark_bar(color=BAR, size=5).encode(
        x=alt.X("date:T", title=None, scale=alt.Scale(domain=[lo, hi]),
                axis=alt.Axis(grid=False)),
        y=alt.Y("to:Q", axis=alt.Axis(title="Turnover",
                                      labelExpr="format(datum.value, '.0f') + '%'")),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
                 alt.Tooltip("to:Q", format=".1f", title="Turnover %")])
        .properties(height=height))


# muted categorical palette for the compare view: house blue first, then gold, teal,
# brick, slate-blue, olive. The benchmark never takes one of these; it is always the
# dashed grey reference.
CAT_PALETTE = ["#2b6cb0", "#b7791f", "#2c7a7b", "#9b2c2c", "#5a6a92", "#8a8635",
               "#976a2e", "#3d7ea6"]


def multi_equity_alt(wide, dashed=None, eq_height=340, dd_height=120):
    """Interactive twin of multi_equity_drawdown_fig.

    Returns (equity_chart, drawdown_chart, legend) where legend is a list of
    (run, colour) pairs for the caller to render as chips. Same frame in, same
    numbers out; drawdowns come from the same cummax as the matplotlib twin.
    """
    runs = [c for c in wide.columns if c != dashed]
    frames = []
    for run in wide.columns:
        s = wide[run].dropna()
        frames.append(pd.DataFrame({"date": pd.to_datetime(s.index), "run": run,
                                    "value": s.values,
                                    "dd": (s / s.cummax() - 1).values * 100.0}))
    df = pd.concat(frames, ignore_index=True)
    order = runs + ([dashed] if dashed in wide.columns else [])
    colors = [CAT_PALETTE[i % len(CAT_PALETTE)] for i in range(len(runs))]
    dashes = [[1, 0]] * len(runs)
    if dashed in wide.columns:
        colors.append(BAR)
        dashes.append([5, 3])
    cscale = alt.Scale(domain=order, range=colors)
    dscale = alt.Scale(domain=order, range=dashes)
    tt = [alt.Tooltip("run:N", title="Run"),
          alt.Tooltip("date:T", format="%d %b %Y", title="Date"),
          alt.Tooltip("value:Q", format=",.0f", title=f"Value, {CCY}"),
          alt.Tooltip("dd:Q", format=".1f", title="Drawdown %")]
    base = alt.Chart(df)
    hov = _nearest_date()
    eq = alt.layer(
        base.mark_line(strokeWidth=1.5).encode(
            x=alt.X("date:T", axis=None),
            y=alt.Y("value:Q", scale=alt.Scale(zero=False),
                    axis=alt.Axis(title=None, grid=True, tickCount=5,
                                  labelExpr="format(datum.value / 1e6, '.2f') + 'M'")),
            color=alt.Color("run:N", scale=cscale, legend=None),
            strokeDash=alt.StrokeDash("run:N", scale=dscale, legend=None)),
        base.mark_rule(color="#b6b9be", strokeWidth=1)
            .encode(x=alt.X("date:T", axis=None)).transform_filter(hov),
        base.mark_point(size=54, filled=True).encode(
            x=alt.X("date:T", axis=None), y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("run:N", scale=cscale, legend=None),
            opacity=alt.condition(hov, alt.value(1), alt.value(0)),
            tooltip=tt).add_params(hov),
    ).properties(height=eq_height)
    dd = base.mark_line(strokeWidth=1.0).encode(
        x=alt.X("date:T", axis=alt.Axis(grid=False, title=None, labelPadding=6,
                                        format="%b %Y", tickCount=6)),
        y=alt.Y("dd:Q", axis=alt.Axis(title=None, grid=True, tickCount=4,
                                      labelExpr="format(datum.value, '.0f') + '%'")),
        color=alt.Color("run:N", scale=cscale, legend=None),
        strokeDash=alt.StrokeDash("run:N", scale=dscale, legend=None),
        tooltip=tt).properties(height=dd_height)
    return _polish(eq), _polish(dd), list(zip(order, colors))
