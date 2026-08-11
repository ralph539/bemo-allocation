import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from html import escape as _esc

import numpy as np
import pandas as pd
import streamlit as st

from backtest.metrics import benchmark_metrics
from reporting import data, charts, periods

st.set_page_config(page_title="Bemo allocation backtest", layout="wide")
# Presentation layer. One block of CSS: typographic hierarchy (serif display for the
# run title, tabular figures everywhere), the KPI band, section labels and the
# holdings table. Colour stays semantic: green/red for gain/loss, bucket colours for
# asset classes, everything else neutral ink on off-white.
st.markdown("""
<style>
[data-testid="stSidebar"] {min-width: 340px; max-width: 340px; background: #f2f2ec;
  border-right: 1px solid #e2e2db;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {font-size: 11px;
  font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: #6b7280;}
[data-testid="stSidebar"] [data-baseweb="select"] > div {border-radius: 0;
  border: 1px solid #cfcfc8; background: #ffffff;}
.bemo-brand {font-size: 15px; font-weight: 700; letter-spacing: .14em; color: #14161a;}
.bemo-brandsub {font-size: 12.5px; color: #6b7280; margin: 2px 0 16px;}
/* the view switcher, as a nav list rather than radio buttons */
.st-key-navview [role="radiogroup"] {gap: 2px;}
.st-key-navview [role="radiogroup"] > label {width: 100%; margin: 0; padding: 6px 10px;
  border-left: 3px solid transparent;}
.st-key-navview [role="radiogroup"] > label > span:first-child {display: none;}
.st-key-navview [data-testid="stRadioOption"] > div > div > div:not([data-testid="stMarkdownContainer"]) {display: none;}
.st-key-navview [role="radiogroup"] > label p {font-size: 13.5px; color: #4b5157;}
.st-key-navview [role="radiogroup"] > label:has(input:checked) {border-left-color: #14161a;
  background: #ffffff;}
.st-key-navview [role="radiogroup"] > label:has(input:checked) p {color: #14161a;
  font-weight: 700;}
html, body {font-variant-numeric: lining-nums tabular-nums;}
.block-container {padding-top: 2.4rem; max-width: 1400px;}
[data-testid="stVerticalBlock"] [data-testid="stVerticalBlock"] {gap: 0.4rem;}
#vg-tooltip-element {font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial,
  sans-serif; font-size: 12.5px; border: 1px solid #d8d8d1; border-radius: 2px;
  color: #1b1e23;}
#vg-tooltip-element td.key {color: #6b7280;}
.bemo-kicker {font-size: 12px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: #6b7280; margin-bottom: 2px;}
.bemo-title {font-family: Georgia, "Times New Roman", serif; font-size: 34px;
  font-weight: 700; line-height: 1.12; color: #14161a; letter-spacing: -.005em;}
.bemo-desc {font-size: 14.5px; line-height: 1.45; color: #4b5157; max-width: 72ch;
  margin: 4px 0 0;}
.bemo-hero {display: flex; flex-wrap: wrap; gap: 12px 64px; margin-top: 18px;
  padding: 16px 0 14px; border-top: 3px solid #14161a; animation: bemo-in .18s ease-out;}
.bemo-hero .lbl, .bemo-strip .lbl {font-size: 11px; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase; color: #6b7280;
  margin-bottom: 4px; white-space: nowrap; cursor: default;}
.bemo-hero .big {font-size: 40px; font-weight: 700; line-height: 1; color: #14161a;
  letter-spacing: -.01em;}
.bemo-hero .big.dim {color: #6b7280; font-weight: 600;}
.bemo-hero .sub {font-size: 13.5px; color: #4b5157; margin-top: 6px;}
.pos {color: #2f855a; font-weight: 600;} .neg {color: #c53030; font-weight: 600;}
.bemo-strip {display: flex; flex-wrap: wrap; gap: 10px 40px; padding: 12px 0;
  border-top: 1px solid #e2e2db; border-bottom: 1px solid #e2e2db;
  animation: bemo-in .18s ease-out;}
.bemo-strip .val {font-size: 21px; font-weight: 600; color: #14161a; margin-top: 1px;}
.bemo-note {font-size: 12.5px; color: #6b7280; margin-top: 6px;}
.bemo-sec {display: flex; justify-content: space-between; align-items: baseline;
  font-size: 13px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  color: #14161a; margin: 30px 0 4px;}
.bemo-sec .hint {font-weight: 400; letter-spacing: 0; text-transform: none;
  font-size: 12.5px; color: #6b7280;}
.bemo-legend {display: flex; gap: 20px; font-size: 12.5px; color: #4b5157;}
.bemo-legend .sw {display: inline-block; width: 18px; border-top: 2.5px solid;
  margin-right: 6px; vertical-align: 3px;}
.bemo-legend .sw.dash {border-top-style: dashed; border-top-width: 2px;}
.bemo-chips {display: flex; flex-wrap: wrap; justify-content: center; gap: 8px 22px;
  font-size: 13px; font-weight: 600; color: #33373d; margin-top: 2px;}
.dot {display: inline-block; width: 9px; height: 9px; border-radius: 50%;
  margin-right: 6px;}
table.bemo-hold {width: 100%; border-collapse: collapse; font-size: 14px;
  animation: bemo-in .18s ease-out;}
table.bemo-hold th {text-align: left; font-size: 11px; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase; color: #6b7280;
  padding: 6px 12px 6px 0; border-bottom: 2px solid #14161a;}
table.bemo-hold th.num {text-align: right;}
table.bemo-hold td {padding: 6px 12px 6px 0; border-bottom: 1px solid #ebebe4;
  color: #1b1e23; vertical-align: middle; line-height: 1.25;}
table.bemo-hold td.num {text-align: right; font-weight: 600; white-space: nowrap;}
table.bemo-hold td.etf {font-family: ui-monospace, "SF Mono", Consolas, Menlo,
  monospace; font-size: 12.5px; color: #4b5157;}
table.bemo-hold td.bar {width: 32%;}
table.bemo-hold td.bar div {height: 5px;}
table.bemo-hold tbody tr:hover {background: #f4f4ee;}
table.bemo-hold td.dim {color: #6b7280; font-weight: 400;}
table.bemo-hold td.pos {color: #2f855a;}
table.bemo-hold td.neg {color: #c53030;}
table.bemo-hold td.zero {color: #9aa0a6;}
table.bemo-trades {font-size: 13px;}
table.bemo-trades th, table.bemo-trades td {padding-right: 18px;}
table.bemo-hold tr.bench td {background: #eef1f5;}
table.bemo-hold td .cbar {height: 4px; background: #2b6cb0; margin-top: 3px;}
table.bemo-hold td.yes {color: #2f855a; font-weight: 700;}
table.bemo-hold td.no {color: #9aa0a6;}
.bemo-scroll {max-height: 640px; overflow-y: auto;}
.bemo-scroll table.bemo-hold {border-collapse: separate; border-spacing: 0;}
.bemo-scroll table.bemo-hold thead th {position: sticky; top: 0; background: #fcfcfa;
  z-index: 2; border-bottom: none; box-shadow: 0 2px 0 #14161a;}
@keyframes bemo-in {from {opacity: 0;} to {opacity: 1;}}
</style>""", unsafe_allow_html=True)

REF = "-"
BENCH = "bench_60_40"
TRADE_ROWS = 400          # beyond this the trade log goes to the CSV
TIERS = ["conservative", "balanced", "growth", "aggressive"]
VARIANTS = ["house", "peer_mid", "peer_low", "peer_high", "us_tilt"]

# every method, grouped the way a reader should think about them. Order is the display order.
GROUPS = {
    "Baseline": ["strategic"],
    "House core": ["mean_cvar"],
    "Offense (tilt harder)": ["max_ret_cvarcap", "mean_variance", "trend_tilt",
                              "trend_tilt_relaxed", "black_litterman_mom", "mean_cvar_anchored"],
    "Risk-off overlays": ["vol_target", "regime_breaker", "dual_momentum"],
    "Combinations": ["cvarcap_relaxed", "cvarcap_breaker", "cvarcap_dualmom",
                     "trend_breaker", "trend_dualmom"],
    "Cap regimes": ["mean_cvar_no_band", "mean_cvar_no_sleeve_caps",
                    "mean_cvar_equity_band_only", "mean_cvar_uncapped",
                    "mean_cvar_band15", "mean_cvar_relaxed",
                    "cvarcap_no_band", "cvarcap_no_sleeve_caps",
                    "cvarcap_equity_band_only", "cvarcap_uncapped"],
    "References (risk only)": ["risk_parity", "hrp", "min_variance", "max_sharpe",
                               "max_diversification", "equal_weight", "inverse_vol", BENCH],
}
REF_ONLY = set(GROUPS["References (risk only)"])
FAMILY_OF = {m: g for g, ms in GROUPS.items() for m in ms}
TIER_METHODS = [m for g, ms in GROUPS.items() for m in ms if m not in REF_ONLY]

NICE = {
    "strategic": "Buy and hold (strategic)",
    "mean_cvar": "Mean-CVaR (house core)",
    "max_ret_cvarcap": "Max return under CVaR cap",
    "mean_variance": "Mean-variance (Markowitz)",
    "trend_tilt": "Trend tilt (momentum mu)",
    "trend_tilt_relaxed": "Trend tilt, relaxed caps",
    "black_litterman_mom": "Black-Litterman, momentum views",
    "mean_cvar_anchored": "Mean-CVaR, anchored (low turnover)",
    "vol_target": "Vol target (scale to 10%)",
    "regime_breaker": "Regime breaker (200d MA)",
    "dual_momentum": "Dual momentum (vs cash)",
    "cvarcap_relaxed": "Max-CVaR + relaxed caps",
    "cvarcap_breaker": "Max-CVaR + regime breaker",
    "cvarcap_dualmom": "Max-CVaR + dual momentum",
    "trend_breaker": "Trend tilt + regime breaker",
    "trend_dualmom": "Trend tilt + dual momentum",
    "mean_cvar_no_band": "No tactical band",
    "mean_cvar_no_sleeve_caps": "No sleeve caps",
    "mean_cvar_equity_band_only": "Equity band only",
    "mean_cvar_uncapped": "Uncapped",
    "mean_cvar_band15": "Wider band 15%",
    "mean_cvar_relaxed": "Relaxed caps",
    "cvarcap_no_band": "CVaR-budget: no band",
    "cvarcap_no_sleeve_caps": "CVaR-budget: no sleeve caps",
    "cvarcap_equity_band_only": "CVaR-budget: equity band only",
    "cvarcap_uncapped": "CVaR-budget only",
    "risk_parity": "Risk parity",
    "hrp": "Hierarchical risk parity",
    "min_variance": "Min variance",
    "max_sharpe": "Max Sharpe",
    "max_diversification": "Max diversification",
    "equal_weight": "Equal weight",
    "inverse_vol": "Inverse volatility",
    BENCH: "60/40 benchmark",
}
DESC = {
    "strategic": "Hold the tier's target weights, rebalanced back on drift. The policy portfolio.",
    "mean_cvar": "The incumbent. Minimise the 95% tail loss subject to the strategic return "
                 "target, inside every cap.",
    "max_ret_cvarcap": "Maximise expected return subject to CVaR <= the strategic portfolio's own "
                       "tail budget. Best active optimiser this round.",
    "mean_variance": "Markowitz twin of the house method: minimise variance instead of tail loss, "
                     "same caps.",
    "trend_tilt": "Boost expected return toward recent winners (12-1m momentum), then optimise "
                  "inside the caps.",
    "trend_tilt_relaxed": "Trend tilt with the tactical band dropped and the EM and AI caps widened.",
    "black_litterman_mom": "Feed point-in-time momentum z-scores as Black-Litterman views, then run "
                           "mean-CVaR. No hindsight.",
    "mean_cvar_anchored": "Mean-CVaR with a ridge pull back to the strategic weights. Lower turnover.",
    "vol_target": "Scale the whole book to a 10% volatility target, park the rest in cash.",
    "regime_breaker": "If equity is below its 200-day average, move half of equity into cash and gold.",
    "dual_momentum": "Sell any sleeve whose momentum is below cash to cash. The 2022 shield.",
    "cvarcap_relaxed": "Max-return-under-CVaR-cap run with the relaxed cap regime.",
    "cvarcap_breaker": "Max-return-under-CVaR-cap with the regime breaker layered on top.",
    "cvarcap_dualmom": "Max-return-under-CVaR-cap with dual momentum on top. Near flat in 2022.",
    "trend_breaker": "Trend tilt with the regime breaker on top.",
    "trend_dualmom": "Trend tilt with dual momentum on top.",
    "mean_cvar_no_band": "Mean-CVaR with the +/-10 point tactical band removed.",
    "mean_cvar_no_sleeve_caps": "Mean-CVaR with the per-sleeve caps removed (equity band kept).",
    "mean_cvar_equity_band_only": "Mean-CVaR keeping only the equity band.",
    "mean_cvar_uncapped": "Mean-CVaR long-only, all caps removed. Shows what the caps cost.",
    "mean_cvar_band15": "Mean-CVaR with a wider 15% tactical band.",
    "mean_cvar_relaxed": "Mean-CVaR under the relaxed cap regime (no band, EM and AI widened).",
    "cvarcap_no_band": "Max return under the CVaR budget, tactical band removed.",
    "cvarcap_no_sleeve_caps": "Max return under the CVaR budget, per-sleeve caps removed.",
    "cvarcap_equity_band_only": "Max return under the CVaR budget, only the equity band kept.",
    "cvarcap_uncapped": "Max return with every mandate cap removed. Only the tail budget "
                        "(CVaR <= the policy's own) remains, because that constraint is the "
                        "method itself. Shows what the fences add beyond the budget.",
    "risk_parity": "Equal risk contribution. Risk structure only, ignores tier and strategic weights.",
    "hrp": "Hierarchical risk parity. Risk structure only.",
    "min_variance": "Classical Markowitz minimum variance, long-only, uncapped.",
    "max_sharpe": "Classical Markowitz tangency, long-only, uncapped.",
    "max_diversification": "Maximise the diversification ratio. Risk structure only.",
    "equal_weight": "Equal weight across sleeves. The naive baseline that is hard to beat.",
    "inverse_vol": "Weight each sleeve by the inverse of its volatility.",
    BENCH: "Passive 60% equity, 40% bonds. The benchmark every run is judged against.",
}
# short row labels for the comparison grid, where the full names do not fit
SHORT = {
    "strategic": "Buy and hold", "mean_cvar": "Mean-CVaR", "max_ret_cvarcap": "Max ret, CVaR cap",
    "mean_variance": "Mean-variance", "trend_tilt": "Trend tilt",
    "black_litterman_mom": "Black-Litterman", "mean_cvar_anchored": "Mean-CVaR anchored",
    "vol_target": "Vol target", "regime_breaker": "Regime breaker",
    "dual_momentum": "Dual momentum", "risk_parity": "Risk parity", "hrp": "HRP",
    "min_variance": "Min variance", "max_sharpe": "Max Sharpe",
    "max_diversification": "Max diversification", "equal_weight": "Equal weight",
    "inverse_vol": "Inverse vol", BENCH: "60/40 benchmark",
    "cvarcap_relaxed": "CVaRcap relaxed", "cvarcap_breaker": "CVaRcap + breaker",
    "cvarcap_dualmom": "CVaRcap + dualmom", "trend_breaker": "Trend + breaker",
    "trend_dualmom": "Trend + dualmom", "trend_tilt_relaxed": "Trend relaxed",
    "mean_cvar_no_band": "MC no band", "mean_cvar_no_sleeve_caps": "MC no sleeves",
    "mean_cvar_equity_band_only": "MC eq-band only", "mean_cvar_uncapped": "MC uncapped",
    "mean_cvar_band15": "MC band 15", "mean_cvar_relaxed": "MC relaxed",
    "mean_cvar_anchored": "MC anchored", "vol_target": "Vol target",
    "cvarcap_no_band": "CVaRcap no band", "cvarcap_no_sleeve_caps": "CVaRcap no sleeves",
    "cvarcap_equity_band_only": "CVaRcap eq-band", "cvarcap_uncapped": "CVaRcap budget-only",
}

UNIVERSE_HELP_ALL = (
    "**Real book** is the 14 sleeves the bank actually holds. Its panel only becomes "
    "complete on 09 May 2019, when the managed-futures sleeve (DBMF) launched, and the "
    "engine needs 756 trading days of history before its first trade. That is why the "
    "backtest starts in mid-2022.\n\n"
    "**Long test** rebuilds the same 14 sleeves from US proxies that go back to 2003, so "
    "the engine can be checked across more market cycles. The two versions differ only in "
    "what stands in for the AI sleeve: QQQ, or an equal-weight AAPL, MSFT and AMZN basket.")

COLS = {"variant": "Variant", "tier": "Tier", "method": "Method", "ann_return": "CAGR %",
        "ann_vol": "Vol %", "sharpe": "Sharpe", "dsr": "Confidence", "beats_bench": "Beats 60/40",
        "excess_return": "vs 60/40 %", "excess_sharpe": "vs 60/40 Sharpe", "max_dd": "Max DD %",
        "s2022_ret": "2022 %", "s2022_dd": "2022 DD %",
        "cvar95": "CVaR95 %", "beta": "Beta", "alpha": "Alpha %", "tracking_error": "TE %",
        "info_ratio": "Info ratio", "turnover": "Turnover"}
STRESS_COLS = ["2022 %", "2022 DD %"]
PCT = ["CAGR %", "Vol %", "Max DD %", "CVaR95 %", "Alpha %", "TE %", "vs 60/40 %",
       "2022 %", "2022 DD %", "Return %"]
RATIO = ["Sharpe", "Beta", "Info ratio", "Turnover", "vs 60/40 Sharpe"]


from engine.io import DB as _DB


def _db_stamp() -> float:
    return os.path.getmtime(_DB)


@st.cache_data(show_spinner=False)
def load_curves(db_stamp: float = None):
    return data.all_curves()


@st.cache_data(show_spinner=False)
def load_rf(universe: str, db_stamp: float = None):
    return data.risk_free(universe)


@st.cache_data(show_spinner=False)
def period_board(universe: str, period: str, db_stamp: float = None) -> pd.DataFrame:
    """Every run in a universe, scored over one window, computed from the curves.

    Sub-period numbers are slices of a single walk-forward, never a re-fit, so they
    stay honest out-of-sample. Sharpe is an excess-return Sharpe over the universe's
    own money-market sleeve, matching the full-sample metrics in the store.
    """
    cur = load_curves(db_stamp)
    cur = cur[cur.universe == universe]
    rf = load_rf(universe, db_stamp)
    rf = None if rf is None or rf.empty else rf
    rows = []
    for (src, var, tier, meth), g in cur.groupby(["source", "variant", "tier", "method"],
                                                 observed=True):
        eq = g.set_index("date")["equity"].sort_index()
        m = periods.metrics(eq, period, rf)
        if np.isnan(m["ret"]):
            continue
        rows.append(dict(source=src, variant=var, tier=tier, method=meth, **m))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    bench = df[df.method == BENCH]
    b_ret = float(bench.ret.iloc[0]) if len(bench) else np.nan
    b_shp = float(bench.sharpe.iloc[0]) if len(bench) else np.nan
    df["excess_return"] = df.ret - b_ret
    df["excess_sharpe"] = df.sharpe - b_shp
    df["beats_bench"] = df.excess_return > 0
    df["run"] = [data.run_label(v, t, m) for v, t, m in zip(df.variant, df.tier, df.method)]
    return df


board = data.scoreboard()      # full-sample metrics, used for the Deflated Sharpe note
_STAMP = _db_stamp()
CURVES = load_curves(_STAMP)
# Only the books that carry the AI sleeve. The no-AI variants and the EUR lab
# re-runs were diagnostics; showing all seven made the picker unreadable.
SHOW_UNIVERSES = ["full_14", "us_long", "us_long_stocks"]
UNIVERSE_NAME = {
    "full_14": "Real book (2022-2026)",
    "us_long": "Long test, AI = QQQ (2003-2026)",
    "us_long_stocks": "Long test, AI = mega-caps (2003-2026)",
}
ALL_UNIVERSES = [u for u in SHOW_UNIVERSES if u in set(CURVES.universe.unique())] \
    or sorted(CURVES.universe.unique())
LIVE_UNIVERSES = set(CURVES.loc[CURVES.source == "live", "universe"].unique())
# universes with enough history to be split into decades
LONG_HISTORY = [u for u in ALL_UNIVERSES
                if len(data.universe_dates(CURVES, u)) > 2500]
# the lab re-runs two books the live backtest already covers in full detail
LIVE_EQUIV = {"eur_book": "full_14", "eur_book_noAI": "no_AI_13"}


def available(universe, method, col):
    # only offer combinations that actually exist in the store
    v = CURVES[(CURVES.universe == universe) & (CURVES.method == method)][col].unique()
    return [x for x in v if x != REF]


def groups_for(universe):
    have = set(CURVES[CURVES.universe == universe].method.unique())
    out = {}
    for g, ms in GROUPS.items():
        keep = [m for m in ms if m in have]
        if keep:
            out[g] = keep
    return out


def confidence_note(n_trials, bench_dsr):
    with st.expander("Why does nothing reach 95%?"):
        st.markdown(
            f"We tried **{int(n_trials)} configurations**. Try enough ideas and one looks brilliant by "
            f"luck alone. Confidence corrects for that: it is the chance the edge is real rather than "
            f"the luckiest of {int(n_trials)} tries.\n\n"
            f"The passive 60/40 benchmark scores **{bench_dsr*100:.1f}%** and fails too. The "
            f"same portfolio would score far higher judged as a single idea: the penalty is "
            f"the search, not the portfolio.\n\n"
            f"**A short window cannot prove an edge over cash once you have tried "
            f"{int(n_trials)} things.** The fix is more history, not more ideas.")


def _blend(fig):
    """Drop the figure's white ground so it sits on the page, not in a box."""
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.set_facecolor("none")
    return fig


def _cell(label, value, help_txt=""):
    t = f' title="{help_txt}"' if help_txt else ""
    return (f'<div class="cell"><div class="lbl"{t}>{label}</div>'
            f'<div class="val">{value}</div></div>')


def bemo_table(df, fmt=None, red_neg=(), bar_col=None, bench_mask=None, scroll=False,
               mono_cols=()):
    """House table: micro-headers, hairline rows, right-aligned tabular numerals.

    Presentation only: values arrive already computed. red_neg columns paint
    negatives red, bar_col adds an in-cell magnitude bar, bench_mask tints the
    benchmark row, scroll wraps the table with a sticky header.
    """
    fmt = fmt or {}
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bmax = float(df[bar_col].max()) if bar_col in num and len(df) else 0.0
    head = "".join(f'<th class="num">{_esc(str(c))}</th>' if c in num
                   else f'<th>{_esc(str(c))}</th>' for c in df.columns)
    body = []
    for i in range(len(df)):
        r = df.iloc[i]
        cls = ' class="bench"' if bench_mask is not None and bool(bench_mask.iloc[i]) else ""
        tds = []
        for c in df.columns:
            v = r[c]
            if c in num:
                txt = "" if pd.isna(v) else fmt.get(c, "{:,.2f}").format(v)
                sty = ' style="color:#c53030"' if c in red_neg and pd.notna(v) and v < 0 else ""
                bar = ""
                if c == bar_col and pd.notna(v) and bmax > 0:
                    bar = (f'<div class="cbar" '
                           f'style="width:{max(float(v), 0.0) / bmax * 100:.0f}%"></div>')
                tds.append(f'<td class="num"{sty}>{txt}{bar}</td>')
            elif str(v) in ("yes", "no") and str(c).startswith("Beats"):
                tds.append(f'<td class="{v}">{v}</td>')
            elif c in mono_cols:
                tds.append(f'<td class="etf">{_esc(str(v))}</td>')
            else:
                tds.append(f'<td>{_esc(str(v))}</td>')
        body.append(f"<tr{cls}>{''.join(tds)}</tr>")
    tbl = (f'<table class="bemo-hold"><thead><tr>{head}</tr></thead>'
           f'<tbody>{"".join(body)}</tbody></table>')
    return f'<div class="bemo-scroll">{tbl}</div>' if scroll else tbl


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<div class="bemo-brand">BEMO</div><div class="bemo-brandsub">'
                'Multi-asset allocation backtests</div>', unsafe_allow_html=True)
    view = st.radio("View", ["Single run", "Compare runs", "Full scoreboard",
                             "Which optimizer won"],
                    label_visibility="collapsed", key="navview")

    def _span_label(u):
        d = data.universe_dates(CURVES, u)
        return UNIVERSE_NAME.get(u, f"{u}  ({d.min():%Y} to {d.max():%Y})")

    universe = st.selectbox("Universe", ALL_UNIVERSES, format_func=_span_label,
                            index=ALL_UNIVERSES.index("full_14")
                            if "full_14" in ALL_UNIVERSES else 0,
                            help=UNIVERSE_HELP_ALL)

    udates = data.universe_dates(CURVES, universe)
    period_opts = periods.available(udates)
    period = st.selectbox("Period", period_opts,
                          format_func=lambda p: periods.label(p, udates),
                          help="Every number and chart on the page is cut to this window.")
    p_start, p_end = periods.bounds(period, udates)
    st.caption(f"{p_start:%d %b %Y} to {p_end:%d %b %Y}"
               + (f". {periods.NOTE[period]}" if period in periods.NOTE else ""))
    # a short universe cannot offer the earlier decades: say so rather than let the
    # short list look like a bug
    if len(period_opts) <= 2 and LONG_HISTORY:
        st.info("Only the long tests reach back to 2003. Pick one above for the "
                "earlier decades and crises.")

pboard = period_board(universe, period, _STAMP)
is_full = period == periods.FULL
is_live = universe in LIVE_UNIVERSES

# ---------------- single run ----------------
if view == "Single run":
    grps = groups_for(universe)
    with st.sidebar:
        group = st.radio("Group", list(grps), format_func=str)
        method = st.radio("Config", grps[group], format_func=lambda m: NICE.get(m, m))
        if method in REF_ONLY:
            variant, tier = REF, REF
        else:
            tiers = [t for t in TIERS if t in available(universe, method, "tier")]
            variants = [v for v in VARIANTS if v in available(universe, method, "variant")]
            tier = st.radio("Tier", tiers) if tiers else REF
            variant = st.radio("Weight variant", variants) if variants else "house"

    # One rendering path for every universe and every window: the figure, the metric
    # block and the tables are built from the walk-forward curve, so a sub-period looks
    # exactly like the full run. The stored ledger adds holdings and trades where it exists.
    def _curve(v, t, m):
        g = CURVES[(CURVES.universe == universe) & (CURVES.variant == v)
                   & (CURVES.tier == t) & (CURVES.method == m)]
        if g.empty:
            return None
        eq = g.set_index("date")["equity"].sort_index()
        s = periods.slice_curve(eq, period)
        return None if s.empty else s

    mine = _curve(variant, tier, method)
    bench_eq = _curve(REF, REF, BENCH)
    if mine is None:
        st.error(f"No data for {NICE.get(method, method)} on {variant} weights, {tier} "
                 f"tier, {universe}, over {periods.label(period, udates)}.")
        st.stop()

    row = pboard[(pboard.variant == variant) & (pboard.tier == tier)
                 & (pboard.method == method)]
    row = row.iloc[0] if len(row) else None
    rf_u = load_rf(universe, _STAMP)
    rf_u = None if rf_u is None or rf_u.empty else rf_u
    bm = (benchmark_metrics(mine, bench_eq, rf_u) if bench_eq is not None
          else {"beta": float("nan"), "alpha": float("nan")})

    START = 1_000_000.0
    end_val = float(mine.iloc[-1]) * START
    profit = end_val - START

    ttl = (NICE.get(method, method) if variant == REF
           else f"{tier.title()} tier, {NICE.get(method, method)}")
    kick = (f"{_esc(UNIVERSE_NAME.get(universe, universe))}"
            + ("" if variant == REF else f" | {_esc(variant)} weights")
            + f" | {p_start:%d %b %Y} to {p_end:%d %b %Y}"
            + f" | {len(mine):,} trading days")
    st.markdown(f'<div class="bemo-kicker">{kick}</div>'
                f'<div class="bemo-title">{_esc(ttl)}</div>'
                f'<p class="bemo-desc">{_esc(DESC.get(method, ""))}</p>',
                unsafe_allow_html=True)
    # CAGR and volatility are blanked below this length; Sharpe and alpha are annualised
    # too, so say plainly that they are not rates here rather than let a 38-day Sharpe
    # of 3.9 read as an edge
    if len(mine) < periods.MIN_ANNUALISE:
        st.warning(f"Only {len(mine)} trading days. Compare total return and drawdown "
                   "here, not CAGR, Sharpe or alpha.")

    # every universe now stores a ledger: the funded book from the live backtest, the
    # long tests from the robustness lab, which used to discard the rows it computed
    rebal = attrib = None
    turn_cell, trade_note = "", ""
    try:
        run = data.load_run(universe, tier, method, variant)
        all_rebal, attrib = run["rebal"], run["attrib"]
        if all_rebal.empty:
            raise KeyError("no ledger rows stored for this run")
        rebal = all_rebal[(all_rebal["date"] >= p_start) & (all_rebal["date"] <= p_end)]
        if rebal.empty:
            rebal = all_rebal
        traded = int(rebal[rebal["breached"]]["date"].nunique())
        cost = rebal["cost_eur"].sum()
        turn_cell = _cell("Turnover", f"{run['metrics']['turnover']:.2f}x",
                          "How much of the book it trades a year, over the full run.")
        trade_note = (f"Traded {traded} of {rebal['date'].nunique()} monthly checks, "
                      f"cost {data.fmt_eur(cost)}")
    except KeyError:
        pass

    # the money outcome leads (what happened to the million, and would 60/40 have done
    # better); rates and ratios follow in one dense stat strip. Same numbers as before,
    # ranked by how often the audience asks for them.
    pos = "pos" if profit >= 0 else "neg"
    hero = (f'<div><div class="lbl" title="Value at the end of the window">Final value</div>'
            f'<div class="big">{data.fmt_eur(end_val)}</div>'
            f'<div class="sub"><span class="{pos}">{"+" if profit >= 0 else "-"}'
            f'{data.fmt_eur(abs(profit))} ({(end_val / START - 1) * 100:+.1f}%)</span>'
            f' on {data.fmt_eur(START)} invested</div></div>')
    if bench_eq is not None and method != BENCH:
        b_end = float(bench_eq.iloc[-1]) * START
        gap = end_val - b_end
        gpos = "pos" if gap >= 0 else "neg"
        # a percentage-point gap, not a return: over 20 years the two totals differ by
        # more than 100 points and calling that a percentage reads as nonsense
        pts_help = ("Total return minus the benchmark's, in percentage points, "
                    "over this window.")
        pts = ("" if row is None else
               f' | <span title="{pts_help}">{row["excess_return"] * 100:+.1f} pts</span>')
        hero += (f'<div><div class="lbl" title="The same stake in the passive benchmark, '
                 f'same window">Same stake in 60/40</div>'
                 f'<div class="big dim">{data.fmt_eur(b_end)}</div>'
                 f'<div class="sub"><span class="{gpos}">{"+" if gap >= 0 else "-"}'
                 f'{data.fmt_eur(abs(gap))}</span> for this run{pts}</div></div>')

    short = row is None or pd.isna(row["ann_return"])
    cells = "".join([
        _cell("CAGR", "n/a" if short else data.fmt_pct(row["ann_return"]),
              "Yearly compounded growth rate. Not shown on a window under 120 "
              "trading days, where annualising is meaningless."),
        _cell("Volatility", "n/a" if row is None or pd.isna(row["ann_vol"])
              else data.fmt_pct(row["ann_vol"])),
        _cell("Sharpe", f"{row['sharpe']:.2f}" if row is not None else "n/a",
              "Return above cash per unit of volatility."),
        _cell("Max drawdown", data.fmt_pct(float((mine / mine.cummax() - 1).min())),
              "Worst fall from a peak inside this window."),
        _cell("CVaR 95", data.fmt_pct(row["cvar95"]) if row is not None else "n/a",
              "Average of the worst 5% of days."),
        _cell("Beta", "n/a" if pd.isna(bm["beta"]) else f"{bm['beta']:.2f}",
              "Share of the benchmark's risk carried."),
        _cell("Alpha", "n/a" if pd.isna(bm["alpha"]) else data.fmt_pct(bm["alpha"]),
              "Return that market exposure cannot explain."),
    ]) + turn_cell
    st.markdown(f'<div class="bemo-hero">{hero}</div>'
                f'<div class="bemo-strip">{cells}</div>'
                + (f'<div class="bemo-note">{trade_note}</div>' if trade_note else ""),
                unsafe_allow_html=True)

    def _value_frame(eq):
        v = eq * START
        return pd.DataFrame({"date": v.index, "value_eur": v.values,
                             "drawdown": (v / v.cummax() - 1).values})

    # same two panels as the report figure, drawn in Altair so any date can be hovered
    # for the exact values. charts.equity_drawdown_fig stays for the static tearsheet.
    bench_frame = None if method == BENCH or bench_eq is None else _value_frame(bench_eq)
    eq_chart, dd_chart = charts.equity_drawdown_alt(_value_frame(mine), bench_frame)
    leg = '<span><span class="sw" style="border-color:#2b6cb0"></span>This run</span>'
    if bench_frame is not None:
        leg += ('<span><span class="sw dash" style="border-color:#4a5568"></span>'
                '60/40 benchmark</span>')
    st.markdown(f'<div class="bemo-sec">Growth of {data.fmt_eur(START)}'
                '<span class="hint">hover any date for the exact values</span></div>'
                f'<div class="bemo-legend">{leg}</div>', unsafe_allow_html=True)
    st.altair_chart(eq_chart, use_container_width=True, theme=None)
    st.altair_chart(dd_chart, use_container_width=True, theme=None)

    if rebal is not None:
        last_date = pd.to_datetime(rebal["date"]).max()
        c = st.columns(2, gap="large")
        with c[0]:
            st.markdown('<div class="bemo-sec">Allocation<span class="hint">latest '
                        f'rebalance, {last_date:%b %Y}</span></div>',
                        unsafe_allow_html=True)
            w = data.latest_weights(rebal)
            # the matplotlib donut keeps its leader-line de-collision: the Altair twin
            # lays labels on the ring itself, so long ones hide behind the arc
            st.pyplot(_blend(charts.donut_fig(w, data.bucket_map(),
                                              figsize=(6.6, 6.0), scale=1.8)))
        with c[1]:
            st.markdown('<div class="bemo-sec">Profit and loss by sleeve'
                        f'<span class="hint">{data.CCY}, whole run</span></div>',
                        unsafe_allow_html=True)
            if is_full:
                st.altair_chart(charts.pnl_alt(attrib), use_container_width=True,
                                theme=None)
            else:
                st.markdown('<div class="bemo-note">Only available over full history. '
                            'Switch Period to see it.</div>', unsafe_allow_html=True)

        hold = data.holdings(rebal, universe)
        st.markdown(f'<div class="bemo-sec">Holdings<span class="hint">on '
                    f'{last_date:%d %b %Y}, largest first</span></div>',
                    unsafe_allow_html=True)
        bcol = charts.BUCKET_COLORS
        maxw = float(hold["Weight %"].max())
        rows = []
        for _, hr in hold.iterrows():
            col = bcol.get(hr["Bucket"], "#718096")
            rows.append(
                f'<tr><td>{_esc(str(hr["Sleeve"]))}</td>'
                f'<td><span class="dot" style="background:{col}"></span>'
                f'{_esc(str(hr["Bucket"]))}</td>'
                f'<td class="etf">{_esc(str(hr["Instrument"]))}</td>'
                f'<td class="num">{hr["Weight %"]:.2f}</td>'
                f'<td class="bar"><div style="width:{hr["Weight %"] / maxw * 100:.1f}%;'
                f'background:{col}"></div></td></tr>')
        st.markdown('<table class="bemo-hold"><thead><tr><th>Sleeve</th><th>Bucket</th>'
                    '<th>Instrument</th><th class="num">Weight %</th><th></th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)

        with st.expander("Trades: every date, every sleeve"):
            tchart = charts.turnover_alt(rebal)
            if tchart is None:
                st.caption("No trades executed in this window.")
            else:
                st.altair_chart(tchart, use_container_width=True, theme=None)
            # the log holds one row per sleeve per monthly check, and on most checks
            # nothing traded. Showing the executed rows first keeps the table readable;
            # the toggle and the CSV still carry every check.
            only_traded = st.checkbox("Only dates where it traded", value=True)
            rb = rebal[rebal["breached"]] if only_traded else rebal
            fr = data.format_rebal(rb)
            st.caption(f"{len(fr):,} rows"
                       + ("" if only_traded else ", including checks where nothing traded"))
            if len(fr) > TRADE_ROWS:
                st.caption(f"Showing the first {TRADE_ROWS:,}. Download the CSV for all "
                           f"{len(fr):,}.")
            rows = []
            for _, tr in fr.head(TRADE_ROWS).iterrows():
                tp = float(tr["Trade %"])
                cls = "pos" if tp > 0 else ("neg" if tp < 0 else "zero")
                arrow = "buy" if tp > 0 else ("sell" if tp < 0 else "")
                rows.append(
                    f'<tr><td class="etf">{tr["Date"]:%Y-%m-%d}</td>'
                    f'<td>{_esc(str(tr["Sleeve"]))}</td>'
                    f'<td class="num">{tr["Target %"]:.2f}</td>'
                    f'<td class="num dim">{tr["Before %"]:.2f}</td>'
                    f'<td class="num">{tr["After %"]:.2f}</td>'
                    f'<td class="num {cls}">{tp:+.2f}</td>'
                    f'<td class="num dim">{arrow}</td>'
                    f'<td class="num">{tr["Trade EUR"]:,.0f}</td>'
                    f'<td class="num dim">{tr["Cost EUR"]:,.0f}</td></tr>')
            st.markdown(
                '<div class="bemo-scroll"><table class="bemo-hold bemo-trades"><thead><tr>'
                '<th>Date</th><th>Sleeve</th><th class="num">Target %</th>'
                '<th class="num">Before %</th><th class="num">After %</th>'
                '<th class="num">Trade %</th><th class="num"></th>'
                f'<th class="num">Trade {data.CCY}</th>'
                f'<th class="num">Cost {data.CCY}</th></tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>',
                unsafe_allow_html=True)
            st.download_button("Download CSV", rebal.to_csv(index=False),
                               f"rebalance_{variant}_{tier}_{method}_{universe}.csv",
                               "text/csv")
    else:
        st.info("No rebalance ledger stored for this run, so there are no holdings or "
                "trades to show. Every other number on the page is unaffected.")

# ---------------- compare ----------------
elif view == "Compare runs":
    st.markdown(
        f'<div class="bemo-kicker">{_esc(UNIVERSE_NAME.get(universe, universe))}'
        f' | {p_start:%d %b %Y} to {p_end:%d %b %Y}</div>'
        '<div class="bemo-title">Compare equity curves</div>'
        f'<p class="bemo-desc">Every run starts at {data.fmt_eur(1_000_000)} on '
        f'{p_start:%d %b %Y}. The 60/40 benchmark is the dashed grey reference.</p>',
        unsafe_allow_html=True)

    cur = CURVES[CURVES.universe == universe].copy()
    cur["run"] = [data.run_label(v, t, m) for v, t, m in zip(cur.variant, cur.tier, cur.method)]
    avail = sorted(cur.run.unique())
    bench_run = BENCH if BENCH in avail else None

    def house(tier, methods):
        got = [f"house/{tier}/{m}" for m in methods if f"house/{tier}/{m}" in avail]
        return ([bench_run] if bench_run else []) + got

    preset = st.pills("Preset", ["Engine vs baseline vs 60/40", "Offense optimisers",
                                 "Risk-off overlays", "All the references", "Custom"],
                      default="Engine vs baseline vs 60/40") or "Custom"
    if preset == "All the references":
        picked = [c for c in GROUPS["References (risk only)"] if c in avail]
    elif preset == "Custom":
        f = st.columns(3)
        fv = f[0].multiselect("Variant", VARIANTS, default=["house"])
        ft = f[1].multiselect("Tier", TIERS, default=["balanced"])
        fm = f[2].multiselect("Method", TIER_METHODS, default=["mean_cvar", "strategic"],
                              format_func=lambda m: NICE.get(m, m))
        picked = [c for c in avail if c == BENCH or (c.count("/") == 2
                  and c.split("/")[0] in fv and c.split("/")[1] in ft
                  and c.split("/")[2] in fm)]
    else:
        tier = st.select_slider("Tier", TIERS, value="balanced")
        if preset == "Engine vs baseline vs 60/40":
            picked = house(tier, ["mean_cvar", "strategic"]) + \
                (["risk_parity"] if "risk_parity" in avail else [])
        elif preset == "Offense optimisers":
            picked = house(tier, ["mean_cvar", "max_ret_cvarcap", "trend_tilt",
                                  "black_litterman_mom"])
        else:
            picked = house(tier, ["mean_cvar", "vol_target", "regime_breaker", "dual_momentum"])

    picked = st.multiselect("Runs on the chart", avail, default=sorted(set(picked)))
    if not picked:
        st.info("Pick at least one run.")
    else:
        cols = {}
        for run in picked:
            g = cur[cur.run == run]
            eq = g.set_index("date")["equity"].sort_index()
            s = periods.slice_curve(eq, period)
            if not s.empty:
                cols[run] = s * 1_000_000.0
        wide = pd.DataFrame(cols).dropna(how="all")
        if wide.empty:
            st.info("Those runs have no data in this window.")
        else:
            eq_c, dd_c, legend = charts.multi_equity_alt(wide, dashed=BENCH)
            leg = "".join(
                f'<span><span class="sw{" dash" if r == BENCH else ""}" '
                f'style="border-color:{c}"></span>{_esc(NICE.get(r, r))}</span>'
                for r, c in legend)
            st.markdown('<div class="bemo-sec">Growth of '
                        f'{data.fmt_eur(1_000_000)}<span class="hint">hover any date '
                        'for the exact values</span></div>'
                        f'<div class="bemo-legend">{leg}</div>', unsafe_allow_html=True)
            st.altair_chart(eq_c, use_container_width=True, theme=None)
            st.altair_chart(dd_c, use_container_width=True, theme=None)
            b = pboard[pboard.run.isin(picked)].copy()
            b = b.rename(columns={"ret": "Return %"}).sort_values("sharpe", ascending=False)
            show = b[["run", "Return %", "ann_return", "ann_vol", "sharpe",
                      "max_dd", "excess_return", "beats_bench"]].rename(
                columns={"run": "Run", "ann_return": "CAGR %", "ann_vol": "Vol %",
                         "sharpe": "Sharpe", "max_dd": "Max DD %",
                         "excess_return": "vs 60/40 %", "beats_bench": "Beats 60/40"})
            for c in ["Return %", "CAGR %", "Vol %", "Max DD %", "vs 60/40 %"]:
                show[c] = show[c] * 100
            show["Beats 60/40"] = show["Beats 60/40"].map({True: "yes", False: "no"})
            st.markdown('<div class="bemo-sec">These runs, scored<span class="hint">'
                        'sorted by Sharpe, best first</span></div>',
                        unsafe_allow_html=True)
            st.markdown(bemo_table(show, red_neg=("Return %", "vs 60/40 %"),
                                   bar_col="Sharpe", bench_mask=(b.method == BENCH),
                                   mono_cols=("Run",)), unsafe_allow_html=True)

# ---------------- scoreboard ----------------
elif view == "Full scoreboard":
    st.markdown(
        f'<div class="bemo-kicker">{_esc(UNIVERSE_NAME.get(universe, universe))}'
        f' | {p_start:%d %b %Y} to {p_end:%d %b %Y}</div>'
        '<div class="bemo-title">Full scoreboard</div>'
        '<p class="bemo-desc">Every configuration in the store, scored over the '
        'selected window.</p>', unsafe_allow_html=True)
    if pboard.empty:
        st.info("No runs cover this window.")
        st.stop()

    b = pboard.copy()
    best = b[b.method != BENCH].loc[lambda d: d.sharpe.idxmax()]
    cells = "".join([
        _cell("Runs", f"{len(b)}"),
        _cell("Beat 60/40", f"{int(b.beats_bench.sum())} of {len(b)}",
              "Higher total return than the benchmark over this window."),
        _cell("Best Sharpe", f"{best.sharpe:.2f}"),
    ])
    st.markdown(f'<div class="bemo-strip" style="border-top:3px solid #14161a;'
                f'margin-top:18px">{cells}</div>'
                f'<div class="bemo-note">Best Sharpe: {_esc(str(best["run"]))}, '
                f'the maximum over {len(b)} correlated runs, so it is the luckiest draw '
                'as much as the best method.</div>', unsafe_allow_html=True)

    if is_full and is_live:
        brow = board[(board.universe == universe) & (board.method == BENCH)]
        if len(brow):
            confidence_note(brow.iloc[0].n_trials, brow.iloc[0].dsr)
    elif not is_full:
        st.caption("Confidence is a full-history number, so it is not shown on a window.")

    all_groups = list(groups_for(universe))
    f = st.columns([2.4, 1.1, 1.2])
    families = f[0].pills("Show", all_groups, selection_mode="multi", default=all_groups)
    only_beat = f[1].toggle("Only beat 60/40", value=False)
    sort_by = f[2].selectbox("Sort by", ["Sharpe", "vs 60/40 %", "Return %", "CAGR %",
                                         "Max DD %"])

    b = b[b.method.map(FAMILY_OF).isin(families or all_groups)]
    if only_beat:
        b = b[b.beats_bench]
    if b.empty:
        st.info("No run matches those filters.")
    else:
        show = b.rename(columns={"ret": "Return %", "variant": "Variant", "tier": "Tier",
                                 "method": "Method", "ann_return": "CAGR %",
                                 "ann_vol": "Vol %", "sharpe": "Sharpe",
                                 "max_dd": "Max DD %", "cvar95": "CVaR95 %",
                                 "excess_return": "vs 60/40 %",
                                 "beats_bench": "Beats 60/40"})
        keep = ["Variant", "Tier", "Method", "Return %", "CAGR %", "Vol %", "Sharpe",
                "Max DD %", "CVaR95 %", "vs 60/40 %", "Beats 60/40"]
        show = show[keep]
        for c in ["Return %", "CAGR %", "Vol %", "Max DD %", "CVaR95 %", "vs 60/40 %"]:
            show[c] = show[c] * 100
        show["Beats 60/40"] = show["Beats 60/40"].map({True: "yes", False: "no"})
        # every sort option here is higher-is-better, drawdown included (it is negative)
        show = show.sort_values(sort_by, ascending=False)
        # display copy only: the CSV below keeps the raw method keys
        disp = show.copy()
        disp["Method"] = disp["Method"].map(lambda m: NICE.get(m, m))
        st.markdown(bemo_table(disp, red_neg=("Return %", "vs 60/40 %"),
                               bar_col="Sharpe",
                               bench_mask=(show["Method"] == BENCH), scroll=True),
                    unsafe_allow_html=True)
        st.download_button("Download CSV", show.to_csv(index=False),
                           f"scoreboard_{universe}_{period}.csv", "text/csv")

# ---------------- which optimizer won ----------------
else:
    st.markdown(
        '<div class="bemo-kicker">All windows at once | the sidebar Period '
        'does not apply here</div>'
        '<div class="bemo-title">Which optimizer won, and when</div>'
        '<p class="bemo-desc">Every method, ranked in every window.</p>',
        unsafe_allow_html=True)

    c = st.columns([1.3, 1, 1.3, 1.1])
    uni = c[0].selectbox("Universe", ALL_UNIVERSES,
                         format_func=lambda u: UNIVERSE_NAME.get(u, u),
                         index=ALL_UNIVERSES.index(universe),
                         help="Only the long tests reach back far enough to compare "
                              "decades.")
    tier_c = c[1].selectbox("Tier", TIERS, index=1)
    metric = c[2].selectbox("Rank by", ["Excess return vs 60/40", "Total return",
                                        "Sharpe", "Max drawdown"])
    var_opts = [v for v in VARIANTS
                if v in CURVES[CURVES.universe == uni].variant.unique()]
    variant_c = c[3].selectbox("Weight variant", var_opts or ["house"],
                               help="Tier-dependent methods exist once per weight "
                                    "variant. Comparing one variant at a time keeps "
                                    "one row per method.")
    if len(data.universe_dates(CURVES, uni)) < 2500:
        d0 = data.universe_dates(CURVES, uni)
        st.info(f"This one only spans {d0.min():%Y} to {d0.max():%Y}, so there are few "
                "windows. Pick a long test above for the full 2003 to 2026 picture.")

    udates_c = data.universe_dates(CURVES, uni)
    per_list = [p for p in periods.available(udates_c) if p != periods.FULL]
    if not per_list:
        st.info("Not enough history to split into windows.")
        st.stop()

    recs = []
    for p in [periods.FULL] + per_list:
        pb = period_board(uni, p, _STAMP)
        if pb.empty:
            continue
        # one weight variant of each tier-dependent method, plus the tier-free references
        keep = pb[((pb.tier == tier_c) & (pb.variant == variant_c)
                   & (~pb.method.isin(REF_ONLY))) | (pb.method.isin(REF_ONLY))]
        for _, r in keep.iterrows():
            recs.append(dict(period=p, method=r.method,
                             method_label=SHORT.get(r.method, NICE.get(r.method, r.method)),
                             ret=r.ret, sharpe=r.sharpe, max_dd=r.max_dd,
                             excess=r.excess_return))
    grid = pd.DataFrame(recs)
    if grid.empty:
        st.info("No comparable runs in this universe.")
        st.stop()

    COL = {"Excess return vs 60/40": ("excess", True, ".0f", 100),
           "Total return": ("ret", True, ".0f", 100),
           "Sharpe": ("sharpe", True, ".2f", 1),
           "Max drawdown": ("max_dd", True, ".0f", 100)}
    col, centre, fmt, mult = COL[metric]
    grid["value"] = grid[col] * mult
    grid["tip"] = grid.apply(
        lambda r: f"return {r.ret*100:+.1f}%, Sharpe {r.sharpe:.2f}, "
                  f"drawdown {r.max_dd*100:.1f}%, vs 60/40 {r.excess*100:+.1f}%", axis=1)

    # blocks first, then crises, taken from the period module so a window added there
    # cannot fall out of the ordering and land in the winner table as a blank row
    canonical = ([n for n, _, _ in periods.BLOCKS] + [n for n, _, _ in periods.CRISES])
    period_order = [p for p in canonical if p in per_list]
    # Higher is better for every metric here, including max drawdown: it is stored as a
    # negative number, so -5% ranks above -30%.
    full_rank = (grid[grid.period == periods.FULL].set_index("method_label")["value"]
                 .sort_values(ascending=False))
    method_order = list(full_rank.index) + [m for m in grid.method_label.unique()
                                            if m not in full_rank.index]

    # Full history is cumulative over 23 years and dwarfs a 5-year block, so it would
    # flatten the colour scale. It sets the row order instead of taking a column.
    cells = grid[grid.period != periods.FULL]
    st.markdown(f'<div class="bemo-sec">{_esc(metric)} by window'
                '<span class="hint">best at the top, hover a cell for the full '
                'readout</span></div>', unsafe_allow_html=True)
    st.altair_chart(
        charts.period_heatmap(cells, "value", period_order, method_order, fmt=fmt,
                              centre_zero=centre, title=metric),
        use_container_width=True, theme=None)
    st.caption("Best at the top, green is always better. The 60/40 benchmark is a row of "
               "its own, so if it wins a column nothing beat it in that window.")

    # winner per window. max_dd is negative, so the largest value is the shallowest fall
    sub = cells
    wins = sub.loc[sub.groupby("period")["value"].idxmax(),
                   ["period", "method_label", "value"]]
    st.markdown('<div class="bemo-sec">Winner in each window</div>',
                unsafe_allow_html=True)
    w = wins.copy()
    w["value"] = w["value"].map(lambda v: f"{v:{fmt}}" + ("%" if mult == 100 else ""))
    w = w.rename(columns={"period": "Window", "method_label": "Winner", "value": metric})
    w["Window"] = pd.Categorical(w["Window"], period_order, ordered=True)
    st.markdown(bemo_table(w.sort_values("Window").astype(str)),
                unsafe_allow_html=True)

    tally = (wins.groupby("method_label").agg(
        wins=("period", "size"), periods=("period", lambda s: ", ".join(s)))
        .reset_index().sort_values("wins", ascending=False))
    st.markdown('<div class="bemo-sec">Windows won per method</div>',
                unsafe_allow_html=True)
    st.altair_chart(charts.winners_chart(tally), use_container_width=True, theme=None)
    n_win = len(wins)
    top = tally.iloc[0]
    st.caption(f"{top.method_label} won {int(top.wins)} of {n_win} windows. The crisis "
               "windows sit inside the calendar blocks, so the windows overlap and the "
               "count is a description, not a score.")

    with st.expander("The full grid as a table"):
        tbl = grid.pivot_table(index="method_label", columns="period", values="value",
                               observed=True)
        tbl = tbl[[p for p in period_order if p in tbl.columns]].reindex(method_order)
        st.dataframe(tbl.style.format("{:.2f}", na_rep="").background_gradient(
            cmap="RdYlGn", axis=None), use_container_width=True)
        st.download_button("Download CSV", grid.to_csv(index=False),
                           f"optimizer_by_period_{uni}_{tier_c}.csv", "text/csv")
