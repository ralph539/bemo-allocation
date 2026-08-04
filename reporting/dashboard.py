import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from reporting import data, charts, periods

st.set_page_config(page_title="Bemo allocation backtest", layout="wide")

REF = "-"
BENCH = "bench_60_40"
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
                    "mean_cvar_band10", "mean_cvar_band15", "mean_cvar_relaxed"],
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
    "mean_cvar_band10": "Wider band 10%",
    "mean_cvar_band15": "Wider band 15%",
    "mean_cvar_relaxed": "Relaxed caps",
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
    "mean_cvar_no_band": "Mean-CVaR with the +/-5% tactical band removed.",
    "mean_cvar_no_sleeve_caps": "Mean-CVaR with the per-sleeve caps removed (equity band kept).",
    "mean_cvar_equity_band_only": "Mean-CVaR keeping only the equity band.",
    "mean_cvar_uncapped": "Mean-CVaR long-only, all caps removed. Shows what the caps cost.",
    "mean_cvar_band10": "Mean-CVaR with a wider 10% tactical band.",
    "mean_cvar_band15": "Mean-CVaR with a wider 15% tactical band.",
    "mean_cvar_relaxed": "Mean-CVaR under the relaxed cap regime (no band, EM and AI widened).",
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
}

UNIVERSE_NOTE = {
    "us_long": "Long-history US proxies (index mutual funds, gold futures, QQQ for the "
               "thematic sleeve). The only universes that reach back to 2003.",
    "us_long_stocks": "As us_long, with the thematic sleeve as an equal-weight AAPL, MSFT, "
                      "AMZN basket. The ETF-plus-single-stock mix.",
    "eur_book": "The live EUR 14-sleeve book. The AI sleeve limits history to 2022.",
    "eur_book_noAI": "The EUR book without the AI sleeve.",
    "us_book_noAI": "The US-proxy book without the AI sleeve.",
    "full_14": "The funded 14-sleeve book, all weight variants and every method.",
    "no_AI_13": "The funded book without the AI sleeve.",
}

COLS = {"variant": "Variant", "tier": "Tier", "method": "Method", "ann_return": "CAGR %",
        "ann_vol": "Vol %", "sharpe": "Sharpe", "dsr": "Confidence", "beats_bench": "Beats 60/40",
        "excess_return": "vs 60/40 %", "excess_sharpe": "vs 60/40 Sharpe", "max_dd": "Max DD %",
        "s2022_ret": "2022 %", "s2022_dd": "2022 DD %",
        "cvar95": "CVaR95 %", "beta": "Beta", "alpha": "Alpha %", "tracking_error": "TE %",
        "info_ratio": "Info ratio", "turnover": "Turnover"}
CORE = ["Variant", "Tier", "Method", "CAGR %", "Vol %", "Sharpe", "Confidence", "Beats 60/40",
        "Max DD %", "Turnover"]
STRESS_COLS = ["2022 %", "2022 DD %"]
VS_BENCH = ["vs 60/40 %", "Beta", "Alpha %", "TE %", "Info ratio"]
PCT = ["CAGR %", "Vol %", "Max DD %", "CVaR95 %", "Alpha %", "TE %", "vs 60/40 %",
       "2022 %", "2022 DD %", "Return %"]
RATIO = ["Sharpe", "Beta", "Info ratio", "Turnover", "vs 60/40 Sharpe"]


@st.cache_data(show_spinner=False)
def load_curves():
    return data.all_curves()


@st.cache_data(show_spinner=False)
def load_rf(universe: str):
    return data.risk_free(universe)


@st.cache_data(show_spinner=False)
def period_board(universe: str, period: str) -> pd.DataFrame:
    """Every run in a universe, scored over one window, computed from the curves.

    Sub-period numbers are slices of a single walk-forward, never a re-fit, so they
    stay honest out-of-sample. Sharpe is an excess-return Sharpe over the universe's
    own money-market sleeve, matching the full-sample metrics in the store.
    """
    cur = load_curves()
    cur = cur[cur.universe == universe]
    rf = load_rf(universe)
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
CURVES = load_curves()
ALL_UNIVERSES = sorted(CURVES.universe.unique())
LIVE_UNIVERSES = set(CURVES.loc[CURVES.source == "live", "universe"].unique())
# universes with enough history to be split into decades
LONG_HISTORY = [u for u in ALL_UNIVERSES
                if len(data.universe_dates(CURVES, u)) > 2500]
# the lab re-runs two books the live backtest already covers in full detail
LIVE_EQUIV = {"eur_book": "full_14", "eur_book_noAI": "no_AI_13"}


def humanise(df):
    out = df[[c for c in COLS if c in df.columns]].rename(columns=COLS)
    for c in PCT:
        if c in out:
            out[c] = out[c] * 100
    if "Confidence" in out:
        out["Confidence"] = out["Confidence"] * 100
    if "Beats 60/40" in out:
        out["Beats 60/40"] = out["Beats 60/40"].map({True: "yes", False: "no"})
    return out


def style_board(df):
    fmt = {c: "{:.2f}" for c in PCT + RATIO if c in df.columns}
    if "Confidence" in df.columns:
        fmt["Confidence"] = "{:.1f}%"
    s = df.style.format(fmt)
    if "Sharpe" in df.columns:
        s = s.background_gradient(cmap="RdYlGn", subset=["Sharpe"], vmin=0.3, vmax=1.3)
    if "Confidence" in df.columns:
        s = s.background_gradient(cmap="RdYlGn", subset=["Confidence"], vmin=40, vmax=100)
    for c in STRESS_COLS:
        if c in df.columns:
            s = s.map(lambda v: "color:#c5221f" if isinstance(v, float) and v < 0 else "",
                      subset=[c])
    if "Beats 60/40" in df.columns:
        s = s.map(lambda v: "background-color:#e6f4ea;color:#137333;font-weight:600"
                  if v == "yes" else "color:#9aa0a6", subset=["Beats 60/40"])
    neg = [c for c in ("Alpha %", "Info ratio", "vs 60/40 %", "Return %") if c in df.columns]
    if neg:
        s = s.map(lambda v: "color:#c5221f" if isinstance(v, float) and v < 0 else "", subset=neg)
    if "Method" in df.columns:
        s = s.map(lambda v: "background-color:#e8f0fe;font-weight:600" if v == BENCH else "",
                  subset=["Method"])
    return s


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


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("Bemo allocation")
    def _span_label(u):
        d = data.universe_dates(CURVES, u)
        return f"{u}  ({d.min():%Y} to {d.max():%Y})"

    universe = st.selectbox("Universe", ALL_UNIVERSES, format_func=_span_label,
                            index=ALL_UNIVERSES.index("full_14")
                            if "full_14" in ALL_UNIVERSES else 0)
    st.caption(UNIVERSE_NOTE.get(universe, ""))

    udates = data.universe_dates(CURVES, universe)
    period_opts = periods.available(udates)
    period = st.selectbox("Period", period_opts,
                          format_func=lambda p: periods.label(p, udates))
    p_start, p_end = periods.bounds(period, udates)
    st.caption(f"{p_start:%d %b %Y} to {p_end:%d %b %Y}"
               + (f". {periods.NOTE[period]}" if period in periods.NOTE else ""))
    if period != periods.FULL:
        st.caption("Sub-period numbers are slices of one walk-forward, never re-fitted.")
    # a short universe cannot offer the earlier decades: say so rather than let the
    # short list look like a bug
    if len(period_opts) <= 2 and LONG_HISTORY:
        yrs = (udates.max() - udates.min()).days / 365.25
        st.info(f"{universe} only holds {yrs:.0f} years of history, so the earlier "
                f"windows do not exist for it. Switch to "
                f"**{'** or **'.join(LONG_HISTORY)}** for the full 2003 to 2026 set "
                "of decades and crises.")

view = st.segmented_control("", ["Single run", "Compare runs", "Full scoreboard",
                                 "Which optimizer won"],
                            default="Single run", label_visibility="collapsed") or "Single run"

pboard = period_board(universe, period)
is_full = period == periods.FULL
is_live = universe in LIVE_UNIVERSES

# ---------------- single run ----------------
if view == "Single run":
    grps = groups_for(universe)
    with st.sidebar:
        group = st.radio("Group", list(grps), format_func=str)
        method = st.radio("Config", grps[group], format_func=lambda m: NICE.get(m, m))
        st.caption(DESC.get(method, ""))
        if method in REF_ONLY:
            variant, tier = REF, REF
            st.caption("Ignores the tier and the strategic weights.")
        else:
            tiers = [t for t in TIERS if t in available(universe, method, "tier")]
            variants = [v for v in VARIANTS if v in available(universe, method, "variant")]
            tier = st.radio("Tier", tiers) if tiers else REF
            variant = st.radio("Weight variant", variants) if variants else "house"
            if len(variants) == 1:
                st.caption(f"This config was only run on the {variants[0]} weights, "
                           "so there is nothing else to pick.")

    row = pboard[(pboard.variant == variant) & (pboard.tier == tier)
                 & (pboard.method == method)]
    if row.empty:
        st.error(f"No data for {NICE.get(method, method)} on {variant} weights, {tier} tier, "
                 f"{universe}, over {period}. Pick another combination.")
        st.stop()
    row = row.iloc[0]

    st.title(NICE.get(method, method) if variant == REF
             else f"{tier.title()} tier, {NICE.get(method, method)}")
    st.caption(f"{universe}{'' if variant == REF else f', {variant} weights'}. "
               f"{periods.label(period, udates)}: {p_start:%d %b %Y} to {p_end:%d %b %Y} "
               f"({int(row['n_days'])} trading days).")
    st.caption(DESC.get(method, ""))

    bench_row = pboard[pboard.method == BENCH]
    r = st.columns(4)
    r[0].metric("Return", data.fmt_pct(row["ret"]),
                help="Total return over the selected window.")
    short = pd.isna(row["ann_return"])
    r[1].metric("CAGR", "n/a" if short else data.fmt_pct(row["ann_return"]),
                help="Annualised growth rate. Not shown on a window under 120 trading "
                     "days: annualising a two-month move produces a number that looks "
                     "like a rate and is not one.")
    r[2].metric("Sharpe", f"{row['sharpe']:.2f}", help="Return above cash per unit of volatility.")
    r[3].metric("vs 60/40", data.fmt_pct(row["excess_return"]),
                help="Total return minus the benchmark's, same window.")
    r = st.columns(4)
    r[0].metric("Volatility", "n/a" if pd.isna(row["ann_vol"])
                else data.fmt_pct(row["ann_vol"]))
    r[1].metric("Max drawdown", data.fmt_pct(row["max_dd"]), help="Worst fall from a peak.")
    r[2].metric("CVaR 95", data.fmt_pct(row["cvar95"]), help="Average of the worst 5% of days.")
    if len(bench_row):
        r[3].metric("60/40 return", data.fmt_pct(bench_row.iloc[0]["ret"]))

    # curve for this run and the benchmark, both rebased to the window start
    def _curve(v, t, m):
        g = CURVES[(CURVES.universe == universe) & (CURVES.variant == v)
                   & (CURVES.tier == t) & (CURVES.method == m)]
        if g.empty:
            return None
        eq = g.set_index("date")["equity"].sort_index()
        s = periods.slice_curve(eq, period)
        return None if s.empty else s * 1_000_000.0

    mine = _curve(variant, tier, method)
    bcur = None if method == BENCH else _curve(REF, REF, BENCH)
    wide = pd.DataFrame({NICE.get(method, method): mine})
    if bcur is not None:
        wide[NICE[BENCH]] = bcur
    st.altair_chart(charts.compare_chart(wide.dropna(how="all"), dashed=NICE[BENCH]),
                    use_container_width=True)
    st.caption("Both lines restart at EUR 1,000,000 on the first day of the window, so the "
               "comparison is like for like.")

    if is_live:
        try:
            run = data.load_run(universe, tier, method, variant)
            all_rebal, attrib = run["rebal"], run["attrib"]
            # the ledger carries dates, so holdings and trades follow the window
            rebal = all_rebal[(all_rebal["date"] >= p_start)
                              & (all_rebal["date"] <= p_end)]
            if rebal.empty:
                rebal = all_rebal

            c = st.columns(2)
            when = "at the end of the window" if not is_full else "on the last rebalance"
            c[0].caption(f"Allocation {when}, coloured by bucket")
            c[0].pyplot(charts.donut_fig(data.latest_weights(rebal), data.bucket_map(),
                                         figsize=(6.6, 6.0), scale=1.8))
            if is_full:
                c[1].caption(f"Profit and loss by sleeve ({data.CCY}), before trading cost")
                c[1].pyplot(charts.pnl_fig(attrib, figsize=(6.4, 5.6), scale=1.7))
            else:
                c[1].caption("Profit and loss by sleeve")
                c[1].info("Sleeve attribution is stored for the whole run, not per day, "
                          "so it cannot be cut to a window. Switch Period to full "
                          "history to see it.")

            hold = data.holdings(rebal)
            last_date = pd.to_datetime(rebal["date"]).max()
            st.caption(f"Portfolio holdings on {last_date:%d %b %Y}, the last rebalance "
                       f"in this window ({len(hold)} sleeves, weights sum to "
                       f"{hold['Weight %'].sum():.1f}%).")
            bcol = charts.BUCKET_COLORS
            styled = (hold.style
                      .apply(lambda r: [f"color: {bcol.get(r['Bucket'], '#333')}; "
                                        "font-weight: 600"] * len(r), axis=1)
                      .format({"Weight %": "{:.2f}"}))
            st.dataframe(styled, use_container_width=True, hide_index=True,
                         column_config={"Weight %": st.column_config.ProgressColumn(
                             format="%.2f%%", min_value=0.0,
                             max_value=float(hold["Weight %"].max()))})
            n_tr = int(rebal[rebal["breached"]]["date"].nunique())
            with st.expander(f"Trades in this window: {n_tr} of "
                             f"{rebal['date'].nunique()} monthly checks"):
                st.pyplot(charts.turnover_fig(rebal))
                only_traded = st.checkbox("Only dates where it traded", value=False)
                rb = rebal[rebal["breached"]] if only_traded else rebal
                st.dataframe(data.format_rebal(rb), use_container_width=True,
                             hide_index=True, height=320)
                st.download_button("Download CSV", rebal.to_csv(index=False),
                                   f"rebalance_{variant}_{tier}_{method}_{universe}.csv",
                                   "text/csv")
        except KeyError:
            st.info("That exact combination has no stored ledger.")
    else:
        twin = LIVE_EQUIV.get(universe)
        st.info(f"{universe} is a robustness-lab universe: it stores equity curves only, "
                "so it carries no holdings, attribution or trade ledger."
                + (f" For the same book with the full detail, switch Universe to "
                   f"**{twin}**." if twin else ""))

# ---------------- compare ----------------
elif view == "Compare runs":
    st.title("Compare equity curves")
    st.caption(f"{universe}, {periods.label(period, udates)}. Every run restarts at "
               f"EUR 1,000,000 on {p_start:%d %b %Y}.")

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
            st.altair_chart(charts.compare_chart(wide, dashed=BENCH),
                            use_container_width=True)
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
            st.dataframe(style_board(show), use_container_width=True, hide_index=True)

# ---------------- scoreboard ----------------
elif view == "Full scoreboard":
    st.title("Full scoreboard")
    st.caption(f"{universe}, {periods.label(period, udates)}.")
    if pboard.empty:
        st.info("No runs cover this window.")
        st.stop()

    b = pboard.copy()
    k = st.columns(3)
    k[0].metric("Runs", len(b))
    k[1].metric("Beat 60/40", int(b.beats_bench.sum()),
                help="Higher total return than the benchmark over this window.")
    best = b[b.method != BENCH].loc[lambda d: d.sharpe.idxmax()]
    # the run name goes in help, not delta: delta renders as a green up-arrow "change"
    k[2].metric("Best Sharpe", f"{best.sharpe:.2f}",
                help=f"{best['run']}. The maximum over {len(b)} correlated runs, so it is "
                     "the luckiest draw as much as the best method.")

    if is_full and is_live:
        brow = board[(board.universe == universe) & (board.method == BENCH)]
        if len(brow):
            confidence_note(brow.iloc[0].n_trials, brow.iloc[0].dsr)
    elif not is_full:
        st.caption("Confidence (Deflated Sharpe) is a full-sample statistic and is not shown "
                   "for a sub-period: a five-year slice cannot carry a multiple-testing "
                   "correction built over the whole history.")

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
        st.dataframe(style_board(show), use_container_width=True, hide_index=True, height=620)
        st.download_button("Download CSV", show.to_csv(index=False),
                           f"scoreboard_{universe}_{period}.csv", "text/csv")

# ---------------- which optimizer won ----------------
else:
    st.title("Which optimizer won, and when")
    st.caption("Every method scored over every window it has data for. Each cell is a "
               "slice of that method's single walk-forward curve, so the weights were "
               "never re-fitted to the window being judged. This view spans all windows "
               "at once, so the sidebar Period does not apply here.")

    c = st.columns([1.3, 1, 1.3, 1.1])
    uni = c[0].selectbox("Universe", ALL_UNIVERSES,
                         index=ALL_UNIVERSES.index(universe),
                         help="Starts from the sidebar universe. Only us_long and "
                              "us_long_stocks reach back far enough to compare decades.")
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
        st.info(f"{uni} only spans {d0.min():%Y} to {d0.max():%Y}, so there are few "
                "windows to compare. Switch to us_long or us_long_stocks above to see "
                "the full 2003 to 2026 picture.")

    udates_c = data.universe_dates(CURVES, uni)
    per_list = [p for p in periods.available(udates_c) if p != periods.FULL]
    if not per_list:
        st.info(f"{uni} does not span enough history to split into windows.")
        st.stop()

    recs = []
    for p in [periods.FULL] + per_list:
        pb = period_board(uni, p)
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

    period_order = [p for p in ["2003-2007", "2008-2012", "2013-2017", "2018-2022",
                                "2023-2026", "GFC 2008", "Euro crisis 2011",
                                "COVID 2020", "Rate shock 2022"] if p in per_list]
    # Higher is better for every metric here, including max drawdown: it is stored as a
    # negative number, so -5% ranks above -30%.
    full_rank = (grid[grid.period == periods.FULL].set_index("method_label")["value"]
                 .sort_values(ascending=False))
    method_order = list(full_rank.index) + [m for m in grid.method_label.unique()
                                            if m not in full_rank.index]

    # Full history is cumulative over 23 years and dwarfs a 5-year block, so it would
    # flatten the colour scale. It sets the row order instead of taking a column.
    cells = grid[grid.period != periods.FULL]
    st.altair_chart(
        charts.period_heatmap(cells, "value", period_order, method_order, fmt=fmt,
                              centre_zero=centre, title=metric),
        use_container_width=True)
    st.caption("Rows are ordered by the full-history result, best at the top. Green is "
               "better throughout: drawdown is stored as a negative number, so a "
               "shallower fall is the greener one. The 60/40 benchmark sits in the grid "
               "as its own row, so if it wins a column, nothing beat it in that window. "
               "The risk-structure optimisers (risk parity, HRP, min variance, max "
               "Sharpe, max diversification, equal weight, inverse vol) ignore the tier "
               "and its caps, so they are a reference, not a like-for-like mandate.")

    # winner per window. max_dd is negative, so the largest value is the shallowest fall
    sub = cells
    wins = sub.loc[sub.groupby("period")["value"].idxmax(),
                   ["period", "method_label", "value"]]
    st.subheader("Winner in each window")
    w = wins.copy()
    w["value"] = w["value"].map(lambda v: f"{v:{fmt}}" + ("%" if mult == 100 else ""))
    w = w.rename(columns={"period": "Window", "method_label": "Winner", "value": metric})
    w["Window"] = pd.Categorical(w["Window"], period_order, ordered=True)
    st.dataframe(w.sort_values("Window"), use_container_width=True, hide_index=True)

    tally = (wins.groupby("method_label").agg(
        wins=("period", "size"), periods=("period", lambda s: ", ".join(s)))
        .reset_index().sort_values("wins", ascending=False))
    st.altair_chart(charts.winners_chart(tally), use_container_width=True)
    n_win = len(wins)
    top = tally.iloc[0]
    st.caption(f"{top.method_label} won {int(top.wins)} of {n_win} windows. Read this as a "
               "description, not a score: the crisis windows sit inside the calendar "
               "blocks, so the windows overlap and a win count is not independent "
               "evidence. A method that wins one window and sits mid-table elsewhere was "
               "probably suited to that regime rather than better in general.")

    with st.expander("The full grid as a table"):
        tbl = grid.pivot_table(index="method_label", columns="period", values="value",
                               observed=True)
        tbl = tbl[[p for p in period_order if p in tbl.columns]].reindex(method_order)
        st.dataframe(tbl.style.format("{:.2f}", na_rep="").background_gradient(
            cmap="RdYlGn", axis=None), use_container_width=True)
        st.download_button("Download CSV", grid.to_csv(index=False),
                           f"optimizer_by_period_{uni}_{tier_c}.csv", "text/csv")

st.caption("Educational reference, not investment advice.")
