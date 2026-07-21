import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from reporting import data, charts

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
       "2022 %", "2022 DD %"]
RATIO = ["Sharpe", "Beta", "Info ratio", "Turnover", "vs 60/40 Sharpe"]

opts = data.options()
board = data.scoreboard()


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
    neg = [c for c in ("Alpha %", "Info ratio", "vs 60/40 %") if c in df.columns]
    if neg:
        s = s.map(lambda v: "color:#c5221f" if isinstance(v, float) and v < 0 else "", subset=neg)
    if "Method" in df.columns:
        s = s.map(lambda v: "background-color:#e8f0fe;font-weight:600" if v == BENCH else "",
                  subset=["Method"])
    return s


def available(universe, method, col):
    # only offer combinations that actually exist in the store (cap regimes and the lab
    # methods only ran on the house weights)
    v = board[(board.universe == universe) & (board.method == method)][col].unique()
    return [x for x in v if x != REF]


def present_methods(universe):
    return set(board[board.universe == universe].method.unique())


def groups_for(universe):
    have = present_methods(universe)
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
            f"The passive 60/40 benchmark scores **{bench_dsr*100:.1f}%** and fails too. The same "
            f"portfolio scores **96.3%** as a single idea and **81.8%** after 120.\n\n"
            f"**A short bull window cannot prove an edge over cash once you have tried "
            f"{int(n_trials)} things.** The fix is more history, not more ideas.")


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("Bemo allocation")
    universe = st.segmented_control("Universe", sorted(opts["universe"].unique()),
                                    default="full_14") or "full_14"

view = st.segmented_control("", ["Single run", "Compare runs", "Full scoreboard",
                                 "Robustness"],
                            default="Single run", label_visibility="collapsed") or "Single run"
bench_row = board[(board.universe == universe) & (board.method == BENCH)].iloc[0]

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
            tier = st.radio("Tier", tiers)
            variant = st.radio("Weight variant", variants)
            if len(variants) == 1:
                st.caption(f"This config was only run on the {variants[0]} weights, "
                           "so there is nothing else to pick.")

    try:
        run = data.load_run(universe, tier, method, variant)
    except KeyError:
        st.error(f"That combination was never run: {NICE.get(method, method)} on {variant} weights, "
                 f"{tier} tier, {universe}. Pick another.")
        st.stop()
    m, value, rebal, attrib = run["metrics"], run["value"], run["rebal"], run["attrib"]
    start, end = value["value_eur"].iloc[0], value["value_eur"].iloc[-1]
    cost = rebal["cost_eur"].sum()

    st.title(NICE.get(method, method) if variant == REF
             else f"{tier.title()} tier, {NICE.get(method, method)}")
    st.caption(f"{universe}{'' if variant == REF else f', {variant} weights'}. "
               f"{value['date'].iloc[0].date()} to {value['date'].iloc[-1].date()}.")
    st.caption(DESC.get(method, ""))

    r = st.columns(4)
    r[0].metric("Start", data.fmt_eur(start))
    r[1].metric("End", data.fmt_eur(end))
    r[2].metric("CAGR", data.fmt_pct(m["ann_return"]), help="Yearly compounded growth rate.")
    r[3].metric("Sharpe", f"{m['sharpe']:.2f}", help="Return above cash per unit of volatility.")
    r = st.columns(4)
    r[0].metric("Volatility", data.fmt_pct(m["ann_vol"]))
    r[1].metric("Max drawdown", data.fmt_pct(m["max_dd"]), help="Worst fall from a peak.")
    r[2].metric("Beta", f"{m['beta']:.2f}", help="Share of the benchmark's risk carried.")
    r[3].metric("Alpha", data.fmt_pct(m["alpha"]), help="Return that market exposure cannot explain.")
    if "s2022_ret" in m.index:
        r = st.columns(4)
        r[0].metric("2022 return", data.fmt_pct(m["s2022_ret"]),
                    help="Return in the 2022 stress year, when stocks and bonds both fell.")
        r[1].metric("2022 drawdown", data.fmt_pct(m["s2022_dd"]))
        r[2].metric("CVaR 95", data.fmt_pct(m["cvar95"]), help="Average of the worst 5% of days.")
        r[3].metric("Turnover", f"{m['turnover']:.2f}x", help="How much of the book it trades a year.")

    traded = int(rebal[rebal["breached"]]["date"].nunique())
    st.caption(f"Traded on {traded} of {rebal['date'].nunique()} monthly checks. "
               f"Turnover {m['turnover']:.2f}x a year. Total cost {data.fmt_eur(cost)}.")

    bench_curve = None if method == BENCH else data.load_run(universe, REF, BENCH, REF)["value"]
    st.pyplot(charts.equity_drawdown_fig(value, bench_curve))
    if bench_curve is not None:
        st.caption("Dashed line is the passive 60/40 benchmark, on both panels.")
    c = st.columns(2)
    c[0].caption("Latest allocation, coloured by bucket")
    c[0].pyplot(charts.donut_fig(data.latest_weights(rebal), data.bucket_map(),
                                 figsize=(6.6, 6.0), scale=1.8))
    c[1].caption(f"Profit and loss by sleeve ({data.CCY}), before trading cost")
    c[1].pyplot(charts.pnl_fig(attrib, figsize=(6.4, 5.6), scale=1.7))

    hold = data.holdings(rebal)
    st.caption(f"Portfolio holdings on the last rebalance ({len(hold)} sleeves, "
               f"weights sum to {hold['Weight %'].sum():.1f}%). "
               "Text colour matches the donut bucket.")
    bcol = charts.BUCKET_COLORS
    styled = (hold.style
              .apply(lambda r: [f"color: {bcol.get(r['Bucket'], '#333')}; font-weight: 600"] * len(r),
                     axis=1)
              .format({"Weight %": "{:.2f}"}))
    st.dataframe(styled, use_container_width=True, hide_index=True,
                 column_config={"Weight %": st.column_config.ProgressColumn(
                     format="%.2f%%", min_value=0.0,
                     max_value=float(hold["Weight %"].max()))})

    with st.expander("Trades: every date, every sleeve"):
        st.pyplot(charts.turnover_fig(rebal))
        only_traded = st.checkbox("Only dates where it traded", value=False)
        rb = rebal[rebal["breached"]] if only_traded else rebal
        st.dataframe(data.format_rebal(rb), use_container_width=True, hide_index=True, height=320,
                     column_config={
                         "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                         "Target %": st.column_config.NumberColumn(format="%.2f"),
                         "Before %": st.column_config.NumberColumn(format="%.2f"),
                         "After %": st.column_config.NumberColumn(format="%.2f"),
                         "Trade %": st.column_config.NumberColumn(format="%.2f"),
                         "Trade EUR": st.column_config.NumberColumn(format="%.0f"),
                         "Cost EUR": st.column_config.NumberColumn(format="%.0f"),
                         "Traded": st.column_config.CheckboxColumn()})
        st.download_button("Download CSV", rebal.to_csv(index=False),
                           f"rebalance_{variant}_{tier}_{method}_{universe}.csv", "text/csv")

# ---------------- compare ----------------
elif view == "Compare runs":
    st.title("Compare equity curves")
    st.caption(f"{universe}. Every run starts from EUR 1,000,000 on the same day.")
    eq = data.curves(universe)
    avail = set(eq.columns)

    def house(tier, methods):
        return [BENCH] + [f"house/{tier}/{m}" for m in methods if f"house/{tier}/{m}" in avail]

    preset = st.pills("Preset", ["Engine vs baseline vs 60/40", "Offense optimisers",
                                 "Risk-off overlays", "Combinations", "The cap ladder",
                                 "Weight variants", "All the references", "Custom"],
                      default="Engine vs baseline vs 60/40") or "Custom"
    picked = []
    if preset == "All the references":
        picked = [c for c in GROUPS["References (risk only)"] if c in avail]
    elif preset == "Custom":
        f = st.columns(3)
        fv = f[0].multiselect("Variant", VARIANTS, default=["house"])
        ft = f[1].multiselect("Tier", TIERS, default=["balanced"])
        fm = f[2].multiselect("Method", TIER_METHODS, default=["mean_cvar", "max_ret_cvarcap",
                                                               "strategic"],
                              format_func=lambda m: NICE.get(m, m))
        picked = [c for c in avail if c == BENCH or (c.count("/") == 2
                  and c.split("/")[0] in fv and c.split("/")[1] in ft and c.split("/")[2] in fm)]
    else:
        tier = st.select_slider("Tier", TIERS, value="balanced")
        if preset == "Engine vs baseline vs 60/40":
            picked = house(tier, ["mean_cvar", "strategic"]) + \
                (["risk_parity"] if "risk_parity" in avail else [])
        elif preset == "Offense optimisers":
            picked = house(tier, ["mean_cvar", "max_ret_cvarcap", "trend_tilt",
                                  "black_litterman_mom"])
        elif preset == "Risk-off overlays":
            picked = house(tier, ["mean_cvar", "vol_target", "regime_breaker", "dual_momentum"])
        elif preset == "Combinations":
            picked = house(tier, ["max_ret_cvarcap", "cvarcap_breaker", "cvarcap_dualmom",
                                  "trend_dualmom"])
        elif preset == "The cap ladder":
            picked = house(tier, ["mean_cvar", "mean_cvar_no_band", "mean_cvar_no_sleeve_caps",
                                  "mean_cvar_equity_band_only", "mean_cvar_uncapped"])
        else:
            picked = [BENCH] + [f"{v}/{tier}/mean_cvar" for v in VARIANTS
                                if f"{v}/{tier}/mean_cvar" in avail]

    picked = st.multiselect("Runs on the chart", sorted(avail), default=sorted(set(picked)))
    if not picked:
        st.info("Pick at least one run.")
    else:
        st.altair_chart(charts.compare_chart(eq[picked], dashed=BENCH), use_container_width=True)
        b = board[(board.universe == universe) & (board.run.isin(picked))].sort_values(
            "sharpe", ascending=False)
        show = humanise(b.assign(variant=b.run)).rename(columns={"Variant": "Run"})
        show = show.drop(columns=[c for c in ("Tier", "Method") if c in show])
        st.dataframe(style_board(show), use_container_width=True, hide_index=True)

# ---------------- scoreboard ----------------
elif view == "Full scoreboard":
    st.title("Full scoreboard")
    b = board[board.universe == universe].copy()
    k = st.columns(3)
    k[0].metric("Runs", len(b))
    k[1].metric("Beat 60/40", int(b.beats_bench.sum()), help="Higher CAGR and higher Sharpe.")
    best = b[b.method != BENCH].loc[lambda d: d.sharpe.idxmax()]
    k[2].metric("Best Sharpe", f"{best.sharpe:.2f}", best["run"])
    confidence_note(bench_row.n_trials, bench_row.dsr)

    all_groups = list(groups_for(universe))
    f = st.columns([2.4, 1.1, 1.2])
    families = f[0].pills("Show", all_groups, selection_mode="multi", default=all_groups)
    only_beat = f[1].toggle("Only beat 60/40", value=False)
    sort_by = f[2].selectbox("Sort by", ["Sharpe", "vs 60/40 %", "Confidence", "CAGR %",
                                         "2022 %", "Alpha %", "Max DD %", "Turnover"])

    b = b[b.method.map(FAMILY_OF).isin(families or all_groups)]
    if only_beat:
        b = b[b.beats_bench]
    c = st.columns(2)
    wide = c[0].toggle("Benchmark-relative columns", value=False)
    show_stress = c[1].toggle("2022 stress columns", value=True)

    if b.empty:
        st.info("No run matches those filters.")
    else:
        show = humanise(b)
        keep = [c for c in CORE + (STRESS_COLS if show_stress else [])
                + (VS_BENCH if wide else []) if c in show.columns]
        show = show[keep].sort_values(sort_by, ascending=sort_by in ("Turnover",))
        st.dataframe(style_board(show), use_container_width=True, hide_index=True, height=620)
        st.download_button("Download CSV", show.to_csv(index=False),
                           f"scoreboard_{universe}.csv", "text/csv")

# ---------------- robustness ----------------
else:
    st.title("Robustness lab")
    st.caption("The house engine walked forward across universes, decades and "
               "crises. Sub-period numbers are slices of one out-of-sample curve "
               "per run, never re-fitted. Ignores the universe picker: this view "
               "spans its own universes, including long-history US proxies to 2000 "
               "and an ETF-plus-stocks mix.")
    try:
        con = data._con()
        rm = con.execute("select * from robustness_metrics").df()
        rc = con.execute("select * from robustness_curves").df()
        con.close()
    except Exception:
        st.info("No robustness tables yet. Run: .venv/bin/python -m backtest.robustness")
        st.stop()

    ENG = "mean_cvar"
    eng_cells = rm[(rm.method == ENG) & (rm.kind != "full")]
    full_eng = rm[(rm.method == ENG) & (rm.kind == "full")]
    k = st.columns(4)
    k[0].metric("Universes", rm.universe.nunique())
    k[1].metric("OOS span", f"{rm.start.min()[:4]} to {rm.end.max()[:4]}")
    k[2].metric("Full runs in profit", f"{int((full_eng.ret > 0).sum())}/{len(full_eng)}")
    aggr = full_eng[full_eng.tier == "aggressive"]
    k[3].metric("Aggressive beats 60/40", f"{int(aggr.beats_bench.sum())}/{len(aggr)}",
                help="Full-history total return vs the 60/40 in the same universe.")

    tier_r = st.pills("Tier", TIERS, default="balanced") or "balanced"
    sub = eng_cells[eng_cells.tier == tier_r]
    period_order = ["2003-2007", "2008-2012", "2013-2017", "2018-2022", "2023-2026",
                    "GFC 2008", "Euro crisis 2011", "COVID 2020", "Rate shock 2022"]

    st.subheader("Excess return vs 60/40 by window")
    piv = sub.pivot_table(index="universe", columns="period", values="excess_ret",
                          observed=True)
    piv = piv[[p for p in period_order if p in piv.columns]]
    st.dataframe(piv.style.format(lambda v: f"{v*100:+.1f}%", na_rep="-")
                 .map(lambda v: "" if pd.isna(v) else
                      ("background-color:#DCEBE2" if v > 0
                       else "background-color:#F3D9D7")),
                 use_container_width=True)
    st.caption("Green = the engine beat 60/40 in that window, same universe. "
               "Crisis columns are peak-to-trough slices.")

    st.subheader("Out-of-sample growth of 1")
    uni = st.pills("Universe", sorted(rc.universe.unique()),
                   default="us_long") or "us_long"
    cur = rc[(rc.universe == uni)
             & (((rc.method == ENG) & (rc.tier == tier_r))
                | (rc.method == BENCH) | ((rc.method == "strategic")
                                          & (rc.tier == tier_r)))]
    wide = cur.pivot_table(index="date", columns="method", values="equity",
                           observed=True).rename(
        columns={ENG: "engine", BENCH: "60/40", "strategic": "strategic"})
    st.line_chart(wide, height=340)

    with st.expander("Full robustness table"):
        st.dataframe(rm, use_container_width=True, hide_index=True, height=480)
        st.download_button("Download CSV", rm.to_csv(index=False),
                           "robustness_metrics.csv", "text/csv")

st.caption("Educational reference, not investment advice.")
