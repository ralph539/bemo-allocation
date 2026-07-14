import os

import duckdb
import pandas as pd

from engine.io import load_config, DB

_BUCKET = None
REF = "-"
CCY = os.environ.get("BEMO_CCY", "EUR")


def _con():
    return duckdb.connect(str(DB), read_only=True)


def options():
    con = _con()
    try:
        return con.execute("select distinct universe, variant, tier, method "
                           "from backtest_metrics").df()
    finally:
        con.close()


def load_run(universe: str, tier: str, method: str, variant: str = "house") -> dict:
    key = [universe, variant, tier, method]
    where = "universe=? and variant=? and tier=? and method=?"
    con = _con()
    try:
        metrics = con.execute(f"select * from backtest_metrics where {where}", key).df()
        if metrics.empty:
            raise KeyError(f"no run for {universe}/{variant}/{tier}/{method}")
        value = con.execute(f"select date, value_eur, daily_ret, drawdown from value_log "
                            f"where {where} order by date", key).df()
        rebal = con.execute(f"select date, sleeve, target_w, w_before, w_after, trade_pct, "
                            f"trade_eur, cost_eur, breached from rebalance_log where {where} "
                            f"order by date, sleeve", key).df()
        attrib = con.execute(f"select sleeve, pnl_eur, avg_weight, contrib_return from "
                             f"attribution where {where} order by pnl_eur desc", key).df()
    finally:
        con.close()
    return {"metrics": metrics.iloc[0], "value": value, "rebal": rebal, "attrib": attrib}


def run_label(variant: str, tier: str, method: str) -> str:
    return method if variant == REF else f"{variant}/{tier}/{method}"


def scoreboard() -> pd.DataFrame:
    con = _con()
    try:
        df = con.execute("select * from backtest_metrics").df()
    finally:
        con.close()
    df["run"] = [run_label(v, t, m) for v, t, m in zip(df.variant, df.tier, df.method)]
    df["gate"] = ["pass" if d >= 0.95 else "FAIL" for d in df.dsr]
    return df


def curves(universe: str) -> pd.DataFrame:
    # every equity curve for a universe, in EUR, as columns keyed by run label
    con = _con()
    try:
        df = con.execute("select variant, tier, method, date, equity from backtest_curves "
                         "where universe=? order by date", [universe]).df()
    finally:
        con.close()
    df["run"] = [run_label(v, t, m) for v, t, m in zip(df.variant, df.tier, df.method)]
    return df.pivot(index="date", columns="run", values="equity") * 1_000_000.0


_REBAL_COLS = {"date": "Date", "sleeve": "Sleeve", "target_w": "Target %", "w_before": "Before %",
               "w_after": "After %", "trade_pct": "Trade %", "trade_eur": "Trade EUR",
               "cost_eur": "Cost EUR", "breached": "Traded"}


def format_rebal(rebal, eps: float = 5e-7) -> pd.DataFrame:
    # solver residuals (order 1e-10) are not weights: clip them before they reach the screen
    df = rebal.copy()
    for c in ["target_w", "w_before", "w_after", "trade_pct"]:
        df[c] = (df[c] * 100).where(df[c].abs() > eps, 0.0)
    for c in ["trade_eur", "cost_eur"]:
        df[c] = df[c].where(df[c].abs() > 0.5, 0.0)
    return df.rename(columns=_REBAL_COLS)[list(_REBAL_COLS.values())]


def bucket_map() -> dict:
    global _BUCKET
    if _BUCKET is None:
        _BUCKET = {s["name"]: s["bucket"] for s in load_config().sleeves}
    return _BUCKET


def proxy_map() -> dict:
    return {s["name"]: s["proxy"] for s in load_config().sleeves}


def latest_weights(rebal):
    last = rebal[rebal["date"] == rebal["date"].max()]
    return last.set_index("sleeve")["w_after"]


def holdings(rebal, eps: float = 5e-5) -> pd.DataFrame:
    # the actual book on the last rebalance: one row per held sleeve, biggest first
    w = latest_weights(rebal)
    bkt, prx = bucket_map(), proxy_map()
    df = pd.DataFrame({
        "Sleeve": w.index,
        "Bucket": [bkt.get(s, "") for s in w.index],
        "ETF": [prx.get(s, "") for s in w.index],
        "Weight %": (w.values * 100),
    })
    df = df[df["Weight %"] > eps * 100].sort_values("Weight %", ascending=False)
    return df.reset_index(drop=True)


def fmt_eur(x: float) -> str:
    return f"{CCY} {x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"
