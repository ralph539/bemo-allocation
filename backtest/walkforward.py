import numpy as np
import pandas as pd

START_EUR = 1_000_000.0


def rebalance_dates(index: pd.DatetimeIndex, min_history: int) -> set:
    # first trading day of each month, once min_history days of history exist
    pos = pd.Series(range(len(index)), index=index)
    firsts = pos.groupby(index.to_period("M")).first()
    return set(index[[p for p in firsts.values if p >= min_history]])


def walk_forward(returns: pd.DataFrame, target_fn, min_history: int,
                 cost_bps: float = 10.0, tol_band: float = 0.05):
    idx = returns.index
    cols = list(returns.columns)
    rebal = rebalance_dates(idx, min_history)
    rate = cost_bps / 1e4
    h, eq, peak = None, 1.0, 1.0
    curve, dates, turnovers = [], [], []
    rebal_rows, value_rows = [], []
    pnl, wsum, ndays = np.zeros(len(cols)), np.zeros(len(cols)), 0
    for t in idx:
        r = returns.loc[t].values
        day_ret = 0.0
        if h is not None:
            pnl += eq * START_EUR * h * r          # today's P&L on holdings set at last rebalance
            wsum += h
            ndays += 1
            day_ret = float(h @ r)
            eq *= 1 + day_ret
            h = h * (1 + r)
            h = h / h.sum()
        if t in rebal:
            target = target_fn(returns.loc[:t]).reindex(cols).values
            val_at = eq * START_EUR                 # value at close of t, before cost
            before = np.zeros(len(cols)) if h is None else h.copy()
            breached = False
            if h is None:
                after = target.copy()
            elif np.max(np.abs(h - target)) > tol_band:
                breached = True
                to = float(np.abs(target - h).sum())
                eq *= 1 - rate * to
                turnovers.append((t, to))
                after = target.copy()
            else:
                after = h.copy()
            h = after
            trade = after - before
            for i, sl in enumerate(cols):
                rebal_rows.append({
                    "date": t, "sleeve": sl, "target_w": float(target[i]),
                    "w_before": float(before[i]), "w_after": float(after[i]),
                    "trade_pct": float(trade[i]), "trade_eur": float(trade[i] * val_at),
                    "cost_eur": float(rate * abs(trade[i]) * val_at) if breached else 0.0,
                    "breached": bool(breached)})
        if h is not None:
            peak = max(peak, eq)
            value_rows.append({"date": t, "value_eur": eq * START_EUR,
                               "daily_ret": day_ret, "drawdown": eq / peak - 1})
            curve.append(eq)
            dates.append(t)
    equity = pd.Series(curve, index=pd.DatetimeIndex(dates))
    attribution = [{"sleeve": cols[i], "pnl_eur": float(pnl[i]),
                    "avg_weight": float(wsum[i] / ndays) if ndays else 0.0,
                    "contrib_return": float(pnl[i] / START_EUR)} for i in range(len(cols))]
    return equity, turnovers, rebal_rows, value_rows, attribution
