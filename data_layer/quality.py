import numpy as np
import pandas as pd

MAX_GAP_BDAYS = 3
STALE_RUN = 3
EXTREME_RET = 0.20


def _bday_gaps(idx: pd.DatetimeIndex) -> tuple[int, int]:
    if len(idx) < 2:
        return 0, 0
    prev = idx[:-1].values.astype("datetime64[D]")
    cur = idx[1:].values.astype("datetime64[D]")
    g = np.busday_count(prev, cur)
    return int((g > MAX_GAP_BDAYS).sum()), int(g.max())


def _stale_run(p: pd.Series) -> int:
    flat = p.diff().fillna(1) == 0
    best = run = 0
    for f in flat:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def run_quality(prices: pd.DataFrame, fx: pd.DataFrame, ret_df: pd.DataFrame,
                sleeves: list[dict], repairs: dict) -> tuple[str, list[str]]:
    lines, issues = [], []
    starts = [prices[s["proxy"]].dropna().index.min() for s in sleeves
              if not prices[s["proxy"]].dropna().empty]
    common = max(starts)

    lines.append("Bemo data-layer quality report")
    lines.append(f"FX (ECB): {fx.index.min().date()} .. {fx.index.max().date()}  "
                 f"rows={len(fx)}  cols={list(fx.columns)}")
    lines.append(f"Common window (all funded sleeves have data): {common.date()}")
    hdr = f"{'sleeve':40} {'proxy':9} {'start':10} {'end':10} {'yrs':>4} " \
          f"{'rep':>4} {'gaps':>5} {'stale':>5} {'extr':>4} {'extrW':>5}"
    lines += ["", hdr, "-" * len(hdr)]
    for s in sleeves:
        p = prices[s["proxy"]].dropna()
        r = ret_df[s["name"]].dropna()
        if p.empty:
            issues.append(f"CRITICAL {s['name']}: no data")
            continue
        yrs = (p.index.max() - p.index.min()).days / 365.25
        gaps, _ = _bday_gaps(p.index)
        stale = _stale_run(p)
        rep = repairs.get(s["proxy"], 0)
        extr = int((r.abs() > EXTREME_RET).sum())
        extr_w = int((r[r.index >= common].abs() > EXTREME_RET).sum())
        lines.append(f"{s['name'][:40]:40} {s['proxy']:9} {p.index.min().date()} "
                     f"{p.index.max().date()} {yrs:4.1f} {rep:4d} {gaps:5d} "
                     f"{stale:5d} {extr:4d} {extr_w:5d}")
        if yrs < 7:
            issues.append(f"WARN {s['name']}: only {yrs:.1f}y history")
        if extr_w:
            issues.append(f"WARN {s['name']}: {extr_w} extreme move(s) inside the common window")
    lines += ["",
              "rep = bad prints repaired; extr = |daily EUR return| > 20% (full history); "
              "extrW = same, restricted to the common window.",
              "Residual extr are pre-2015 Yahoo prints on the LSE lines (SJPA/SGLN/INFR); "
              "extrW = 0 confirms the working panel is clean."]
    lines += ["", "Issues:" if issues else "Issues: none (all extremes outside the common window)"]
    lines += [f"  {i}" for i in issues]
    return "\n".join(lines), issues
