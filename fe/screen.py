"""Reproducible symbol eligibility from a pre-study screen window only.

The event of interest is a mid move of at least ``threshold_ticks`` over the
primary horizon. A symbol whose tick grid makes that event practically
unreachable contributes almost no positives to pooled training and dominates
pooled metrics through between-symbol scale. Eligibility therefore requires,
over the screen window alone:

* a common stock (four digits, not an ETF prefix) with a reference price floor,
* liquidity (median continuous turnover) and a usable touch (median spread),
* continuous trading on most sampled seconds (no disposition call auction),
* a reachable event: the 1 s-origin rate of |mid move| >= threshold at the
  primary horizon at or above ``minimum_event_rate``.

Eligible symbols are ranked by median turnover (not by the event rate, which
would select the most speculative names) and the top ``size`` are kept. A fixed
salted hash splits them into development and label-holdout symbols; the split
never reads anything but the symbol code.
"""

from __future__ import annotations

import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from ...io import write_yaml
from ..signal.universe import to_ticks
from .profile import calendar_days, work


def _day(args):
    root, symbol, day, horizon, threshold, start, until = args
    path = Path(root) / symbol / f"{day}.csv.zst"
    if not path.is_file():
        return None
    df = pl.read_csv(
        path,
        columns=["Type", "Timestamp", "StatusMask", "Price", "TradeVolume", "BidPrice1", "AskPrice1"],
        schema_overrides={"Price": pl.Float64, "BidPrice1": pl.Float64, "AskPrice1": pl.Float64, "TradeVolume": pl.Float64, "Timestamp": pl.Int64},
    )
    books = df.filter(pl.col("Type") == "B")
    if books.height < 100:
        return None
    ts = books["Timestamp"].to_numpy()
    sec = ((ts // 1_000_000) + 8 * 3600) % 86400 + (ts % 1_000_000) / 1e6
    bid, ask, status = books["BidPrice1"].to_numpy(), books["AskPrice1"].to_numpy(), books["StatusMask"].to_numpy()
    good = (status == 0) & (bid > 0) & (ask > bid)
    grid = np.arange(start, until + horizon + 1, 1.0)
    idx = np.searchsorted(sec, grid, side="right") - 1
    ok = idx >= 0
    cont = np.zeros(len(grid), bool)
    cont[ok] = good[idx[ok]]
    mid = np.full(len(grid), np.nan)
    sel = ok & cont
    mid[sel] = to_ticks((bid[idx[sel]] + ask[idx[sel]]) / 2)
    n0 = until - start + 1
    move = mid[horizon : horizon + n0] - mid[:n0]
    move = move[np.isfinite(move)]
    session = (sec >= start) & (sec <= until)
    trades = df.filter((pl.col("Type") == "T") & (pl.col("StatusMask") == 0) & (pl.col("TradeVolume") > 0))
    in_session = good & session
    return {
        "symbol": symbol,
        "day": day,
        "continuous_fraction": float(cont[:n0].mean()),
        "event_rate": float(np.mean(np.abs(move) >= threshold)) if len(move) else np.nan,
        "spread_ticks": float(np.median(to_ticks(ask[in_session]) - to_ticks(bid[in_session]))) if in_session.any() else np.nan,
        "turnover": float((trades["Price"] * trades["TradeVolume"]).sum() * 1000),
        "mid_price": float(np.median((bid[in_session] + ask[in_session]) / 2)) if in_session.any() else np.nan,
    }


def split_holdout(symbols, fraction, salt):
    order = sorted(symbols, key=lambda s: hashlib.sha256(f"{salt}:{s}".encode()).hexdigest())
    n = round(len(order) * fraction)
    return sorted(order[n:]), sorted(order[:n])


def screen(profile, workers=12) -> dict:
    cfg = profile["universe"]
    rule = cfg["rule"]
    lo, hi = (str(x) for x in cfg["screen_window"])
    days = [d for d in calendar_days(profile) if lo <= d <= hi]
    root = profile["paths"]["market_data"]
    info = Path(profile["paths"]["reference_info"]) / f"{days[0]}.csv"
    with info.open(encoding="utf-8-sig") as stream:
        prices = {r["證券代號"].strip(): float(r["前日收盤價"] or 0) for r in csv.DictReader(stream)}
    candidates = sorted(s for s in os.listdir(root) if len(s) == 4 and s.isdigit() and not s.startswith("0") and prices.get(s, 0.0) >= float(rule["minimum_reference_price"]))
    horizon, threshold = int(rule["event_horizon_seconds"]), float(profile["labels"]["threshold_ticks"])
    start = int(profile["session"]["start"][:2]) * 3600 + int(profile["session"]["start"][2:4]) * 60
    until = int(profile["session"]["until"][:2]) * 3600 + int(profile["session"]["until"][2:4]) * 60
    out = work(profile) / "universe"
    out.mkdir(parents=True, exist_ok=True)
    daily = out / "screen_daily.parquet"
    if daily.is_file():
        frame = pl.read_parquet(daily)
    else:
        jobs = [(root, s, d, horizon, threshold, start, until) for s in candidates for d in days]
        with ProcessPoolExecutor(workers) as pool:
            rows = [r for r in pool.map(_day, jobs, chunksize=8) if r]
        frame = pl.DataFrame(rows)
        frame.write_parquet(daily)
    # A limit-locked day has no two-sided mid, so its rate is NaN. Polars orders
    # NaN above every number; aggregate NaN as missing and require finite values.
    measured = [c for c in frame.columns if c not in ("symbol", "day")]
    frame = frame.with_columns([pl.col(c).fill_nan(None) for c in measured])
    summary = frame.group_by("symbol").agg(
        days=pl.len(),
        turnover=pl.col("turnover").median(),
        spread_ticks=pl.col("spread_ticks").median(),
        continuous_fraction=pl.col("continuous_fraction").median(),
        event_rate=pl.col("event_rate").mean(),
        mid_price=pl.col("mid_price").median(),
    )
    eligible = summary.filter(
        pl.all_horizontal(pl.col(c).is_not_null() for c in ("turnover", "spread_ticks", "continuous_fraction", "event_rate"))
        & (pl.col("days") >= int(rule["minimum_days"]))
        & (pl.col("turnover") >= float(rule["minimum_turnover"]))
        & (pl.col("spread_ticks") <= float(rule["maximum_spread_ticks"]))
        & (pl.col("continuous_fraction") >= float(rule["minimum_continuous_fraction"]))
        & (pl.col("event_rate") >= float(rule["minimum_event_rate"]))
    ).sort(["turnover", "symbol"], descending=[True, False])
    chosen = eligible.head(int(rule["size"]))
    summary.sort("turnover", descending=True).write_csv(out / "screen_summary.csv")
    development, holdout = split_holdout(chosen["symbol"].to_list(), float(cfg["holdout"]["fraction"]), str(cfg["holdout"]["salt"]))
    result = {
        "window": [days[0], days[-1]],
        "screen_days": days,
        "candidates": len(candidates),
        "eligible": eligible.height,
        "rule": rule,
        "chosen": chosen.to_dicts(),
        "development": development,
        "holdout": holdout,
    }
    write_yaml(out / "universe.yaml", result)
    return result
