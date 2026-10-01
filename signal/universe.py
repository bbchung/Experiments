"""Universe screen over pre-development days only.

Per candidate stock and screen day (raw CSV, continuous books): median turnover,
mean spread in ticks, and the rate of |mid change| >= threshold ticks at the
screen horizon on a 1 s grid. Stocks liquid enough with a usable spread are
ranked by that base rate; the top N are frozen for the whole study.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from .profile import work

# TWSE stock ladder: (lower price bound, tick).
LADDER = [(0.0, 0.01), (10.0, 0.05), (50.0, 0.1), (100.0, 0.5), (500.0, 1.0), (1000.0, 5.0)]
BOUNDS = np.array([b for b, _ in LADDER])
SIZES = np.array([s for _, s in LADDER])
BASE = np.concatenate([[0.0], np.cumsum(np.diff(BOUNDS) / SIZES[:-1])])


def to_ticks(price):
    band = np.searchsorted(BOUNDS, price, side="right") - 1
    return BASE[band] + (price - BOUNDS[band]) / SIZES[band]


def _day(args):
    root, symbol, day, horizon, threshold = args
    path = Path(root) / symbol / f"{day}.csv.zst"
    if not path.is_file():
        return None
    df = pl.read_csv(
        path,
        columns=["Type", "Timestamp", "StatusMask", "Price", "TradeVolume", "BidPrice1", "AskPrice1"],
        schema_overrides={"Price": pl.Float64, "BidPrice1": pl.Float64, "AskPrice1": pl.Float64, "TradeVolume": pl.Float64, "Timestamp": pl.Int64},
    ).filter(pl.col("StatusMask") == 0)
    trades = df.filter(pl.col("Type") == "T")
    books = df.filter((pl.col("Type") == "B") & (pl.col("BidPrice1") > 0) & (pl.col("AskPrice1") > pl.col("BidPrice1")))
    if books.height < 100:
        return None
    ts = books["Timestamp"].to_numpy()
    sec = ((ts // 1_000_000) + 8 * 3600) % 86400 + (ts % 1_000_000) / 1e6
    bid, ask = books["BidPrice1"].to_numpy(), books["AskPrice1"].to_numpy()
    mid = to_ticks((bid + ask) / 2)
    grid = np.arange(9 * 3600 + 600, 13 * 3600 + horizon + 1, 1.0)
    idx = np.searchsorted(sec, grid, side="right") - 1
    gm = np.where(idx >= 0, mid[np.clip(idx, 0, None)], np.nan)
    d = gm[horizon:] - gm[:-horizon]
    d = d[: (13 * 3600 - (9 * 3600 + 600)) + 1]
    d = d[np.isfinite(d)]
    session = (sec >= 9 * 3600 + 600) & (sec <= 13 * 3600)
    return {
        "symbol": symbol,
        "day": day,
        "rate": float(np.mean(np.abs(d) >= threshold)) if len(d) else np.nan,
        "spread": float(np.mean(to_ticks(ask[session]) - to_ticks(bid[session]))) if session.any() else np.nan,
        "turnover": float((trades["Price"] * trades["TradeVolume"]).sum() * 1000),
    }


def screen(profile, days: list[str]) -> dict:
    cfg = profile["universe"]["screen"]
    root = profile["paths"]["market_data"]
    info = Path(cfg["reference_info"]) / f"{days[0]}.csv" if cfg.get("reference_info") else None
    symbols = sorted(s for s in os.listdir(root) if len(s) == 4 and s.isdigit() and not s.startswith("0"))
    if info and info.is_file():
        with info.open(encoding="utf-8-sig") as stream:
            prices = {r["證券代號"].strip(): float(r["前日收盤價"] or 0) for r in csv.DictReader(stream)}
        symbols = [s for s in symbols if prices.get(s, 0) >= cfg.get("min_price", 10)]
    jobs = [(root, s, d, int(cfg["horizon_seconds"]), float(profile["target"]["threshold_ticks"])) for s in symbols for d in days]
    rows = []
    with ProcessPoolExecutor(24) as pool:
        rows = [r for r in pool.map(_day, jobs, chunksize=16) if r]
    frame = pl.DataFrame(rows)
    summary = frame.group_by("symbol").agg(days=pl.len(), turnover=pl.col("turnover").median(), spread=pl.col("spread").median(), rate=pl.col("rate").mean())
    chosen = (
        summary.filter(
            (pl.col("days") >= cfg["min_days"]) & (pl.col("turnover") >= cfg["min_turnover"]) & (pl.col("spread") <= cfg["max_spread_ticks"]) & (pl.col("rate") >= cfg["min_rate"])
        )
        .sort(["rate", "symbol"], descending=[True, False])
        .head(cfg["top"])
    )
    out = work(profile) / "universe"
    out.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out / "screen_daily.parquet")
    result = {"window": [days[0], days[-1]], "symbols": chosen["symbol"].to_list(), "rule": {k: v for k, v in cfg.items() if k != "window"}}
    (out / "universe.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    return result


def symbols(profile) -> list[str]:
    fixed = profile["universe"].get("symbols")
    if fixed:
        return list(fixed)
    path = work(profile) / "universe" / "universe.yaml"
    if not path.exists():
        raise RuntimeError("run `screen` first or freeze universe.symbols in the profile")
    return yaml.safe_load(path.read_text())["symbols"]
