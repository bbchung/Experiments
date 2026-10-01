"""Continuous-trading regime from raw partitions (1 s causal grid).

A sample needs a continuous book at its origin; its label needs one at the
endpoint. A symbol-day whose in-session books are mostly non-continuous (a
disposition stock trading by periodic call auction) is a different mechanism and
is excluded wholesale.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from .profile import work

T0, T1 = 8 * 3600 + 59 * 60, 13 * 3600 + 30 * 60
SESSION = (9 * 3600 + 600, 13 * 3600)


def _one(args):
    root, symbol, day = args
    path = Path(root) / symbol / f"{day}.csv.zst"
    if not path.is_file():
        return None
    df = pl.read_csv(
        path,
        columns=["Type", "Timestamp", "StatusMask", "BidPrice1", "AskPrice1"],
        schema_overrides={"Timestamp": pl.Int64, "BidPrice1": pl.Float64, "AskPrice1": pl.Float64},
    ).filter(pl.col("Type") == "B")
    if df.height == 0:
        return None
    ts = df["Timestamp"].to_numpy()
    sec = ((ts // 1_000_000) + 8 * 3600) % 86400 + (ts % 1_000_000) / 1e6
    status = df["StatusMask"].to_numpy()
    bid, ask = df["BidPrice1"].to_numpy(), df["AskPrice1"].to_numpy()
    good = (status == 0) & (bid > 0) & (ask > bid)
    grid = np.arange(T0, T1 + 1)
    idx = np.searchsorted(sec, grid, side="right") - 1
    cont = np.zeros(len(grid), bool)
    ok = idx >= 0
    cont[ok] = good[idx[ok]]
    # The last event's state does not extend past the end of the recording.
    cont[grid > sec[-1] + 60] = False
    session = (sec >= SESSION[0]) & (sec <= SESSION[1])
    return {
        "symbol": symbol,
        "day": day,
        "books_session": int(session.sum()),
        "books_continuous": int((session & good).sum()),
        "grid": cont,
    }


def build(profile, days, symbols, workers=16):
    """Write work/regime/<day>.npz (continuous flags per symbol) and regime.parquet."""
    out = work(profile) / "regime"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(profile["paths"]["market_data"], s, d) for d in days for s in symbols if not (out / f"{d}.npz").exists()]
    per_day, rows = {}, []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_one, jobs, chunksize=4):
            if r is None:
                continue
            per_day.setdefault(r["day"], {})[r["symbol"]] = r.pop("grid")
            rows.append(r)
    for day, flags in per_day.items():
        np.savez_compressed(out / f"{day}.npz", **flags)
    table = out / "regime.parquet"
    frame = pl.DataFrame(rows) if rows else None
    if table.exists() and frame is not None:
        frame = pl.concat([pl.read_parquet(table), frame]).unique(["symbol", "day"], keep="last")
    if frame is not None:
        frame.write_parquet(table)


def auction_days(profile, threshold=0.5) -> set:
    frame = pl.read_parquet(work(profile) / "regime" / "regime.parquet")
    frame = frame.with_columns(frac=pl.col("books_continuous") / pl.col("books_session").clip(1))
    bad = frame.filter(pl.col("frac") < threshold)
    return set(zip(bad["symbol"].to_list(), bad["day"].to_list()))


_CACHE: dict = {}


def flags(profile, day) -> dict:
    if day not in _CACHE:
        _CACHE[day] = dict(np.load(work(profile) / "regime" / f"{day}.npz"))
    return _CACHE[day]


def continuous_at(profile, day, symbol, seconds_of_day) -> np.ndarray:
    grid = flags(profile, day).get(symbol)
    if grid is None:
        return np.zeros(len(seconds_of_day), bool)
    return grid[np.clip(np.asarray(seconds_of_day) - T0, 0, len(grid) - 1)]
