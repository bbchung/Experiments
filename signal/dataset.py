"""Column-major float32 matrices of a feature set per role.

The base set (with labels) defines the rows of a role: its symbol-days outside
the auction regime, origins with a continuous book, every `stride`-th sample.
Every other feature set is cached at exactly those (day, symbol, time) rows.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import regime
from .profile import work

KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq", "origin_mid_ticks")


def label_name(h):
    return f"mid_ret_{h}[{h}s]"


def seconds_of_day(t_us):
    return ((np.asarray(t_us) // 1_000_000) + 8 * 3600) % 86400


def numeric_columns(material: Path) -> list[str]:
    sample = min((material / "data").glob("*/*/values.parquet"))
    return [f.name for f in pq.read_schema(sample) if f.name not in KEYS and not f.name.startswith("mid_ret_") and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))]


def cache_dir(profile, set_name, role) -> Path:
    return work(profile) / "cache" / set_name / role


def build_base(profile, set_name, role, days, stride):
    material = work(profile) / "material" / set_name
    out = cache_dir(profile, set_name, role)
    out.mkdir(parents=True, exist_ok=True)
    columns = numeric_columns(material)
    horizons = profile["target"]["horizons"]
    labels = [label_name(h) for h in horizons]
    bad = regime.auction_days(profile)
    parts = [(d, p.parent.name, p) for d in days for p in sorted((material / "data" / d).glob("*/values.parquet")) if (p.parent.name, d) not in bad]

    def rows_of(part):
        day, sym, path = part
        t = pq.read_table(path, columns=["SampleTime"]).column(0).to_numpy()[::stride]
        return np.flatnonzero(regime.continuous_at(profile, day, sym, seconds_of_day(t))) * stride

    with ThreadPoolExecutor(16) as pool:
        keep = list(pool.map(rows_of, parts))
    n = sum(len(k) for k in keep)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(n, len(columns)), fortran_order=True)
    Y = np.full((n, len(labels)), np.nan, np.float32)
    day_a, sym_a, t_a = np.empty(n, "U8"), np.empty(n, "U8"), np.empty(n, np.int64)
    offsets = np.cumsum([0] + [len(k) for k in keep])

    def fill(i):
        day, sym, path = parts[i]
        rows = keep[i]
        if not len(rows):
            return
        present = set(pq.read_schema(path).names)
        table = pq.read_table(path, columns=["SampleTime", *[l for l in labels if l in present], *columns])
        a, b = offsets[i], offsets[i + 1]
        block = np.empty((len(rows), len(columns)), np.float32)
        for j, c in enumerate(columns):
            block[:, j] = table.column(c).to_numpy(zero_copy_only=False)[rows]
        X[a:b] = block
        for j, l in enumerate(labels):
            if l in present:
                Y[a:b, j] = table.column(l).to_numpy(zero_copy_only=False)[rows]
        day_a[a:b], sym_a[a:b] = day, sym
        t_a[a:b] = table.column("SampleTime").to_numpy()[rows]

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(fill, range(len(parts))))
    X.flush()
    np.savez(out / "meta.npz", Y=Y, day=day_a, symbol=sym_a, time=t_a, horizons=np.asarray(horizons))
    (out / "columns.json").write_text(json.dumps(columns))
    return n, len(columns)


def build_aligned(profile, set_name, role, base_set):
    """Cache `set_name` at the rows of the base cache for `role`."""
    ref = open_cache(profile, base_set, role)
    material = work(profile) / "material" / set_name
    columns = numeric_columns(material)
    out = cache_dir(profile, set_name, role)
    out.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(len(ref["day"]), len(columns)), fortran_order=True)
    groups: dict = {}
    for i, key in enumerate(zip(ref["day"].tolist(), ref["symbol"].tolist())):
        groups.setdefault(key, []).append(i)
    missing = []

    def fill(key):
        day, sym = key
        rows = np.asarray(groups[key])
        path = material / "data" / day / sym / "values.parquet"
        block = np.full((len(rows), len(columns)), np.nan, np.float32)
        if path.is_file():
            table = pq.read_table(path, columns=["SampleTime", *columns])
            t = table.column("SampleTime").to_numpy()
            want = ref["time"][rows]
            pos = np.searchsorted(t, want)
            ok = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == want)
            for j, c in enumerate(columns):
                block[ok, j] = table.column(c).to_numpy(zero_copy_only=False)[pos[ok]]
            if not ok.all():
                missing.append((day, sym, int((~ok).sum())))
        else:
            missing.append((day, sym, len(rows)))
        X[rows] = block

    with ThreadPoolExecutor(12) as pool:
        list(pool.map(fill, list(groups)))
    X.flush()
    np.savez(out / "meta.npz", day=ref["day"], symbol=ref["symbol"], time=ref["time"])
    (out / "columns.json").write_text(json.dumps(columns))
    return missing


def open_cache(profile, set_name, role) -> dict:
    d = cache_dir(profile, set_name, role)
    meta = dict(np.load(d / "meta.npz"))
    return {"X": np.load(d / "X.npy", mmap_mode="r"), "columns": json.loads((d / "columns.json").read_text()), **meta}


def load_role(profile, role, base_set, selection: dict[str, list[str] | None]) -> dict:
    """Materialize selected columns from several aligned sets: {set: columns or None (all)}."""
    base = open_cache(profile, base_set, role)
    blocks, names = [], []
    for set_name, cols in selection.items():
        cache = base if set_name == base_set else open_cache(profile, set_name, role)
        if set_name != base_set and not np.array_equal(cache["time"], base["time"]):
            raise RuntimeError(f"{set_name} rows differ from {base_set} ({role})")
        cols = cache["columns"] if cols is None else cols
        index = {c: j for j, c in enumerate(cache["columns"])}
        block = np.empty((len(base["day"]), len(cols)), np.float32)
        for k, c in enumerate(cols):
            block[:, k] = cache["X"][:, index[c]]
        blocks.append(block)
        names += [f"{set_name}:{c}" for c in cols]
    X = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    X[np.isneginf(X)] = -1e30
    X[np.isposinf(X)] = 1e30
    horizons = [int(h) for h in base["horizons"]]
    sec = seconds_of_day(base["time"])
    ok = {h: np.isfinite(base["Y"][:, k]) for k, h in enumerate(horizons)}
    for day in np.unique(base["day"]):
        md = base["day"] == day
        for sym in np.unique(base["symbol"][md]):
            m = md & (base["symbol"] == sym)
            for h in horizons:
                ok[h][m] &= regime.continuous_at(profile, day, sym, sec[m] + h)
    return {"X": X, "columns": names, "Y": base["Y"], "day": base["day"], "symbol": base["symbol"], "time": base["time"], "horizons": horizons, "ok": ok}
