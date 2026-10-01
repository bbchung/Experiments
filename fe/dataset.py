"""Shared evaluation rows (cohort) and aligned feature matrices per set.

The anchor set defines rows: every 30 s origin of a material (day, symbol) whose
symbol-day is mostly continuous and whose origin second is continuous. A label
is usable at horizon h only if it is finite and the endpoint second is
continuous. Feature availability never removes a row. Each other set is cached
at exactly the cohort rows (missing partitions or rows stay NaN).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..signal import regime
from . import profile as P

KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")


def label_name(h):
    return f"mid_ret_{h}[{h}s]"


def seconds_of_day(t_us):
    return ((np.asarray(t_us) // 1_000_000) + 8 * 3600) % 86400


def material_root(profile, set_name) -> Path:
    return P.work(profile) / "material" / set_name


def cohort_path(profile) -> Path:
    return P.work(profile) / "cohort.npz"


def build_cohort(profile, jobs):
    root = material_root(profile, "anchor")
    horizons = [int(h) for h in profile["labels"]["horizons"]]
    bad = regime.auction_days(profile)
    parts = [(d, s) for d, s in sorted(jobs) if (s, d) not in bad and (root / "data" / d / s / "values.parquet").is_file()]

    def load(part):
        day, sym = part
        t = pq.read_table(root / "data" / day / sym / "values.parquet", columns=["SampleTime", "origin_mid_ticks", *[label_name(h) for h in horizons]])
        time = t.column("SampleTime").to_numpy()
        sec = seconds_of_day(time)
        keep = regime.continuous_at(profile, day, sym, sec)
        y = np.column_stack([t.column(label_name(h)).to_numpy() for h in horizons]).astype(np.float64)
        valid = np.column_stack([np.isfinite(y[:, k]) & regime.continuous_at(profile, day, sym, sec + h) for k, h in enumerate(horizons)])
        return day, sym, time[keep], y[keep], valid[keep], t.column("origin_mid_ticks").to_numpy()[keep]

    with ThreadPoolExecutor(12) as pool:
        loaded = list(pool.map(load, parts))
    day = np.concatenate([np.full(len(x[2]), x[0]) for x in loaded])
    sym = np.concatenate([np.full(len(x[2]), x[1]) for x in loaded])
    out = {
        "day": day.astype("U8"),
        "symbol": sym.astype("U8"),
        "time": np.concatenate([x[2] for x in loaded]),
        "y": np.concatenate([x[3] for x in loaded]),
        "valid": np.concatenate([x[4] for x in loaded]),
        "origin_mid_ticks": np.concatenate([x[5] for x in loaded]),
        "horizons": np.asarray(horizons),
    }
    np.savez(cohort_path(profile), **out)
    return {"rows": len(out["time"]), "partitions": len(parts), "excluded_auction_days": sorted(f"{s}@{d}" for s, d in bad)}


def load_cohort(profile) -> dict:
    return dict(np.load(cohort_path(profile)))


def cache_dir(profile, set_name) -> Path:
    return P.work(profile) / "cache" / set_name


def build_cache(profile, set_name):
    """float32 numeric matrix and int32 categorical codes of SET at the cohort rows."""
    c = load_cohort(profile)
    root = material_root(profile, set_name)
    sample = next(iter(sorted((root / "data").glob("*/*/values.parquet"))))
    schema = pq.read_schema(sample)
    numeric = [
        f.name
        for f in schema
        if f.name not in KEYS and not f.name.startswith("mid_ret_") and f.name != "origin_mid_ticks" and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))
    ]
    categorical = [f.name for f in schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type) or pa.types.is_dictionary(f.type)]
    out = cache_dir(profile, set_name)
    out.mkdir(parents=True, exist_ok=True)
    n = len(c["time"])
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(n, len(numeric)))
    C = np.full((n, len(categorical)), -1, np.int32)
    vocab = [{} for _ in categorical]
    groups: dict = {}
    for i, key in enumerate(zip(c["day"].tolist(), c["symbol"].tolist())):
        groups.setdefault(key, []).append(i)
    missing = []

    def read(key):
        day, sym = key
        rows = np.asarray(groups[key])
        path = root / "data" / day / sym / "values.parquet"
        block = np.full((len(rows), len(numeric)), np.nan, np.float32)
        cats = None
        ok = np.zeros(len(rows), bool)
        pos = None
        if path.is_file():
            table = pq.read_table(path)
            t = table.column("SampleTime").to_numpy()
            want = c["time"][rows]
            pos = np.searchsorted(t, want)
            ok = (pos < len(t)) & (t[np.minimum(pos, len(t) - 1)] == want)
            present = set(table.schema.names)
            for j, name in enumerate(numeric):
                if name in present:
                    block[ok, j] = table.column(name).to_numpy(zero_copy_only=False)[pos[ok]]
            cats = []
            for name in categorical:
                if name not in present:
                    cats.append(None)
                    continue
                encoded = table.column(name).combine_chunks().dictionary_encode()
                indices = encoded.indices.to_numpy(zero_copy_only=False)
                indices = np.where(encoded.indices.is_valid().to_numpy(zero_copy_only=False), indices, -1)
                cats.append((encoded.dictionary.to_pylist(), indices))
        return key, rows, block, cats, ok, pos

    with ThreadPoolExecutor(10) as pool:
        for key, rows, block, cats, ok, pos in pool.map(read, list(groups)):
            X[rows] = block
            if cats is not None:
                for j, item in enumerate(cats):
                    if item is None:
                        continue
                    words, indices = item
                    codes = vocab[j]
                    mapping = np.asarray([codes.setdefault(w, len(codes)) for w in words] + [-1], np.int32)
                    col = np.full(len(rows), -1, np.int32)
                    local = indices[pos[ok]]
                    col[ok] = mapping[np.where(local >= 0, local, len(words))]
                    C[rows, j] = col
            if not ok.all():
                missing.append((key[0], key[1], int((~ok).sum())))
    X.flush()
    np.save(out / "C.npy", C)
    (out / "columns.json").write_text(json.dumps({"numeric": numeric, "categorical": categorical, "vocab": [list(v) for v in vocab]}))
    (out / "missing.json").write_text(json.dumps(missing))
    return {"rows": n, "numeric": len(numeric), "categorical": len(categorical), "missing_partitions_or_rows": len(missing)}


def open_cache(profile, set_name) -> dict:
    d = cache_dir(profile, set_name)
    meta = json.loads((d / "columns.json").read_text())
    return {"X": np.load(d / "X.npy", mmap_mode="r"), "C": np.load(d / "C.npy", mmap_mode="r"), **meta}
