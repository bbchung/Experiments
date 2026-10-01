"""Type projections over canonical native material, never Python feature formulas."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from AstraResearch.io import ContractError, file_hash, read_yaml, write_yaml
from AstraResearch.material import KEYS, fragments, partition_frame

HORIZONS = (60, 120, 180, 300)


def numeric_domain(dataset, features, directory):
    """Freeze column eligibility from train; never saturate native finite values."""
    if dataset.metadata["role"] not in {"train", "engineering_smoke"}:
        raise ContractError("Model numeric-domain selection can only inspect train")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "numeric-domain.yaml"
    recipe = {"dataset": dataset.ref.document(), "features": features, "domain": "float32", "minimum_finite": 100}
    if path.exists():
        result = read_yaml(path)
        if result["recipe"] != recipe:
            raise ContractError("Numeric-domain scan belongs to another native recipe")
        return result["supported"]
    parts = sorted(dataset.metadata["partitions"], key=lambda part: (part["day"], part["symbol"]))
    finite, oversized = np.zeros(len(features), dtype=np.int64), np.zeros(len(features), dtype=np.int64)
    low, high = np.full(len(features), np.inf), np.full(len(features), -np.inf)
    special, examples = np.zeros((3, len(features)), dtype=np.int64), {}
    maximum = np.finfo(np.float32).max
    for serial, part in enumerate(parts):
        values = partition_frame(dataset, part, features).to_numpy(dtype=float)
        valid = np.isfinite(values)
        outside = valid & (np.abs(values) > maximum)
        finite += valid.sum(axis=0)
        oversized += outside.sum(axis=0)
        low = np.minimum(low, np.where(valid, values, np.inf).min(axis=0))
        high = np.maximum(high, np.where(valid, values, -np.inf).max(axis=0))
        special += np.vstack([np.isneginf(values).sum(axis=0), np.isnan(values).sum(axis=0), np.isposinf(values).sum(axis=0)])
        for column in np.flatnonzero(outside.any(axis=0)):
            name = features[column]
            if name not in examples:
                examples[name] = {"day": part["day"], "symbol": part["symbol"], "native_values": values[outside[:, column], column][:5].tolist()}
        if serial % 64 == 0:
            print(f"train numeric domain: {serial + 1}/{len(parts)} partitions", flush=True)
    nonconstant = (low < high) | ((np.vstack([finite, special]) > 0).sum(axis=0) > 1)
    supported = [name for index, name in enumerate(features) if not oversized[index] and finite[index] >= 100 and nonconstant[index]]
    pd.DataFrame(
        {
            "feature": features,
            "finite": finite,
            "warmup": special[0],
            "nan": special[1],
            "positive_infinity": special[2],
            "minimum": low,
            "maximum": high,
            "finite_outside_float32": oversized,
            "nonconstant": nonconstant,
            "supported": [name in supported for name in features],
        }
    ).to_csv(directory / "numeric-domain.csv", index=False)
    write_yaml(
        path,
        {
            "recipe": recipe,
            "supported": supported,
            "excluded_overflow": {name: {"train_rows": int(oversized[index]), **examples[name]} for index, name in enumerate(features) if oversized[index]},
            "excluded_unavailable_or_constant": [name for index, name in enumerate(features) if not oversized[index] and (finite[index] < 100 or not nonconstant[index])],
            "population": "all_native_origins_preserved_column_selection_only",
        },
    )
    return supported


def nonoverlap(day, symbol, times, horizon):
    """Fix origins from native time alone, before reading scores or outcomes."""
    last, keep = {}, np.zeros(len(times), dtype=bool)
    for index, (d, s, t) in enumerate(zip(day, symbol, times, strict=True)):
        group = str(d), str(s)
        if int(t) >= last.get(group, 0) + horizon * 1_000_000:
            keep[index] = True
            last[group] = int(t)
    return keep


def project(dataset, features, directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "projection.json"
    expected = {"dataset": dataset.ref.document(), "features": features, "horizons": list(HORIZONS), "cast": "native_numeric_to_float32_preserve_sentinels"}
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored.get("recipe") != expected or any(file_hash(directory / name) != signature for name, signature in stored["files"].items()):
            raise ContractError("Projection belongs to another native recipe")
        return np.load(directory / "x.npy", mmap_mode="r"), pd.read_parquet(directory / "rows.parquet"), pd.read_csv(directory / "health.csv")
    parts = sorted(dataset.metadata["partitions"], key=lambda part: (part["day"], part["symbol"]))
    count = sum(pq.ParquetFile(dataset.file(fragments(part)[0])).metadata.num_rows for part in parts)
    matrix = np.lib.format.open_memmap(directory / "x.npy", mode="w+", dtype=np.float32, shape=(count, len(features)))
    low, high = np.full(len(features), np.inf), np.full(len(features), -np.inf)
    finite = np.zeros(len(features), dtype=np.int64)
    special = np.zeros((3, len(features)), dtype=np.int64)
    labels = [name for horizon in HORIZONS for name in (f"mid_return_ticks[{horizon}s]", f"mid_endpoint.up.5[{horizon}s]", f"mid_endpoint.down.5[{horizon}s]")]
    rows, offset = [], 0
    for serial, part in enumerate(parts):
        table = partition_frame(dataset, part, [*features, *KEYS, *labels])
        values = table[features].to_numpy(dtype=float)
        valid = np.isfinite(values)
        with np.errstate(over="ignore", invalid="ignore"):
            cast = values.astype(np.float32)
        if np.any(valid & ~np.isfinite(cast)):
            columns = [features[index] for index in np.flatnonzero((valid & ~np.isfinite(cast)).any(axis=0))]
            raise ContractError(f"Finite native features exceeded the train-frozen model float32 domain: {columns}; day={part['day']} symbol={part['symbol']}")
        matrix[offset : offset + len(table)] = cast
        finite += valid.sum(axis=0)
        low = np.minimum(low, np.where(valid, values, np.inf).min(axis=0))
        high = np.maximum(high, np.where(valid, values, -np.inf).max(axis=0))
        special += np.vstack([np.isneginf(values).sum(axis=0), np.isnan(values).sum(axis=0), np.isposinf(values).sum(axis=0)])
        origin = table[[*KEYS, *labels]].copy()
        origin["day"], origin["symbol"] = part["day"], part["symbol"]
        rows.append(origin)
        offset += len(table)
        if serial % 32 == 0:
            print(f"projection {dataset.metadata['role']}: {serial + 1}/{len(parts)} partitions", flush=True)
    matrix.flush()
    row_data = pd.concat(rows, ignore_index=True)
    if offset != count or row_data.duplicated(["day", "symbol", *KEYS]).any() or row_data.day.min() < "20260101":
        raise ContractError("Native projection lost origins or crossed the date floor")
    states = (np.vstack([finite, special]) > 0).sum(axis=0)
    health = pd.DataFrame(
        {
            "feature": features,
            "finite": finite,
            "warmup": special[0],
            "nan": special[1],
            "positive_infinity": special[2],
            "minimum": low,
            "maximum": high,
            "nonconstant": (low < high) | (states > 1),
        }
    )
    health.to_csv(directory / "health.csv", index=False)
    row_data.to_parquet(directory / "rows.parquet", index=False)
    manifest.write_text(json.dumps({"recipe": expected, "files": {name: file_hash(directory / name) for name in ("x.npy", "rows.parquet", "health.csv")}}, indent=2) + "\n")
    return matrix, row_data, health


def native_classes(rows, horizon):
    y = rows[f"mid_return_ticks[{horizon}s]"].to_numpy(dtype=float)
    up = rows[f"mid_endpoint.up.5[{horizon}s]"].to_numpy(dtype=float)
    down = rows[f"mid_endpoint.down.5[{horizon}s]"].to_numpy(dtype=float)
    known = np.isfinite(y) & np.isfinite(up) & np.isfinite(down)
    if not np.array_equal(np.isfinite(y), known) or not np.isin(up[known], [0, 1]).all() or not np.isin(down[known], [0, 1]).all():
        raise ContractError("Native endpoint class states differ")
    if np.any((up[known] + down[known]) > 1) or not np.array_equal(up[known], y[known] >= 5) or not np.array_equal(down[known], y[known] <= -5):
        raise ContractError("Native endpoint classes disagree with their native signed endpoint")
    # Encode the two native indicator columns; no endpoint arithmetic is produced here.
    classes = np.full(len(y), -1, dtype=np.int32)
    classes[known] = np.where(up[known] == 1, 1, np.where(down[known] == 1, 2, 0))
    return classes, known


def project_nominal(dataset, features, directory, anchors):
    """Keep native nominal strings and the existing missing token; no ordinal map."""
    path = directory / "nominal.parquet"
    manifest = directory / "nominal-projection.yaml"
    recipe = {"dataset": dataset.ref.document(), "features": features, "source": "native_nominal_strings"}
    if path.exists():
        if not manifest.exists():
            raise ContractError("Nominal projection lacks its native recipe manifest")
        stored = read_yaml(manifest)
        if stored["recipe"] != recipe or stored["sha256"] != file_hash(path):
            raise ContractError("Nominal projection belongs to another native recipe")
        result = pd.read_parquet(path)
    else:
        tables = []
        for part in sorted(dataset.metadata["partitions"], key=lambda part: (part["day"], part["symbol"])):
            table = partition_frame(dataset, part, [*features, *KEYS])
            table["day"], table["symbol"] = part["day"], part["symbol"]
            tables.append(table)
        result = pd.concat(tables, ignore_index=True)
        result.to_parquet(path, index=False)
        write_yaml(manifest, {"recipe": recipe, "sha256": file_hash(path)})
    if not result[["day", "symbol", *KEYS]].equals(anchors[["day", "symbol", *KEYS]]):
        raise ContractError("Nominal native inputs differ from numeric row anchors")
    values = result[features].astype(object).fillna("NaN")
    if any(any(not isinstance(value, str) for value in values[feature].unique()) for feature in features):
        raise ContractError("Native nominal inputs must remain strings")
    return values


def census(rows, directory, role):
    entries = []
    for horizon in HORIZONS:
        classes, valid = native_classes(rows, horizon)
        independent = nonoverlap(rows.day.to_numpy(), rows.symbol.to_numpy(), rows.SampleTime.to_numpy(), horizon)
        for (day, symbol), indices in rows.groupby(["day", "symbol"], sort=True).indices.items():
            ok = valid[indices]
            keep = independent[indices] & ok
            entries.append(
                {
                    "role": role,
                    "day": str(day),
                    "symbol": str(symbol),
                    "horizon": horizon,
                    "origins": len(indices),
                    "known": int(ok.sum()),
                    "up": int((classes[indices][ok] == 1).sum()),
                    "down": int((classes[indices][ok] == 2).sum()),
                    "nonoverlap_known": int(keep.sum()),
                    "nonoverlap_up": int((classes[indices][keep] == 1).sum()),
                    "nonoverlap_down": int((classes[indices][keep] == 2).sum()),
                }
            )
    result = pd.DataFrame(entries)
    result.to_csv(directory / f"census-{role}.csv", index=False)
    return result


def tail_selection(matrix, rows, features, eligible, output, seats=400):
    """Train-only endpoint MI, checked in a later chronological train subperiod."""
    independent = nonoverlap(rows.day.to_numpy(), rows.symbol.to_numpy(), rows.SampleTime.to_numpy(), 300)
    target, known = native_classes(rows, 300)
    days = sorted(rows.day.unique())
    boundary = days[max(1, int(len(days) * 0.7))]
    fit = np.flatnonzero(independent & known & (rows.day.to_numpy() < boundary))
    confirm = np.flatnonzero(independent & known & (rows.day.to_numpy() >= boundary))

    def information(codes, labels):
        counts = np.bincount(codes * 3 + labels, minlength=48).reshape(16, 3).astype(float)
        probability = counts / counts.sum()
        denominator = probability.sum(axis=1)[:, None] * probability.sum(axis=0)[None, :]
        ok = probability > 0
        return float(np.sum(probability[ok] * np.log(probability[ok] / denominator[ok])))

    def codes(values, edges):
        result = np.searchsorted(edges, values, side="right").clip(0, 12)
        result[np.isneginf(values)], result[np.isnan(values)], result[np.isposinf(values)] = 13, 14, 15
        return result

    records = []
    for index, name in enumerate(features):
        if name not in eligible:
            continue
        a, b = matrix[fit, index], matrix[confirm, index]
        normal = a[np.isfinite(a)]
        if len(normal) < 100:
            continue
        edges = np.unique(np.quantile(normal, np.arange(1, 13) / 13))
        fit_mi, confirmation_mi = information(codes(a, edges), target[fit]), information(codes(b, edges), target[confirm])
        records.append({"feature": name, "fit_mi": fit_mi, "confirmation_mi": confirmation_mi, "rank": min(fit_mi, confirmation_mi)})
    table = pd.DataFrame(records).sort_values(["rank", "feature"], ascending=[False, True])
    table.to_csv(output / "tail-selector.csv", index=False)
    selected, per_module = [], {}
    for name in table.feature:
        module = name.split(".")[0]
        if per_module.get(module, 0) < 12:
            selected.append(name)
            per_module[module] = per_module.get(module, 0) + 1
        if len(selected) == seats:
            break
    write_yaml(
        output / "tail-selector.yaml",
        {
            "features": selected,
            "procedure": "native_endpoint_ternary_MI_first_70_percent_train_confirm_last_30_percent_train",
            "selection_rows": {"fit": len(fit), "confirmation": len(confirm)},
            "seats": seats,
            "module_cap": 12,
            "normal_quantile_bins": 13,
            "sentinel_bins": 3,
            "boundary": str(boundary),
        },
    )
    return selected
