"""Shared read-only native projections, including causal float64 origin mids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...contracts import ArtifactRef
from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from ...material import KEYS, partition_frame
from ...store import Store


def native_classes(rows, horizon):
    move = rows[f"mid_return_ticks[{horizon}s]"].to_numpy(dtype=np.float64)
    up = rows[f"mid_endpoint.up.5[{horizon}s]"].to_numpy(dtype=np.float64)
    down = rows[f"mid_endpoint.down.5[{horizon}s]"].to_numpy(dtype=np.float64)
    known = np.isfinite(move)
    if not np.array_equal(known, np.isfinite(up) & np.isfinite(down)):
        raise ContractError("Endpoint move and indicator availability disagree")
    if not np.array_equal(up[known], move[known] >= 5) or not np.array_equal(down[known], move[known] <= -5):
        raise ContractError("Native endpoint indicators disagree with original float64 endpoints")
    labels = np.full(len(rows), -1, dtype=np.int32)
    labels[known] = np.where(up[known] == 1, 1, np.where(down[known] == 1, 2, 0))
    return labels, known


def sampled_mask(rows, interval):
    """Origin thinning uses time alone; histories still use the original 10 s grid."""
    if interval < 10 or interval % 10:
        raise ContractError("Evaluation origins must be an integer multiple of the native 10 s grid")
    return rows.SampleTime.to_numpy(dtype=np.int64) % (interval * 1_000_000) == 0


def load_projection(path):
    path = Path(path)
    manifest = json.loads((path / "projection.json").read_text())
    if any(file_hash(path / filename) != expected for filename, expected in manifest["files"].items()):
        raise ContractError("Native projection differs from its originally published file hashes")
    matrix = np.load(path / "x.npy", mmap_mode="r")
    rows = pd.read_parquet(path / "rows.parquet")
    rows["day"], rows["symbol"] = rows.day.astype(str), rows.symbol.astype(str)
    features = manifest["recipe"]["features"]
    if matrix.shape != (len(rows), len(features)) or rows.duplicated(["day", "symbol", *KEYS]).any():
        raise ContractError("Native projection shape or row identity is invalid")
    if rows.day.min() < "20260101" or (rows.SampleBookTime > rows.SampleTime).any():
        raise ContractError("Native projection crossed its history/causality boundary")
    return matrix, rows, features, manifest


def extract_mids(projection, store, destination):
    """Read one native metadata column; do not regenerate features or labels."""
    matrix, anchors, _, manifest = load_projection(projection)
    del matrix
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = destination.with_suffix(".yaml")
    expected = {"dataset": manifest["recipe"]["dataset"], "anchors": file_hash(Path(projection) / "rows.parquet"), "column": "OriginMidPrice"}
    if destination.exists():
        receipt = read_yaml(receipt_path)
        if receipt["recipe"] != expected or receipt["sha256"] != file_hash(destination):
            raise ContractError("Origin-mid projection changed")
        return np.load(destination, mmap_mode="r")
    dataset = store.resolve(ArtifactRef(**expected["dataset"]))
    blocks = []
    offset = 0
    for number, part in enumerate(sorted(dataset.metadata["partitions"], key=lambda p: (p["day"], p["symbol"]))):
        table = partition_frame(dataset, part, [*KEYS, "OriginMidPrice"])
        table["day"], table["symbol"] = str(part["day"]), str(part["symbol"])
        if not table[["day", "symbol", *KEYS]].reset_index(drop=True).equals(anchors.iloc[offset : offset + len(table)][["day", "symbol", *KEYS]].reset_index(drop=True)):
            raise ContractError("Origin-mid metadata does not match the complete native sample keys")
        blocks.append(table.OriginMidPrice.to_numpy(dtype=np.float64))
        offset += len(table)
        if number % 128 == 0:
            print(f"mid projection {number + 1}/{len(dataset.metadata['partitions'])}", flush=True)
    if offset != len(anchors):
        raise ContractError("Origin-mid projection lost native rows")
    mids = np.concatenate(blocks)
    if not np.isfinite(mids).all() or (mids <= 0).any():
        raise ContractError("Origin mid must be positive and finite on the original native origins")
    temporary = destination.with_suffix(".tmp.npy")
    np.save(temporary, mids)
    temporary.replace(destination)
    write_yaml(receipt_path, {"recipe": expected, "sha256": file_hash(destination), "rows": len(mids), "native_verified": True})
    return mids


def baseline_scope(profile):
    parent = read_yaml(Path(profile["paths"]["parent_study"]) / "protocol.yaml")
    arm = parent["arms"]["current_nominal"]
    return list(arm["features"]), list(parent["native_nominal_features"]), digest(parent)


def prepare(profile):
    output = Path(profile["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    store = Store(Path(profile["paths"]["native_store"]))
    parent = Path(profile["paths"]["parent_study"])
    # Already-exposed forward is development, never sealed OOS. Preparation does
    # not fit, select, score, or feed these outcomes into the hypotheses.
    for role in ("train", "tune", "forward"):
        extract_mids(parent / role, store, output / "shared" / f"mid-{role}.npy")
