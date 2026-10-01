"""Root-invoked serial full H10 replay, gated by frozen component evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

OUTPUT = Path(__file__).resolve().parent
KEYS = ["SampleTime", "SampleBookTime", "SampleBookSeq"]


def read(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(record):
    path = Path(record["path"])
    if digest(path) != record["sha256"]:
        raise RuntimeError("Bound native dependency drift: " + str(path))
    return path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    temporary.replace(path)


def successful_status(job):
    status = read(Path(job["work"]) / "status" / (job["day"] + ".yaml"))
    if status.get("status") != "completed" or status.get("fatal_error") is not False or str(status.get("trading_date")) != job["day"]:
        raise RuntimeError("Native day did not complete successfully: " + job["day"])


def first_day_reference(plan, day):
    profile = read(Path(plan["reference_profile"]["path"]))
    parent = Path(profile["paths"]["parent_study"]) / "train" / "rows.parquet"
    mid_path = Path(profile["paths"]["output"]) / "shared" / "mid-train.npy"
    mid_receipt = read(mid_path.with_suffix(".yaml"))
    if not mid_receipt.get("native_verified") or digest(mid_path) != mid_receipt["sha256"]:
        raise RuntimeError("Original physical mid reference is not verified")
    mids = np.load(mid_path, mmap_mode="r")
    if mids.dtype != np.float64 or mids.ndim != 1:
        raise RuntimeError("Original physical mid lost its float64 vector contract")
    parts, selected_mids, offset = [], [], 0
    for batch in pq.ParquetFile(parent).iter_batches(batch_size=32768, columns=["day", "symbol", *KEYS], use_threads=False):
        days = batch.column(0).to_numpy(zero_copy_only=False)
        positions = np.flatnonzero(days == day)
        if len(positions):
            parts.append(pa.Table.from_batches([batch]).take(pa.array(positions)))
            selected_mids.append(np.asarray(mids[offset + positions]))
        offset += len(batch)
    if offset != len(mids) or not parts:
        raise RuntimeError("Original first-day observation/mid reference is incomplete")
    reference = pa.concat_tables(parts)
    return reference.append_column("OriginMidPrice", pa.array(np.concatenate(selected_mids), type=pa.float64()))


def validate_first_day(plan, job):
    reference = first_day_reference(plan, job["day"])
    symbols = reference["symbol"].to_numpy(zero_copy_only=False)
    records = []
    for symbol, path in zip(job["symbols"], job["outputs"], strict=True):
        wanted = reference.take(pa.array(np.flatnonzero(symbols == symbol))).select([*KEYS, "OriginMidPrice"])
        actual = pq.read_table(path, columns=[*KEYS, "OriginMidPrice"], use_threads=False)
        if len(actual) != len(wanted) or len(actual) != 1381:
            raise RuntimeError("First monolithic day changed parent origin count: " + symbol)
        for field in KEYS:
            if actual[field].type != pa.int64() or not actual[field].equals(wanted[field]):
                raise RuntimeError("First monolithic day changed original key: " + symbol + "/" + field)
        for table in (actual, wanted):
            if table["OriginMidPrice"].type != pa.float64() or table["OriginMidPrice"].null_count:
                raise RuntimeError("First monolithic day lost raw float64 mid")
        if not np.array_equal(actual["OriginMidPrice"].to_numpy().view(np.uint64), wanted["OriginMidPrice"].to_numpy().view(np.uint64)):
            raise RuntimeError("First monolithic day changed raw physical mid bits: " + symbol)
        records.append({"symbol": symbol, "path": path, "sha256": digest(Path(path)), "rows": 1381, "exact_parent_keys_and_mid_bits": True})
    if job["symbols"] != plan["universe_order"] or len(records) != 16:
        raise RuntimeError("First monolithic day must prove all sixteen fixed symbols")
    return records


def replay_day(plan, job):
    verify(job["config"])
    for record in plan["raw_inputs"] + plan["basic_info_inputs"]:
        if record.get("day", Path(record["path"]).stem) == job["day"]:
            verify(record)
    for absent in plan["raw_higher_priority_paths_must_remain_absent"]:
        if job["day"] in Path(absent).name and Path(absent).exists():
            raise RuntimeError("Native raw source precedence changed: " + absent)
    if any(Path(path).exists() for path in job["outputs"]):
        raise RuntimeError("Fresh native outputs required; never overwrite day " + job["day"])
    started = time.monotonic()
    with (Path(job["work"]) / "native.log").open("w") as log:
        result = subprocess.run(job["command"], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **job["environment"]}, check=False)
    if result.returncode or not all(Path(path).is_file() for path in job["outputs"]):
        raise RuntimeError("Native replay failed/incomplete: " + job["day"])
    successful_status(job)
    return {
        "day": job["day"],
        "seconds": time.monotonic() - started,
        "config_sha256": job["config"]["sha256"],
        "outputs": [{"path": path, "sha256": digest(Path(path))} for path in job["outputs"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--first-day", action="store_true")
    stage.add_argument("--remaining", action="store_true")
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--native-validation", type=Path, required=True)
    args = parser.parse_args()
    plan_path = OUTPUT / "full-preparation-receipt.yaml"
    plan = read(plan_path)
    for name in ("reference_profile", "calendar", "original_template", "binary", "source_build", "runtime_snapshot", "pilot_validation"):
        verify(plan[name])
    for record in plan["runtime_dependencies"]:
        verify(record)
    lineage, native = read(args.lineage_receipt), read(args.native_validation)
    if lineage.get("passed") is not True or native.get("passed") is not True:
        raise RuntimeError("Separate frozen component lineage and native validation must pass before full replay")
    if lineage.get("h10_binary_sha256") != plan["binary"]["sha256"] or native.get("binary_sha256") != plan["binary"]["sha256"]:
        raise RuntimeError("Authorized lineage/native support belongs to another producer")
    binding = {
        "plan_sha256": digest(plan_path),
        "script_sha256": digest(Path(__file__)),
        "lineage_receipt_sha256": digest(args.lineage_receipt),
        "native_validation_sha256": digest(args.native_validation),
        "method_contract_identity": lineage["method_contract_identity"],
    }
    proof_path = OUTPUT / "full-first-day-validation.yaml"
    journal_path = OUTPUT / "full-replay-state.yaml"
    with (OUTPUT / "full-replay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first = plan["jobs"][0]
        if args.first_day:
            if proof_path.exists() or journal_path.exists():
                raise RuntimeError("First-day proof already exists; no implicit overwrite")
            result = replay_day(plan, first)
            cells = validate_first_day(plan, first)
            write(proof_path, {"schema": "h10-full-first-day-validation-v1", **binding, "passed": True, "day": first["day"], "cells": cells, "labels_read": False, "model_fits": 0})
            write(journal_path, {"schema": "h10-full-serial-replay-v1", **binding, "records": [result], "complete": False})
            print("FIRST_DAY_VALIDATED", first["day"], "all16exactkeys+midbits", round(result["seconds"], 2), "seconds", flush=True)
            return
        proof, journal = read(proof_path), read(journal_path)
        if proof.get("passed") is not True or any(proof.get(key) != value or journal.get(key) != value for key, value in binding.items()):
            raise RuntimeError("First-day proof or resume lineage/config/script binding changed")
        for record in proof["cells"]:
            verify(record)
        completed = {record["day"]: record for record in journal["records"]}
        for job in plan["jobs"]:
            if job["day"] in completed:
                if completed[job["day"]]["config_sha256"] != job["config"]["sha256"]:
                    raise RuntimeError("Resume day config identity changed")
                successful_status(job)
                for record in completed[job["day"]]["outputs"]:
                    verify(record)
                continue
            record = replay_day(plan, job)
            journal["records"].append(record)
            journal["complete"] = len(journal["records"]) == len(plan["jobs"])
            write(journal_path, journal)
            print("NATIVE_DAY", job["day"], round(record["seconds"], 2), "seconds", len(journal["records"]), "/", len(plan["jobs"]), flush=True)


if __name__ == "__main__":
    main()
