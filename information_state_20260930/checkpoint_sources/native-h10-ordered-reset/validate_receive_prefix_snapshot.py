"""Read-only exact native receive-prefix proof; no replay, model or labels."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

OUTPUT = Path(__file__).resolve().parent
PREFIX = OUTPUT / "receive-prefix"
KEYS = ["SampleTime", "SampleBookTime", "SampleBookSeq"]


def read(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    value = digest(path)
    if expected is not None and value != expected:
        raise RuntimeError("Bound receive-prefix dependency changed: " + str(path))
    return {"path": str(path), "sha256": value}


def same_bits(left, right):
    if left.type != right.type or len(left) != len(right) or not left.is_null().equals(right.is_null()):
        return False
    if pa.types.is_float64(left.type):
        return np.array_equal(left.to_numpy().view(np.uint64), right.to_numpy().view(np.uint64))
    return left.equals(right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PREFIX / "validation-snapshot.yaml")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise RuntimeError("Fresh prefix validation receipt required; no overwrite")
    preparation_path = PREFIX / "preparation-receipt.yaml"
    preparation = read(preparation_path)
    if not preparation.get("genuine_straddle_found"):
        raise RuntimeError("No genuine source support; do not manufacture a clock")
    config = Path(preparation["config"]["path"])
    work = config.parent
    status_path = work / "status" / (preparation["day"] + ".yaml")
    status = read(status_path)
    if status.get("status") != "completed" or status.get("fatal_error") is not False:
        raise RuntimeError("Root native prefix replay has not completed successfully")
    full_path = Path(preparation["source_full_native"]["path"])
    prefix_path = work / "data" / preparation["day"] / preparation["symbol"] / "values.parquet"
    full, truncated = (pq.read_table(path, use_threads=False) for path in (full_path, prefix_path))
    errors = []
    if any(name.startswith(("mid_return_ticks[", "sticky_return_ticks[", "mid_endpoint.")) for name in full.column_names + truncated.column_names):
        raise RuntimeError("Receive-prefix proof must remain feature-only")
    if not full.schema.equals(truncated.schema, check_metadata=True):
        errors.append("native schema or metadata changed")
    for table in (full, truncated):
        if not all(table[key].type == pa.int64() and table[key].null_count == 0 for key in KEYS):
            errors.append("native key contract changed")
    full_keys = list(zip(*(full[key].to_pylist() for key in KEYS), strict=True))
    prefix_keys = list(zip(*(truncated[key].to_pylist() for key in KEYS), strict=True))
    if len(full_keys) != len(set(full_keys)) or len(prefix_keys) != len(set(prefix_keys)):
        errors.append("duplicate native origin keys")
    positions = {key: index for index, key in enumerate(full_keys)}
    if not prefix_keys or any(key not in positions for key in prefix_keys):
        errors.append("truncated replay introduced or lost all original origins")
    valid_positions = [positions[key] for key in prefix_keys if key in positions]
    expected_prefix = full.take(pa.array(valid_positions, type=pa.int64()))
    field_checks = {name: name in truncated.column_names and same_bits(expected_prefix[name], truncated[name]) for name in full.column_names}
    all_bits = all(field_checks.values())
    if not all_bits:
        errors.append("one or more common native field bits differ")
    r, s, d, a = (preparation[name] for name in ("R", "S", "D", "A"))
    straddle = r < s <= d < a and preparation["carrier"]["symbol"] != preparation["symbol"] and preparation["carrier"]["receive"] == d
    if not straddle:
        errors.append("real receive straddle no longer satisfies R<S<=D<A")
    origin = tuple(preparation["origin_keys"][key] for key in KEYS)
    common_origin = origin in prefix_keys
    if not common_origin or origin[0] != s:
        errors.append("declared real origin is absent or changed")
    if any(time > d for time in truncated["SampleTime"].to_pylist()):
        errors.append("EOF generated an origin after the final retained receive")
    if common_origin:
        index = prefix_keys.index(origin)
        for variant in ("ordered", "reset"):
            exchange = truncated[f"H10_{variant}_processed_cluster_exchange_time"][index].as_py()
            available = truncated[f"H10_{variant}_processed_cluster_available_time"][index].as_py()
            if exchange >= preparation["pending_exchange"] or available > s:
                errors.append("open target cluster was closed or backdated into the earlier real origin")
    plan_path = OUTPUT / "full-preparation-receipt.yaml"
    plan = read(plan_path)
    inputs = [
        record(preparation_path),
        record(plan_path, preparation["full_plan_sha256"]),
        record(config, preparation["config"]["sha256"]),
        record(full_path, preparation["source_full_native"]["sha256"]),
        record(prefix_path),
        record(status_path),
        record(preparation["binary"]["path"], preparation["binary"]["sha256"]),
        record(preparation["calendar"]["path"], preparation["calendar"]["sha256"]),
        record(preparation["runtime_snapshot"]["path"], preparation["runtime_snapshot"]["sha256"]),
    ]
    for node in read(Path(preparation["runtime_snapshot"]["path"]))["dependencies"]:
        inputs.append(record(node["path"], node["sha256"]))
    for node in plan["basic_info_inputs"]:
        if Path(node["path"]).stem == preparation["day"]:
            inputs.append(record(node["path"], node["sha256"]))
    for node in preparation["cuts"]:
        inputs += [record(node["source"]["path"], node["source"]["sha256"]), record(node["path"], node["sha256"])]
    sources = [record(Path(__file__)), record(OUTPUT / "prepare_receive_prefix.py", preparation["script_sha256"])]
    sources.append(record(preparation["executed_script"]["path"], preparation["executed_script"]["sha256"]))
    snapshot_path = OUTPUT / "compiled-source-snapshot.yaml"
    snapshot = read(snapshot_path)
    build_path = OUTPUT / "producer-source-build.yaml"
    build = read(build_path)
    if snapshot.get("schema") != "compiled-source-snapshot-v1" or snapshot.get("compiled_binary_sha256") != preparation["binary"]["sha256"]:
        raise RuntimeError("Compiled source snapshot belongs to another producer")
    inputs += [record(snapshot_path), record(build_path, snapshot["original_build_receipt"]["sha256"]), record(PREFIX / "validation.yaml")]
    compiled_root = OUTPUT / "compiled-source"
    checkout_root = OUTPUT.parents[3]
    snapshot_bindings = {node["path"]: node["sha256"] for node in snapshot["sources"]}
    if len(snapshot_bindings) != len(build["sources"]):
        raise RuntimeError("Compiled snapshot lost an original producer source")
    for node in build["sources"]:
        archived = compiled_root / Path(node["path"]).relative_to(checkout_root)
        if snapshot_bindings.get(str(archived)) != node["sha256"]:
            raise RuntimeError("Compiled source snapshot disagrees with prebuild provenance")
        sources.append(record(archived, node["sha256"]))
    for node in preparation["source_files"]:
        archived = compiled_root / Path(node["path"]).relative_to(checkout_root)
        if snapshot_bindings.get(str(archived)) != node["sha256"]:
            raise RuntimeError("Archived native ABI/parser contract differs from the executed prefix cut")
    if any(Path(node["path"]).is_relative_to(checkout_root / "src") for node in sources):
        raise RuntimeError("Prefix source closure must use compiled immutable CPP copies")
    sources = list({node["path"]: node for node in sources}.values())
    inputs = list({node["path"]: node for node in inputs}.values())
    result = {
        "schema": "native-receive-prefix-validation-v1",
        "passed": not errors,
        "genuine_straddle_found": bool(straddle),
        "common_origin_present": common_origin,
        "all_common_origin_bits_exact": all_bits,
        "binary_sha256": preparation["binary"]["sha256"],
        "sources": sources,
        "inputs": inputs,
        "day": preparation["day"],
        "symbol": preparation["symbol"],
        "R": r,
        "S": s,
        "D": d,
        "A": a,
        "full_rows": len(full),
        "prefix_rows": len(truncated),
        "common_rows": len(valid_positions),
        "field_bit_checks": field_checks,
        "pending_exchange": preparation["pending_exchange"],
        "errors": errors,
        "labels_read": False,
        "model_fits": 0,
        "native_replays_launched": 0,
        "generated_events": 0,
        "scope": "Actual open-cluster origin remains bit-exact when unread future receive suffix is removed; immutable compiled producer has no EOF cluster flush",
        "compiled_source_snapshot_bound": True,
        "previous_live_source_receipt_preserved": str(PREFIX / "validation.yaml"),
    }
    destination.write_text(yaml.safe_dump(result, sort_keys=False))
    print("PREFIX_VALIDATED", result["passed"], "common rows", result["common_rows"], "all fields", len(field_checks), "errors", errors)


if __name__ == "__main__":
    main()
