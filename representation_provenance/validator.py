from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ...io import digest, file_hash, read_yaml

NATIVE_KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")
HORIZONS = ("60s", "120s", "180s", "300s")
LABEL_COLUMNS = tuple(f"{family}[{horizon}]" for family in ("mid_return_ticks", "sticky_return_ticks", "mid_endpoint.up.5", "mid_endpoint.down.5") for horizon in HORIZONS)
FLOW_ALLOWED = tuple(f"FlowResponseSurprise.0.{family}{index}.0" for family in ("flow_surprise", "response_innovation", "response_coupling") for index in range(3))
CROSS_ALLOWED = tuple(f"CrossReturnContext.0.{family}{index}.0" for family in ("target_return_z", "residual_return_z") for index in range(6)) + (
    "CrossReturnContext.0.target_sigma_bps.0",
    "CrossReturnContext.0.peer_sigma_ratio.0",
)
H10_NUMERIC_FIELDS = (
    "direction",
    "log_work_strength",
    "signed_after_containment_share",
    "signed_after_repair_share",
    "delayed_repair_fraction",
    "signed_renewed_work",
    "mid_progress_5ticks",
)
CONTROL_CELLS = {("20260203", "2609"), ("20260203", "2337")}
PILOT_CELLS = {(day, symbol) for day in ("20260119", "20260120") for symbol in ("2308", "2317")}


def _column(table, name):
    return table.column(name).combine_chunks()


def _values(column):
    return column.to_numpy(zero_copy_only=False)


def _equal_bits(left, right):
    """Preserve float bits and Arrow validity; never collapse null into NaN."""
    if left.type != right.type or len(left) != len(right):
        return np.zeros(max(len(left), len(right)), dtype=bool)
    valid_left = _values(left.is_valid())
    valid_right = _values(right.is_valid())
    if pa.types.is_floating(left.type):
        dtype = np.uint64 if pa.types.is_float64(left.type) else np.uint32 if pa.types.is_float32(left.type) else np.uint16
        same = _values(left).view(dtype) == _values(right).view(dtype)
    elif pa.types.is_integer(left.type) or pa.types.is_boolean(left.type):
        # Nullable integer arrays otherwise convert to float64, losing exact
        # identities above 2**53 even though Arrow retains their integer bits.
        fill = False if pa.types.is_boolean(left.type) else 0
        same = _values(left.fill_null(fill)) == _values(right.fill_null(fill))
    else:
        same = np.array([a == b for a, b in zip(left.to_pylist(), right.to_pylist(), strict=True)], dtype=bool)
    return (~valid_left & ~valid_right) | (valid_left & valid_right & same)


def _key_errors(table, day, symbol):
    errors = []
    if len(day) != 8 or not day.isdigit() or day < "20260101" or not symbol:
        errors.append("invalid declared day/symbol identity")
    for name in NATIVE_KEYS:
        if name not in table.column_names:
            errors.append(f"missing native key: {name}")
        elif table.schema.field(name).type != pa.int64() or table.column(name).null_count:
            errors.append(f"native key must retain nonnull int64: {name}")
    for name, declared in (("day", day), ("symbol", symbol)):
        if name in table.column_names and any(value != declared for value in table.column(name).to_pylist()):
            errors.append(f"declared {name} disagrees with row identity")
    if not errors:
        keys = list(zip(*(table.column(name).to_pylist() for name in NATIVE_KEYS), strict=True))
        if len(keys) != len(set(keys)):
            errors.append("duplicate complete native sample keys")
    if table.num_rows == 0:
        errors.append("empty native partition")
    return errors


def _state(value):
    if value is None:
        return "null"
    if np.isnan(value):
        return "nan"
    if np.isneginf(value):
        return "negative_infinity"
    if np.isposinf(value):
        return "positive_infinity"
    return "finite"


def _difference(left, right, changed):
    indices = np.flatnonzero(changed)
    pairs = Counter(f"{_state(left[index].as_py())}->{_state(right[index].as_py())}" for index in indices)
    finite = sum(_state(left[index].as_py()) == _state(right[index].as_py()) == "finite" for index in indices)
    return {"rows_changed": len(indices), "finite_rows_changed": finite, "state_transitions": dict(sorted(pairs.items()))}


def compare_historical_components(original, current, day, symbol):
    """Account for exact documented differences without changing either input."""
    errors = _key_errors(original, day, symbol) + _key_errors(current, day, symbol)
    required = (*NATIVE_KEYS, "OriginMidPrice", *LABEL_COLUMNS, *FLOW_ALLOWED, *CROSS_ALLOWED)
    for name in required:
        if name not in original.column_names or name not in current.column_names:
            errors.append(f"missing historical component: {name}")
    if not original.schema.equals(current.schema, check_metadata=True):
        errors.append("historical/current native schemas differ")
    if original.num_rows != current.num_rows:
        errors.append("historical/current row counts differ")
    if "OriginMidPrice" in original.column_names and original.schema.field("OriginMidPrice").type != pa.float64():
        errors.append("OriginMidPrice must retain native float64")
    differences = {}
    if not errors:
        for name in original.column_names:
            left, right = _column(original, name), _column(current, name)
            changed = ~_equal_bits(left, right)
            if not changed.any():
                continue
            if name in FLOW_ALLOWED or name in CROSS_ALLOWED:
                if left.type != pa.float64():
                    errors.append(f"documented component must retain float64: {name}")
                    continue
                differences[name] = _difference(left, right, changed)
                if name in FLOW_ALLOWED:
                    allowed = _values(left.is_valid()) & _values(right.is_valid()) & np.isnan(_values(left)) & np.isneginf(_values(right))
                    if np.any(changed & ~allowed):
                        errors.append(f"FlowResponse difference is not historical NaN->warmup: {name}")
            else:
                errors.append(f"unaccounted bit/state difference: {name}")
    return {"day": day, "symbol": symbol, "passed": not errors, "rows": original.num_rows, "differences": differences, "errors": errors}


def compare_correction_receipt(result, receipt):
    """Require the measured correction footprint to match its prior evidence."""
    errors = []
    if str(receipt["day"]) != result["day"]:
        errors.append("historical correction receipt day differs from control")
    reference = receipt["symbols"][result["symbol"]]["reference_ten_second"]
    expected = reference["feature_bitwise_mismatches"]
    actual = result["differences"]
    if result["rows"] != reference["common_origins"] or reference["native_label_bitwise_mismatches"]:
        errors.append("historical correction receipt origin/label claim disagrees")
    if set(actual) != set(expected):
        errors.append("changed component names differ from historical correction receipt")
    for name in actual.keys() & expected.keys():
        observed = actual[name]
        availability = sum(count for transition, count in observed["state_transitions"].items() if transition != "finite->finite")
        if any(observed[key] != expected[name][key] for key in ("rows_changed", "finite_rows_changed")) or availability != expected[name]["availability_changed"]:
            errors.append(f"changed component counts differ from historical correction receipt: {name}")
    return {"passed": not errors, "errors": errors}


def compare_candidate_join(parent, candidate, day, symbol):
    """Require the exact original origin, physical mid and receive availability."""
    errors = _key_errors(parent, day, symbol) + _key_errors(candidate, day, symbol)
    if parent.num_rows != candidate.num_rows:
        errors.append("candidate must preserve every original origin in order")
    for name in (*NATIVE_KEYS, "OriginMidPrice"):
        if name not in parent.column_names or name not in candidate.column_names:
            errors.append(f"missing exact join component: {name}")
            continue
        if name == "OriginMidPrice" and (parent.schema.field(name).type != pa.float64() or candidate.schema.field(name).type != pa.float64()):
            errors.append("candidate OriginMidPrice must retain native float64")
        if not _equal_bits(_column(parent, name), _column(candidate, name)).all():
            errors.append(f"candidate exact key/mid mismatch: {name}")
    if any(name in candidate.column_names for name in LABEL_COLUMNS):
        errors.append("feature-only candidate unexpectedly contains regenerated native labels")
    if "SampleTime" in candidate.column_names and candidate.schema.field("SampleTime").type == pa.int64():
        origin = _values(_column(candidate, "SampleTime"))
        for instance, label in ((0, "ordered"), (1, "reset")):
            phase_name = f"DemandRepairRenewal.{instance}.phase.0"
            names = [f"DemandRepairRenewal.{instance}.{field}.0" for field in H10_NUMERIC_FIELDS]
            clocks = [f"H10_{label}_processed_cluster_{axis}_time" for axis in ("exchange", "available")]
            missing = [name for name in (phase_name, *names, *clocks) if name not in candidate.column_names]
            if missing:
                errors.append(f"missing H10 state/availability fields: {missing}")
                continue
            numeric = []
            for name in (*names, *clocks):
                if candidate.schema.field(name).type != pa.float64() or candidate.column(name).null_count:
                    errors.append(f"H10 numeric field must retain nonnull float64: {name}")
                numeric.append(_values(_column(candidate, name)))
            if any(candidate.schema.field(name).type != pa.float64() or candidate.column(name).null_count for name in (*names, *clocks)):
                continue
            exchange, available = numeric[-2:]
            if np.any(~np.isfinite(available) | (available < 0) | (available != np.floor(available)) | (available > origin)):
                errors.append(f"H10 {label} availability is not an observed integer time <= origin")
            if np.any(~np.isfinite(exchange) | (exchange < 0) | (exchange != np.floor(exchange)) | (exchange > available) | ((exchange == 0) != (available == 0))):
                errors.append(f"H10 {label} exchange/availability clock states disagree")
            values = np.column_stack(numeric[: len(names)])
            for index, phase in enumerate(candidate.column(phase_name).to_pylist()):
                row = values[index]
                if phase == "warmup":
                    valid = np.isneginf(row).all()
                elif phase == "inactive":
                    valid = (row.view(np.uint64) == 0).all()
                elif phase in ("censored", "stale"):
                    valid = np.isnan(row).all()
                elif phase in ("contained", "repaired", "renewed", "resumed"):
                    valid = np.isfinite(row).all() and row[0] in (-1.0, 1.0) and row[1] > 0.0
                else:
                    valid = False
                if not valid:
                    errors.append(f"H10 {label} phase/numeric state disagreement at row {index}: {phase}")
                    break
    return {"day": day, "symbol": symbol, "passed": not errors, "rows": parent.num_rows, "errors": errors}


def _digest(path):
    return file_hash(Path(path))


def validate_manifest(manifest_path):
    """Validate frozen file bindings first, then bounded native components."""
    manifest_path = Path(manifest_path).resolve()
    report = {
        "schema": "native-component-lineage-validation-v2",
        "passed": False,
        "baseline_parent_reproduced": False,
        "current_control_invariant": False,
        "historical_components_accounted": False,
        "candidate_exact_original_join": False,
        "historical_native_gate_v1_preserved_failed": False,
        "fe_evaluation_judge_changed": False,
        "feature_or_label_translation_performed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": None,
        "historical_runtime_identity_claimed": False,
        "sources": [],
        "inputs": [],
        "errors": [],
    }
    cache = {}

    def bound(node, destination):
        if not isinstance(node, dict) or set(node) != {"path", "sha256"}:
            raise ValueError("bound files require exactly path and sha256")
        path = Path(node["path"])
        path = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
        expected = node["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"invalid SHA256 binding: {path}")
        actual = _digest(path)
        if actual != expected:
            raise ValueError(f"bound file SHA256 mismatch: {path}")
        cache[path] = actual
        destination.append({"path": str(path), "sha256": actual})
        return path

    try:
        report["manifest_sha256"] = _digest(manifest_path)
        manifest = read_yaml(manifest_path)
        if not isinstance(manifest, dict):
            raise TypeError("lineage manifest must be a mapping")
        method = manifest["method_contract"]
        contract_path = bound({"path": method["path"], "sha256": method["sha256"]}, report["inputs"])
        identity_path = bound({"path": method["identity_path"], "sha256": method["identity_sha256"]}, report["inputs"])
        identity = identity_path.read_text().strip()
        if identity != method["identity"] or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("method contract identity anchor disagrees")
        contract = read_yaml(contract_path)
        if not isinstance(contract, dict) or contract.get("identity") != identity or digest({key: value for key, value in contract.items() if key != "identity"}) != identity:
            raise ValueError("method contract content disagrees with identity anchor")
        if contract["manifest_payload_identity"] != digest({key: value for key, value in manifest.items() if key != "method_contract"}):
            raise ValueError("producer roles and artifact bindings differ from the frozen method manifest payload")
        report["manifest_payload_identity"] = contract["manifest_payload_identity"]
        if identity_path != contract_path.with_name(contract_path.name + ".identity"):
            raise ValueError("method identity anchor must accompany its frozen contract")
        report["method_contract_path"] = str(contract_path)
        report["method_contract_identity"] = identity
        for scope in ("sources", "inputs"):
            if not contract[scope]:
                raise ValueError(f"method lacks frozen {scope} closure")
            for node in contract[scope]:
                bound(node, report[scope])
        running_source = _digest(Path(__file__).resolve())
        if not any(Path(node["path"]).name == "validator.py" and node["sha256"] == running_source for node in contract["sources"]):
            raise ValueError("running validator differs from its independently frozen source closure")
        report["validator_source_sha256"] = running_source
        bound(manifest["reproduction_plan"], report["sources"])
        correction_path = bound(manifest["historical_correction_receipt"], report["sources"])
        correction = read_yaml(correction_path)
        old_gate_path = bound(manifest["gate_v1_failure"], report["sources"])
        old_gate = read_yaml(old_gate_path)
        if not isinstance(old_gate, dict) or old_gate.get("passed") is not False:
            raise ValueError("historical NativeGateV1 failure must remain explicit and unchanged")
        report["historical_native_gate_v1_preserved_failed"] = True
        for name in ("original", "current", "h10"):
            producer = manifest["producers"][name]
            binary = bound(producer["binary"], report["sources"])
            report[f"{name}_binary_sha256"] = cache[binary]
            if not producer["runtime_dependencies"] or not producer["source_receipts"]:
                raise ValueError(f"producer lacks runtime/source provenance: {name}")
            for item in (*producer["runtime_dependencies"], *producer["source_receipts"]):
                bound(item, report["sources"])
        controls = manifest["controls"]
        if len(controls) != len(CONTROL_CELLS) or {(str(cell["day"]), str(cell["symbol"])) for cell in controls} != CONTROL_CELLS:
            raise ValueError("method must retain both fixed historical control cells")
        report["controls"] = []
        for cell in controls:
            paths = {key: bound(cell[key], report["inputs"]) for key in ("parent_original", "original_reproduced", "current_preserved", "h10_control")}
            if not cell["input_receipts"]:
                raise ValueError("control lacks config/raw dependency provenance")
            for node in cell["input_receipts"]:
                bound(node, report["inputs"])
            result = compare_historical_components(pq.read_table(paths["parent_original"]), pq.read_table(paths["current_preserved"]), str(cell["day"]), str(cell["symbol"]))
            result["historical_correction_receipt"] = compare_correction_receipt(result, correction)
            result["errors"].extend(result["historical_correction_receipt"]["errors"])
            result["passed"] = not result["errors"]
            result["baseline_parent_reproduced"] = cache[paths["parent_original"]] == cache[paths["original_reproduced"]]
            result["current_control_invariant"] = cache[paths["current_preserved"]] == cache[paths["h10_control"]]
            report["controls"].append(result)
        report["baseline_parent_reproduced"] = all(cell["baseline_parent_reproduced"] for cell in report["controls"])
        report["current_control_invariant"] = all(cell["current_control_invariant"] for cell in report["controls"])
        report["historical_components_accounted"] = all(cell["passed"] for cell in report["controls"])
        joins = manifest["candidate_joins"]
        if len(joins) != len(PILOT_CELLS) or {(str(cell["day"]), str(cell["symbol"])) for cell in joins} != PILOT_CELLS:
            raise ValueError("method must retain all four fixed native pilot join cells")
        report["candidate_joins"] = []
        for cell in joins:
            parent = bound(cell["parent_original"], report["inputs"])
            features = bound(cell["h10_features"], report["inputs"])
            if not cell["input_receipts"]:
                raise ValueError("candidate join lacks native config/raw provenance")
            for node in cell["input_receipts"]:
                bound(node, report["inputs"])
            parent_columns = [*NATIVE_KEYS, "OriginMidPrice"]
            parent_schema = pq.read_schema(parent)
            parent_columns.extend(name for name in ("day", "symbol") if name in parent_schema.names)
            report["candidate_joins"].append(compare_candidate_join(pq.read_table(parent, columns=parent_columns), pq.read_table(features), str(cell["day"]), str(cell["symbol"])))
        report["candidate_exact_original_join"] = all(cell["passed"] for cell in report["candidate_joins"])
        for path, expected in cache.items():
            if _digest(path) != expected:
                raise ValueError(f"bound file changed during component validation: {path}")
        report["passed"] = all(
            report[key]
            for key in (
                "baseline_parent_reproduced",
                "current_control_invariant",
                "historical_components_accounted",
                "candidate_exact_original_join",
                "historical_native_gate_v1_preserved_failed",
            )
        )
        if not report["passed"]:
            report["errors"].append("one or more independently required component integrity checks failed")
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, pa.ArrowException) as error:
        report["errors"].append(str(error))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = validate_manifest(args.manifest)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite immutable validation receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(report, sort_keys=False))
    print(json.dumps({"passed": report["passed"], "output": str(output), "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
