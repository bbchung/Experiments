from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np

from ...io import ContractError, atomic_bytes, digest, file_hash, read_yaml, write_yaml

MAX_EXACT_INTEGER = 2**53
RULES = {
    "exchange_domain": "vendor_exchange_microseconds",
    "available_domain": "receive_microseconds",
    "origin_domain": "receive_microseconds",
    "available_at_or_before_origin": True,
    "exchange_at_or_before_available": False,
    "exchange_at_or_before_origin": False,
    "paired_positive_zero_unobserved": True,
    "observed_requires_positive_clocks": True,
    "maximum_integer_exclusive": MAX_EXACT_INTEGER,
    "diagnostic_float_dtype": "float64",
    "origin_integer_dtype": "int64",
    "timestamp_translation": False,
    "epsilon_or_rounding": False,
    "sampled_global_monotonicity_required": False,
}
UNCHANGED = {
    "feature_values": True,
    "labels": True,
    "sampling": True,
    "splits": True,
    "training": True,
    "fe_metrics": True,
    "fe_acceptance": True,
}


def validate_clocks(exchange, available, origin, *, observed=None):
    """Validate native stored clocks without comparing independent domains.

    Zero pairs denote no closed observation. The optional Boolean mask states
    which rows require a closed observation, rather than inferring it from a
    numeric sentinel or treating every sampled snapshot as a new observation.
    No array is rounded, clipped, shifted, sorted, or modified.
    """
    exchange, available, origin = map(np.asarray, (exchange, available, origin))
    if origin.dtype != np.int64 or origin.ndim != 1 or not len(origin):
        raise ContractError("Origin must be a nonempty native int64 vector")
    if np.any((origin <= 0) | (origin >= MAX_EXACT_INTEGER)):
        raise ContractError("Origin exceeds the positive exact integer domain")
    for name, values in (("exchange", exchange), ("available", available)):
        if values.dtype != np.float64 or values.shape != origin.shape:
            raise ContractError(f"{name} must retain native float64 shape")
        if np.any(~np.isfinite(values) | (values < 0) | (values >= MAX_EXACT_INTEGER) | (values != np.floor(values))):
            raise ContractError(f"{name} is not an exact nonnegative integer clock")
        if np.any((values == 0) & np.signbit(values)):
            raise ContractError(f"{name} has a negative-zero unobserved state")
    exchange_int = exchange.astype(np.int64)
    available_int = available.astype(np.int64)
    if np.any((exchange_int == 0) != (available_int == 0)):
        raise ContractError("Exchange and availability zero states disagree")
    if observed is not None:
        observed = np.asarray(observed)
        if observed.dtype != np.bool_ or observed.shape != origin.shape:
            raise ContractError("Observed mask must be an exact Boolean vector")
        if np.any(observed & (available_int == 0)):
            raise ContractError("Observed state requires positive clocks")
    if np.any(available_int > origin):
        raise ContractError("Receive availability exceeds the sample origin")
    return {
        "rows": len(origin),
        "observed_rows": int(np.count_nonzero(available_int)),
        "exchange_after_available_rows": int(np.count_nonzero(exchange_int > available_int)),
        "exchange_after_origin_rows": int(np.count_nonzero(exchange_int > origin)),
        "available_after_origin_rows": 0,
    }


def load_profile(path):
    path = Path(path).resolve()
    profile = read_yaml(path)
    if not isinstance(profile, dict) or profile.get("schema") != "native-temporal-method-profile-v1":
        raise ContractError("Invalid temporal methodology profile")
    if profile.get("rules") != RULES or profile.get("unchanged") != UNCHANGED:
        raise ContractError("Temporal method cannot alter the FE judge or clock semantics")
    floor = str(profile.get("data_floor"))
    if len(floor) != 8 or not floor.isdigit() or floor < "20260101" or any(type(profile.get(key)) is not int or profile[key] != 0 for key in ("training_fits", "native_replays")):
        raise ContractError("Temporal study scope violates the fixed data/compute budget")
    scope = profile.get("census_scope", {})
    if set(scope) != {"native_cells", "rows_per_variant"} or any(type(value) is not int or value <= 0 for value in scope.values()):
        raise ContractError("Temporal census population must be declared before validation")
    proof = profile.get("source_proof")
    if (
        not isinstance(proof, list)
        or not proof
        or any(not isinstance(value, str) or not value or not Path(value).is_absolute() for value in proof)
        or len(proof) != len(set(proof))
    ):
        raise ContractError("Temporal method requires explicit nonempty source proof paths")
    evidence = profile.get("evidence")
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"census", "raw_witness", "native_offset_tests"}
        or not all(isinstance(value, str) and value for value in evidence.values())
    ):
        raise ContractError("Temporal method requires independently recorded evidence")
    return profile


def _node(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def _bound(node):
    if not isinstance(node, dict) or set(node) != {"path", "sha256"} or not isinstance(node["path"], str):
        raise ContractError("Invalid temporal evidence file binding")
    actual = _node(node["path"])
    if actual != node:
        raise ContractError(f"Temporal evidence file changed: {node['path']}")
    return actual


def _closure(report, sources, inputs):
    for scope, target in (("sources", sources), ("inputs", inputs)):
        nodes = report.get(scope)
        if not isinstance(nodes, list) or not nodes:
            raise ContractError(f"Evidence requires explicit {scope} closure")
        for node in nodes:
            _bound(node)
            target.append(node)


def study_evidence(profile):
    """Check source-bound real census/witness and independent native invariance.

    Full-file producer hashes in the census are historical journal bindings.
    Current diagnostic projection hashes are measured by its bound census
    source; this validator does not claim a second full native-file hash pass.
    """
    sources, inputs = [], []
    documents = {}
    for name, path in profile["evidence"].items():
        inputs.append(_node(path))
        documents[name] = read_yaml(Path(path))
    census, witness, offset = (documents[name] for name in ("census", "raw_witness", "native_offset_tests"))
    if census.get("schema") != "h10-native-diagnostic-clock-census-v1" or census.get("labels_read") is not False or census.get("model_outputs_read") is not False:
        raise ContractError("Temporal census must be independent of labels and model scores")
    if census.get("native_replays_launched") != 0 or not isinstance(census.get("native_cells"), int) or census["native_cells"] <= 0:
        raise ContractError("Invalid bounded temporal census scope")
    if census["native_cells"] != profile["census_scope"]["native_cells"]:
        raise ContractError("Temporal census differs from its declared population")
    cells = census.get("cells")
    if not isinstance(cells, list) or len(cells) != census["native_cells"]:
        raise ContractError("Temporal census omitted declared native cells")
    identities = set()
    for cell in cells:
        identity = str(cell["day"]), str(cell["symbol"])
        if len(identity[0]) != 8 or not identity[0].isdigit() or identity[0] < str(profile["data_floor"]) or not identity[1] or identity in identities:
            raise ContractError("Temporal census has an invalid or duplicated native cell")
        identities.add(identity)
        if not isinstance(cell.get("rows"), int) or cell["rows"] <= 0:
            raise ContractError("Temporal census contains an empty native cell")
        for key in ("producer_recorded_sha256", "projection_sha256"):
            value = cell.get(key, "")
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ContractError("Temporal census lacks exact measured/journal identities")
    for variant in ("ordered", "reset"):
        total = census["totals"][variant]
        if total.get("rows") != sum(cell["rows"] for cell in cells) or total["rows"] != profile["census_scope"]["rows_per_variant"]:
            raise ContractError("Temporal census total row count disagrees")
        for key in ("available_after_origin_rows", "available_after_origin_cells", "invalid_integer_clock_cells", "negative_zero_clock_cells", "zero_pair_mismatch_cells"):
            if total.get(key) != 0:
                raise ContractError(f"Actual temporal census violates {key}")
        for key in ("exchange_after_available_rows", "exchange_after_origin_rows", "available_after_origin_rows"):
            counts = [cell[variant].get(key) for cell in cells]
            if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > cell["rows"] for cell, count in zip(cells, counts, strict=True)):
                raise ContractError("Invalid per-cell temporal event counts")
            if total.get(key) != sum(counts):
                raise ContractError("Temporal census event count disagrees")
        for cell in cells:
            stats = cell[variant]
            if any(stats.get(key) is not True for key in ("exact_nonnegative_integer_clocks", "exact_positive_zero_clocks", "paired_zero_state")):
                raise ContractError("Actual native clock storage or zero state is invalid")
    if any(census["totals"][variant]["exchange_after_available_rows"] <= 0 for variant in ("ordered", "reset")):
        raise ContractError("No actual cross-domain assumption counterexample")
    if witness.get("generated_rows") != 0 or witness.get("timestamp_changes") != 0 or witness.get("first_native_failure") != census.get("first_failure"):
        raise ContractError("Raw witness is not the unchanged actual census counterexample")
    if _bound(census.get("raw_witness")) != _node(profile["evidence"]["raw_witness"]):
        raise ContractError("Raw witness does not match its census binding")
    raw_source = witness.get("raw_source", {})
    inputs.append(_bound({key: raw_source[key] for key in ("path", "sha256")}))
    sources.append(_bound(witness.get("decoder_source")))
    failure = witness["first_native_failure"]
    matches = [cell for cell in cells if (str(cell["day"]), str(cell["symbol"])) == (failure.get("day"), failure.get("symbol"))]
    if len(matches) != 1 or type(failure.get("row_index")) is not int or not 0 <= failure["row_index"] < matches[0]["rows"]:
        raise ContractError("Raw witness counterexample is outside the declared native population")
    if any(failure.get(key) != matches[0].get(key) for key in ("path", "producer_recorded_sha256")):
        raise ContractError("Raw witness counterexample differs from its census cell identity")
    values = failure["values"]
    if type(values.get("SampleTime")) is not int:
        raise ContractError("Raw witness origin must retain its native integer identity")
    exchange = int(values["H10_ordered_processed_cluster_exchange_time"])
    available = int(values["H10_ordered_processed_cluster_available_time"])
    origin = int(values["SampleTime"])
    validate_clocks(
        np.array([values["H10_ordered_processed_cluster_exchange_time"]], dtype=np.float64),
        np.array([values["H10_ordered_processed_cluster_available_time"]], dtype=np.float64),
        np.array([origin], dtype=np.int64),
        observed=np.array([True]),
    )
    records = witness.get("raw_records", [])
    pair = None
    for first, second in pairwise(records):
        if (
            first.get("type") == second.get("type") == "B"
            and first.get("status") == second.get("status") == 0
            and first.get("exchange") == exchange
            and second.get("exchange", 0) > exchange
            and second.get("receive") == available
        ):
            pair = first, second
            break
    if pair is None or exchange <= available or pair[0]["receive"] > available or not (pair[0]["ordinal"] < pair[1]["ordinal"]):
        raise ContractError("Raw BOOK witness does not demonstrate a received closing pair")
    if failure["day"] != raw_source.get("day") or failure["symbol"] != raw_source.get("symbol"):
        raise ContractError("Raw witness is from a different native cell")
    if offset.get("schema") != "native-exchange-clock-offset-tests-v1" or any(
        offset.get(key) is not True
        for key in ("passed", "ordered_numeric_bits_exact", "ordered_phase_exact", "reset_numeric_bits_exact", "reset_phase_exact", "native_producer_binary_unchanged")
    ):
        raise ContractError("Native exchange-offset invariance has not passed")
    if offset.get("positive_offset_us") != 3_600_000_000 or offset.get("negative_offset_us") != -3_600_000_000 or offset.get("feature_or_label_translation_performed") is not False:
        raise ContractError("Native offset experiment differs from its frozen scope")
    _closure(census, sources, inputs)
    _closure(offset, sources, inputs)
    binary_node = _bound({key: census["binary"][key] for key in ("path", "sha256")})
    if binary_node not in offset["inputs"]:
        raise ContractError("Native offset experiment is not bound to the census producer binary")
    inputs.append(binary_node)
    return {
        "schema": "native-temporal-method-validation-v1",
        "passed": True,
        "native_cells": census["native_cells"],
        "totals": census["totals"],
        "raw_received_counterexample_confirmed": True,
        "native_exchange_offset_invariance_confirmed": True,
        "future_receive_availability_permitted": False,
        "fe_evaluation_judge_changed": False,
        "sources": sources,
        "inputs": inputs,
    }


def freeze(profile, path, sources=(), inputs=()):
    """Freeze validated independent methodology before an FE guard changes."""
    profile_path, path = Path(profile).resolve(), Path(path).resolve()
    anchor = Path(str(path) + ".identity")
    if path.exists() or anchor.exists():
        raise ContractError("Temporal methodology freeze is immutable")
    loaded = load_profile(profile_path)
    evidence = study_evidence(loaded)
    source_nodes, input_nodes = _mandatory_nodes(profile_path, loaded, evidence)
    source_nodes += [_node(value) for value in sources]
    input_nodes += [_node(value) for value in inputs]
    manifest = {
        "schema": "native-temporal-method-frozen-v1",
        "profile_path": str(profile_path),
        "profile": loaded,
        "validation": {key: value for key, value in evidence.items() if key not in ("sources", "inputs")},
        "sources": _unique_nodes(source_nodes),
        "inputs": _unique_nodes(input_nodes),
    }
    manifest["identity"] = digest(manifest)
    write_yaml(path, manifest)
    atomic_bytes(anchor, (manifest["identity"] + "\n").encode())
    return verify(path)


def _mandatory_nodes(profile_path, profile, evidence):
    sources = list(evidence["sources"])
    sources.extend(_node(file) for file in Path(__file__).parent.glob("*.py"))
    sources.append(_node(Path(__file__).parents[2] / "io.py"))
    sources.extend(_node(file) for file in profile["source_proof"])
    return _unique_nodes(sources), _unique_nodes([*evidence["inputs"], _node(profile_path)])


def _unique_nodes(nodes):
    result = {}
    for node in nodes:
        previous = result.setdefault(node["path"], node)
        if previous != node:
            raise ContractError("Inconsistent temporal evidence identity")
    return sorted(result.values(), key=lambda node: node["path"])


def verify(path):
    path = Path(path).resolve()
    manifest = read_yaml(path)
    if manifest.get("schema") != "native-temporal-method-frozen-v1":
        raise ContractError("Unknown temporal frozen methodology")
    identity = manifest.get("identity")
    if digest({key: value for key, value in manifest.items() if key != "identity"}) != identity or Path(str(path) + ".identity").read_text().strip() != identity:
        raise ContractError("Temporal methodology identity drift")
    profile = load_profile(manifest["profile_path"])
    if profile != manifest["profile"]:
        raise ContractError("Temporal methodology profile or validation drift")
    evidence = study_evidence(profile)
    validation = {key: value for key, value in evidence.items() if key not in ("sources", "inputs")}
    if manifest.get("validation") != validation:
        raise ContractError("Temporal methodology validation differs from current mandatory evidence")
    for node in (*manifest["sources"], *manifest["inputs"]):
        _bound(node)
    required_sources, required_inputs = _mandatory_nodes(Path(manifest["profile_path"]), profile, evidence)
    for scope, nodes in (("sources", required_sources), ("inputs", required_inputs)):
        for node in nodes:
            if node not in manifest[scope]:
                raise ContractError(f"Mandatory temporal {scope} member is not frozen: {node['path']}")
    return manifest
