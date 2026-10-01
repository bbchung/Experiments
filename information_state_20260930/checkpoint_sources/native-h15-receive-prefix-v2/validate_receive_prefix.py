"""Versioned exact H15 prefix comparison with explicit empty-event absence.

Root binds original KEY-only evidence, freezes this comparison, then checks native
values. The already completed V1 replay and its immutable receipts are reused.
No cut, replay, feature computation, empty artifact, label or model is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT = Path(__file__).resolve().parent
V1 = OUTPUT.parent / "native-h15-receive-prefix"
PLAN = OUTPUT / "PREFIX_V2_PLAN.md"
TEST = OUTPUT / "test_prefix.py"
spec = importlib.util.spec_from_file_location("h15_prefix_v1_immutable", V1 / "prepare_receive_prefix.py")
prefix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix)
producer = prefix.producer
require, record, check_record = prefix.require, prefix.record, prefix.check_record
canonical_method, unique_records = prefix.canonical_method, prefix.unique_records
digest, read_yaml, write_yaml = prefix.digest, prefix.read_yaml, prefix.write_yaml
KEYS = prefix.KEYS
METHOD_SCHEMA = "h15-real-receive-prefix-comparison-method-v2"
BOUND_SCHEMA = "h15-real-receive-prefix-bound-comparison-v2"
VALIDATION_SCHEMA = "h15-real-receive-prefix-validation-v2"
CONTRACT = {
    "schema": "h15-real-receive-prefix-comparison-contract-v2",
    "selection": "unchanged frozen V1 first genuine pending-E receive straddle and exact completed native job",
    "absence": "only event partition with exactly zero original KEYs SampleTime<=D; periodic absence always rejects",
    "freeze": "original KEY-only counts/order/digests, schema and existing execution/artifact presence before reading any native value",
    "comparison": "exact complete original keys throughD, all11Alpha/27Info/rawmid values and categories; selected original S mandatory",
    "empty_artifact": "never synthesize an empty table or file; absence remains absence and is frozen explicitly",
    "native_replays_launched": 0,
    "generated_events": 0,
    "labels_read": False,
    "model_fits": 0,
}


def native_schema(schema):
    categories = {f"PeerTradeInformation.0.{name}.0" for name in producer.ALPHA_CATEGORICAL}
    names = (
        set(KEYS)
        | {f"PeerTradeInformation.0.{name}.0" for name in producer.ALPHA}
        | {f"H15_{name}" for name in producer.INFO}
        | {"OriginMidPrice", "OriginMidTicks", "OriginBidTicks", "OriginAskTicks"}
    )
    require(len(schema.names) == len(names) and set(schema.names) == names, "undeclared/missing native fields or labels")
    for name in names:
        expected = pa.int64() if name in KEYS else pa.string() if name in categories else pa.float64()
        require(schema.field(name).type == expected, "native key/float64/category schema changed")
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def parent_context():
    """Validate source/metadata closure only; never read raw/native table values."""
    receipt_path = V1 / "preparation-receipt.yaml"
    receipt = read_yaml(receipt_path)
    selection_path = V1 / "frozen-selection-method.yaml"
    comparison_path = V1 / "frozen-comparison-method.yaml"
    selection, preparation = prefix.verify_selection(selection_path)
    comparison = canonical_method(comparison_path, "h15-real-receive-prefix-comparison-method-v1")
    old_bound_path = V1 / "bound-comparison.yaml"
    old_bound = read_yaml(old_bound_path)
    require(
        receipt.get("schema") == "h15-real-receive-prefix-preparation-v1"
        and receipt.get("genuine_straddle_found") is True
        and receipt["selection_method"] == record(selection_path)
        and old_bound["selection_identity"] == receipt["selection_identity"] == selection["identity"]
        and old_bound["preparation"] == record(receipt_path)
        and old_bound["preparation_identity"] == digest(receipt)
        and comparison["plan"] == record(old_bound_path),
        "immutable V1 comparison/preparation/selection identity changed",
    )
    selected = receipt["selected"]
    require(all(prefix.positive_clock(selected[k]) for k in ("R", "S", "D", "A", "pending_exchange")), "unsafe/fractional straddle clock")
    require(selected["R"] < selected["S"] <= selected["D"] < selected["A"], "genuine selected receive straddle changed")
    require(receipt["producer_identity"] == prefix.PRODUCER_IDENTITY and receipt["support_identity"] == prefix.SUPPORT_IDENTITY, "different H15 source study")
    require(receipt["binary"] == preparation["binary"] and receipt["binary"]["sha256"] == producer.BINARY_SHA256, "different native binary")
    require([c["symbol"] for c in receipt["cuts"]] == list(producer.CARRIERS), "exact16 source cuts required")
    require(all(c["last_effective_receive"] <= selected["D"] for c in receipt["cuts"]), "future receive in source cut")
    require(
        read_yaml(Path(receipt["config"]["path"])) == prefix.prefix_config(read_yaml(Path(receipt["original_config"]["path"])), V1 / "raw"),
        "prefix config changed beyond original raw directory",
    )
    original_job = next(j for j in preparation["jobs"] if j["day"] == prefix.DAY)
    require(receipt["full_outputs"] == [record(Path(path)) for path in original_job["outputs"]], "different original pair output population")
    own_sources = [record(Path(__file__)), record(PLAN), record(TEST)]
    inherited_inputs = [*receipt["inputs"], record(receipt_path), record(old_bound_path)]
    producer.require_closure(comparison, "sources", selection["sources"])
    producer.require_closure(comparison, "inputs", inherited_inputs)
    sources = unique_records([*own_sources, *selection["sources"], *comparison["sources"]])
    inputs = unique_records(
        [
            *selection["inputs"],
            *comparison["inputs"],
            *inherited_inputs,
            record(selection_path),
            record(selection_path.with_suffix(".yaml.identity")),
            record(comparison_path),
            record(comparison_path.with_suffix(".yaml.identity")),
        ]
    )
    return receipt, comparison, sources, inputs


def execution_evidence(receipt, old_comparison_identity):
    """Receipt must describe exactly the files the completed V1 job produced."""
    execution_path = V1 / "truncated/execution-receipt.yaml"
    status_path = V1 / "truncated/status" / f"{prefix.DAY}.yaml"
    execution, status = read_yaml(execution_path), read_yaml(status_path)
    require(status.get("status") == "completed" and status.get("fatal_error") is False, "native prefix job did not complete")
    require(
        execution.get("method_identity") == old_comparison_identity
        and execution.get("job") == "receive_prefix"
        and type(execution.get("exit_code")) is int
        and execution["exit_code"] == 0
        and execution.get("command") == receipt["command"],
        "existing execution is not the exact frozen V1 comparison command",
    )
    expected_paths = [str(V1 / "truncated" / folder / prefix.DAY / target / "values.parquet") for folder in ("data", "events") for target in producer.TARGETS]
    require(receipt["prefix_outputs"] == expected_paths, "configured target/partition paths changed")
    present_paths = [path for path in expected_paths if Path(path).is_file()]
    require(
        type(execution.get("outputs")) is list and [n["path"] for n in execution["outputs"]] == present_paths,
        "execution output list differs from actual configured artifact presence",
    )
    require(type(execution.get("log")) is dict and execution["log"]["path"] == str(V1 / "truncated/native.log"), "different execution log")
    inputs = unique_records([record(execution_path), record(status_path), execution["log"], *execution["outputs"]])
    return execution, inputs


def partition_proofs(receipt, execution):
    """Read original KEYs and schemas only, before freezing value comparisons."""
    output_nodes = {node["path"]: node for node in execution["outputs"]}
    require(len(receipt["full_outputs"]) == len(receipt["prefix_outputs"]) == 4, "exact original pair data/event partition population required")
    proofs = []
    for i, (full_node, prefix_path) in enumerate(zip(receipt["full_outputs"], receipt["prefix_outputs"], strict=True)):
        partition, target = ("data" if i < 2 else "events"), producer.TARGETS[i % 2]
        schema = pq.read_schema(full_node["path"])
        schema_identity = native_schema(schema)
        keys = prefix.native_keys(pq.read_table(full_node["path"], columns=list(KEYS), use_threads=False))
        expected_keys = [key for key in keys if key[0] <= receipt["selected"]["D"]]
        present = Path(prefix_path).is_file()
        require(present or partition == "events" and not expected_keys, "missing periodic or nonempty original event prefix")
        if present:
            require(schema.equals(pq.read_schema(prefix_path), check_metadata=True), "full/prefix schema metadata differs")
            require(output_nodes.get(prefix_path) == record(Path(prefix_path)), "present prefix is not the executed immutable artifact")
        proofs.append(
            {
                "partition": partition,
                "target": target,
                "full": full_node,
                "prefix_path": prefix_path,
                "prefix_present": present,
                "prefix": output_nodes[prefix_path] if present else None,
                "full_key_rows": len(keys),
                "expected_key_rows": len(expected_keys),
                "expected_keys_identity": digest(expected_keys),
                "schema_identity": schema_identity,
                "absent_empty_event": not present,
            }
        )
    return proofs


def derive_bound():
    receipt, comparison, sources, inputs = parent_context()
    execution, execution_inputs = execution_evidence(receipt, comparison["identity"])
    for node in unique_records([*sources, *inputs, *execution_inputs]):
        check_record(node)
    return {
        "schema": BOUND_SCHEMA,
        "contract": CONTRACT,
        "contract_identity": digest(CONTRACT),
        "v1_comparison_identity": comparison["identity"],
        "selection_identity": receipt["selection_identity"],
        "preparation_identity": digest(receipt),
        "selected": receipt["selected"],
        "binary": receipt["binary"],
        "partitions": partition_proofs(receipt, execution),
        "sources": sources,
        "inputs": unique_records([*inputs, *execution_inputs]),
        "original_keys_only_read": True,
        "native_values_read": False,
        "native_replays_launched": 0,
        "generated_events": 0,
        "labels_read": False,
        "model_fits": 0,
    }


def bind_comparison():
    destination = OUTPUT / "bound-comparison.yaml"
    require(not destination.exists(), "fresh V2 comparison bind required; no overwrite/resume")
    bound = derive_bound()
    write_yaml(destination, bound)
    return bound


def verify(path):
    method = canonical_method(path, METHOD_SCHEMA)
    bound_path = OUTPUT / "bound-comparison.yaml"
    require(method["plan"] == record(bound_path), "different V2 comparison bind")
    bound = read_yaml(bound_path)
    current = derive_bound()
    require(bound == current, "frozen KEY-only absence/presence or parent evidence changed")
    for scope, nodes in (("sources", current["sources"]), ("inputs", [*current["inputs"], record(bound_path)])):
        producer.require_closure(method, scope, nodes)
        for node in method[scope]:
            check_record(node)
    return method, bound


def check(path):
    destination = OUTPUT / "validation.yaml"
    require(not destination.exists(), "fresh V2 validation receipt required")
    method, bound = verify(path)
    selected, checks = bound["selected"], []
    for proof in bound["partitions"]:
        result = {
            "partition": proof["partition"],
            "target": proof["target"],
            "full_path": proof["full"]["path"],
            "prefix_path": proof["prefix_path"],
            "prefix_present": proof["prefix_present"],
            "absent_empty_event": proof["absent_empty_event"],
        }
        if not proof["prefix_present"]:
            require(proof["partition"] == "events" and proof["expected_key_rows"] == 0 and not Path(proof["prefix_path"]).exists(), "frozen empty-event absence changed")
            result.update(common_rows=0, empty_original_key_population_proven=True, field_bit_checks={})
        else:
            full = pq.read_table(proof["full"]["path"], use_threads=False)
            truncated = pq.read_table(proof["prefix_path"], use_threads=False)
            require(native_schema(full.schema) == native_schema(truncated.schema) == proof["schema_identity"], "native schema drift after freeze")
            result.update(prefix.compare_tables(full, truncated, selected["D"]))
            if proof["partition"] == "data" and proof["target"] == selected["target"]:
                origin = tuple(selected["origin_keys"][name] for name in KEYS)
                keys = prefix.native_keys(truncated)
                require(origin in keys and origin[0] == selected["S"], "mandatory selected original30s S absent")
                index = keys.index(origin)
                require(
                    0 < truncated[f"H15_processed_{selected['leg']}_exchange_time"][index].as_py() < selected["pending_exchange"]
                    and truncated[f"H15_processed_{selected['leg']}_available_time"][index].as_py() <= selected["R"],
                    "pending own-E cluster published/backdated by EOF or clock",
                )
                result["selected_common_origin_present"] = True
        checks.append(result)
    require(any(c.get("selected_common_origin_present") is True for c in checks), "mandatory selected S comparison missing")
    result = {
        "schema": VALIDATION_SCHEMA,
        "passed": True,
        "genuine_straddle_found": True,
        "common_origin_present": True,
        "all_common_origin_bits_exact": True,
        "pending_cluster_not_published_at_origin": True,
        "empty_event_absence_exactly_proven": True,
        "producer_identity": prefix.PRODUCER_IDENTITY,
        "binary_sha256": bound["binary"]["sha256"],
        "selection_method_identity": bound["selection_identity"],
        "original_comparison_method_identity": bound["v1_comparison_identity"],
        "comparison_method_identity": method["identity"],
        "selected": selected,
        "tables": checks,
        "sources": method["sources"],
        "inputs": unique_records([*method["inputs"], record(path), record(path.with_suffix(".yaml.identity"))]),
        "native_replays_launched": 0,
        "labels_read": False,
        "model_fits": 0,
        "generated_events": 0,
        "predictive_admission": False,
        "v1_artifacts_modified": False,
    }
    write_yaml(destination, result)
    return {
        "passed": True,
        "receipt": str(destination),
        "present_native_tables": sum(p["prefix_present"] for p in bound["partitions"]),
        "absent_empty_events": sum(p["absent_empty_event"] for p in bound["partitions"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind-comparison", action="store_true")
    mode.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.bind_comparison:
        bound = bind_comparison()
        print(
            {
                "bound_only": True,
                "receipt": str(OUTPUT / "bound-comparison.yaml"),
                "sources": len(bound["sources"]),
                "inputs": len(bound["inputs"]),
                "native_values_read": False,
                "native_replays_launched": 0,
            }
        )
    else:
        print(check(args.check.resolve()))


if __name__ == "__main__":
    main()
