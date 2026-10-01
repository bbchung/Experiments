import hashlib
from pathlib import Path

import yaml
from astra.io import digest

R = Path(__file__).resolve().parent
A = R.parents[2]


def read(p):
    return yaml.safe_load(p.read_text())


def node(p):
    p = Path(p).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def write_new(p, d):
    if p.exists():
        raise FileExistsError(p)
    p.write_text(yaml.safe_dump(d, sort_keys=False))


def main():
    proof = read(R / "original-reproduction.yaml")
    assert proof["complete"] and proof["passed"]
    pilot = read(R / "independent-native-validation.yaml")
    assert pilot["original_screen_pass"] is False
    assert pilot["native_feature_integrity"] and pilot["preserved_new_invariance"]
    assert all(pilot["support_checks"].values())
    runtime = read(R / "runtime-snapshot.yaml")
    snapshot = read(R / "compiled-source-snapshot.yaml")
    prep = read(R / "preparation-receipt.yaml")
    full = read(R / "full-preparation-receipt.yaml")
    compiled = [{"path": p["path"], "sha256": p["sha256"]} for p in snapshot["sources"]]
    for p in compiled:
        assert node(p["path"]) == p
    gate_path = R / "gate-v1-failure.yaml"
    if not gate_path.exists():
        write_new(
            gate_path,
            {
                "schema": "preserved-native-gate-v1-failure",
                "passed": False,
                "support_only_passed": True,
                "historical_original_all_byte_correspondence": False,
                "independent_report": node(R / "independent-native-validation.yaml"),
                "original_gate": node(A / "experiments/information_state_20260930/H10_NATIVE_SCREEN.md"),
                "qualification": "Historical whole-producer correspondence failed and remains failed. Component-lineage validation is a separate method, not a reclassification.",
            },
        )
    src = [node(p) for p in sorted((A / "astra/representation_provenance").glob("*.py"))]
    src += [
        node(A / "experiments/information_state_20260930/H10_COMPONENT_GATE_V2.md"),
        node(A / "tests/test_representation_provenance.py"),
        node(A / "experiments/information_state_20260930/PROVENANCE_METHOD.md"),
        node(Path(__file__)),
        node(A / "astra/io.py"),
        node(R / "validate_qualified_gate.py"),
    ]
    inputs = [
        node(R / name)
        for name in [
            "producer-reproduction-plan.yaml",
            "producer-reproduction-plan.yaml.identity",
            "producer-reproduction-plan-v2.yaml",
            "producer-reproduction-plan-v2.yaml.identity",
            "runtime-snapshot.yaml",
            "original-reproduction.yaml",
            "producer-source-build.yaml",
            "compiled-source-snapshot.yaml",
            "preparation-receipt.yaml",
            "baseline-parity.yaml",
            "replay-receipt.yaml",
            "independent-native-validation.yaml",
            "gate-v1-failure.yaml",
            "provenance-tests-final.log",
        ]
    ]
    inputs += [
        node(A / "runs/fe_origin_20260930/warmup-state-parity-1/results.yaml"),
        node(A / "experiments/fe_origin_20260930/SOURCE_AUDIT.md"),
        node(A / "runs/fe_origin_20260930/frozen-native-engine/receipt.json"),
    ]
    inputs.append(node(R / "full-preparation-receipt.yaml"))
    for item in full["raw_inputs"]:
        if item["day"] in ["20260119", "20260120", "20260203"]:
            inputs.append({"path": item["path"], "sha256": item["sha256"]})
    for item in full["basic_info_inputs"]:
        if Path(item["path"]).stem in ["20260119", "20260120", "20260203"]:
            inputs.append({"path": item["path"], "sha256": item["sha256"]})
    inputs.append(node(A / "experiments/fe_origin_20260930/calendar.yaml"))
    for p in runtime["dependencies"]:
        inputs.append(node(p["path"]))
    for p in [A / "runs/fe_origin_20260930/frozen-native-engine/coco", R.parent / "native-baseline/coco", R / "coco"]:
        inputs.append(node(p))
    deps = [node(p["path"]) for p in runtime["dependencies"]]
    correction = node(A / "runs/fe_origin_20260930/warmup-state-parity-1/results.yaml")
    producers = {
        "original": {
            "binary": node(A / "runs/fe_origin_20260930/frozen-native-engine/coco"),
            "runtime_dependencies": deps,
            "source_receipts": [node(A / "runs/fe_origin_20260930/frozen-native-engine/receipt.json")],
        },
        "current": {"binary": node(R.parent / "native-baseline/coco"), "runtime_dependencies": deps, "source_receipts": [node(R / "preparation-receipt.yaml"), correction]},
        "h10": {"binary": node(R / "coco"), "runtime_dependencies": deps, "source_receipts": [node(R / "producer-source-build.yaml"), node(R / "compiled-source-snapshot.yaml")]},
    }
    controls = []
    for pair in prep["parity_pairs"]:
        d, s = pair["day"], pair["symbol"]
        records = [q for q in proof["records"] if q["symbol"] == s]
        assert len(records) == 1
        controls.append(
            {
                "day": d,
                "symbol": s,
                "parent_original": node(pair["original_native"]),
                "original_reproduced": node(records[0]["output"]),
                "current_preserved": node(R / "parity/preserved" / s / "data" / d / s / "values.parquet"),
                "h10_control": node(R / "parity/new" / s / "data" / d / s / "values.parquet"),
                "input_receipts": [
                    node(pair["original_config"]),
                    node(R / "parity/preserved" / s / "config.yaml"),
                    node(R / "parity/new" / s / "config.yaml"),
                    node(R / "original-reproduction" / s / "config.yaml"),
                    node(R / "preparation-receipt.yaml"),
                    node(R / "producer-reproduction-plan-v2.yaml"),
                ],
            }
        )
    joins = []
    for job in prep["pilot_jobs"]:
        d, s = job["day"], job["symbol"]
        w = Path(job["work"])
        joins.append(
            {
                "day": d,
                "symbol": s,
                "parent_original": node(job["original_native"]),
                "h10_features": node(w / "data" / d / s / "values.parquet"),
                "input_receipts": [node(w / "config.yaml"), node(job["original_config"]), node(R / "preparation-receipt.yaml")],
            }
        )
    payload = {
        "schema": "component-lineage-manifest-v2",
        "reproduction_plan": node(R / "producer-reproduction-plan-v2.yaml"),
        "historical_correction_receipt": correction,
        "gate_v1_failure": node(gate_path),
        "producers": producers,
        "controls": controls,
        "candidate_joins": joins,
    }
    method = {
        "schema": "native-component-provenance-method-v2",
        "manifest_payload_identity": digest(payload),
        "procedure": "strict_binary_reproduction_plus_documented_component_accounting_and_exact_candidate_join",
        "scope": {"control_cells": [["20260203", "2609"], ["20260203", "2337"]], "candidate_cells": [[d, s] for d in ["20260119", "20260120"] for s in ["2308", "2317"]]},
        "policy": {
            "historical_all_byte_parity_failed": True,
            "historical_runtime_equality_claimed": False,
            "current_frozen_runtime_original_parent_reproduction_required": True,
            "exact_9flow_14cross_prior_receipt_counts": True,
            "unaccounted_diff_reject": True,
            "exact_mid_keys_labels": True,
            "native_receive_availability_not_future": True,
            "numeric_missing_states_preserved": True,
            "no_translation": True,
            "fe_judge_changed": False,
            "new_baseline_after_method_freeze": True,
        },
        "method_validation": {
            "independent_synthetic_tests": "tests/test_representation_provenance.py before this freeze",
            "positive_native_proof": node(R / "original-reproduction.yaml"),
            "no_model_fits": True,
        },
        "sources": src + compiled,
        "inputs": inputs,
    }
    method["identity"] = digest(method)
    mp = R / "component-method-frozen.yaml"
    write_new(mp, method)
    mp.with_name(mp.name + ".identity").write_text(method["identity"] + "\n")
    manifest = {
        **payload,
        "method_contract": {**node(mp), "identity_path": str(mp) + ".identity", "identity_sha256": node(str(mp) + ".identity")["sha256"], "identity": method["identity"]},
    }
    write_new(R / "component-lineage-manifest.yaml", manifest)
    print("METHOD_FROZEN", method["identity"], len(method["sources"]), len(method["inputs"]))


if __name__ == "__main__":
    main()
