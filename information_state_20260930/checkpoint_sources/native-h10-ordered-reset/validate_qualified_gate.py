import json
from pathlib import Path

from astra.io import ContractError, digest, file_hash, read_yaml, write_yaml

R = Path(__file__).resolve().parent
A = R.parents[2]


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def unique(items):
    out = {}
    for q in items:
        p = str(Path(q["path"]).resolve())
        if p in out and out[p]["sha256"] != q["sha256"]:
            raise ContractError("Conflicting dependency SHA")
        if file_hash(Path(p)) != q["sha256"]:
            raise ContractError(f"Changed dependency: {p}")
        out[p] = {"path": p, "sha256": q["sha256"]}
    return list(out.values())


def main():
    dest = R / "validation.yaml"
    if dest.exists():
        raise FileExistsError(dest)
    pilot = read_yaml(R / "independent-native-validation.yaml")
    lineage = read_yaml(R / "producer-lineage-validation.yaml")
    method = read_yaml(R / "component-method-frozen.yaml")
    full = read_yaml(R / "full-preparation-receipt.yaml")
    checks = {
        "fixed_four_cell_mechanism_support": all(pilot["support_checks"].values()),
        "original_native_feature_integrity": pilot["native_feature_integrity"] is True,
        "original_gate_failed_preserved": pilot["original_screen_pass"] is False and lineage["historical_native_gate_v1_preserved_failed"] is True,
        "fixed_historical_parent_reproduced": lineage["baseline_parent_reproduced"] is True,
        "new_cpp_control_invariant": lineage["current_control_invariant"] is True,
        "historical_components_exactly_accounted": lineage["historical_components_accounted"] is True,
        "candidate_exact_original_join": lineage["candidate_exact_original_join"] is True,
        "component_method_passed": lineage["passed"] is True,
        "no_judge_or_value_translation": lineage["fe_evaluation_judge_changed"] is False and lineage["feature_or_label_translation_performed"] is False,
        "frozen_method_identity": method["identity"]
        == digest({k: v for k, v in method.items() if k != "identity"})
        == lineage["method_contract_identity"]
        == (R / "component-method-frozen.yaml.identity").read_text().strip(),
        "same_immutable_binary": file_hash(R / "coco") == lineage["h10_binary_sha256"] == full["binary"]["sha256"],
    }
    sources = unique(lineage["sources"] + [record(Path(__file__)), record(A / "experiments/information_state_20260930/H10_COMPONENT_GATE_V2.md")])
    inputs = unique(
        lineage["inputs"]
        + [record(R / n) for n in ["component-lineage-manifest.yaml", "producer-lineage-validation.yaml", "full-preparation-receipt.yaml", "independent-native-validation.yaml"]]
        + [{k: v for k, v in q.items() if k in ("path", "sha256")} for q in full["raw_inputs"] + full["basic_info_inputs"]]
        + [{"path": j["config"]["path"], "sha256": j["config"]["sha256"]} for j in full["jobs"]]
    )
    result = {
        "schema": "h10-native-component-qualified-materialization-v2",
        "passed": all(checks.values()),
        "binary_sha256": file_hash(R / "coco"),
        "checks": checks,
        "original_native_gate_v1_passed": False,
        "qualified_component_gate": True,
        "full_cohort_integrity_checked_in_prepare": True,
        "native_prefix_required_before_fe_freeze": True,
        "predictive_admission": False,
        "sources": sources,
        "inputs": inputs,
    }
    write_yaml(dest, result)
    print(json.dumps({"passed": result["passed"], "checks": checks, "sources": len(sources), "inputs": len(inputs)}))
    if not result["passed"]:
        raise ContractError("Qualified native gate failed")


if __name__ == "__main__":
    main()
