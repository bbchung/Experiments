"""Root-only binding repair; immutable H16 V1 science and preflight retained."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PARENT = OUTPUT.parent / "native-h16-auction-cycle"
sys.path.insert(0, str(PARENT))
import native_pilot as study
from astra.io import digest, read_yaml, write_yaml

PARENT_ID = "957a75f8b3de487b949fee619693c8153f42815cc49ba275fd2b807531dd8539"


def repaired_payload(parent, profile, parent_links, own_sources):
    """Add the already frozen same profile to inputs; permit no science edit."""
    require = study.require
    require(parent.get("identity") == PARENT_ID, "wrong rejected producer")
    sources = {node["path"]: node["sha256"] for node in parent["sources"]}
    require(sources.get(profile["path"]) == profile["sha256"], "repair profile must be the exact already frozen source")
    inputs = {node["path"]: node["sha256"] for node in parent["inputs"]}
    require(profile["path"] not in inputs, "repair applies only to the rejected missing-input binding")
    return {
        **{key: value for key, value in parent.items() if key not in ("identity", "status", "sources", "inputs")},
        "status": "draft",
        "binding_version": 2,
        "rejected_parent_identity": PARENT_ID,
        "repair": "same_frozen_profile_added_to_input_closure",
        "sources": study.unique_records([*parent["sources"], *own_sources]),
        "inputs": study.unique_records([*parent["inputs"], profile, *parent_links]),
        "native_outputs_decoded_before_freeze": 0,
        "feature_math_or_procedure_changed": False,
    }


def prepare_repair():
    parent_path = PARENT / "frozen-producer-method.yaml"
    parent = study.canonical_method(parent_path, study.PRODUCER_SCHEMA)
    study.require(parent["identity"] == PARENT_ID, "wrong immutable rejected parent")
    for scope in ("sources", "inputs"):
        for node in parent[scope]:
            study.check_record(node)
    preparation = read_yaml(study.check_record(parent["preparation"]))
    study.require(digest(preparation) == parent["preparation_identity"], "original preparation changed")
    for job in [preparation["registration_control"], *preparation["jobs"]]:
        study.require(job is not None, "unchanged native jobs mandatory")
        study.require(not (Path(job["work"]) / "execution-receipt.yaml").exists(), "binding repair requires no previous execution")
        study.require(not (Path(job["work"]) / "native.log").exists(), "binding repair requires no previous native launch")
        study.require(all(not Path(path).exists() for path in job["outputs"]), "binding repair requires zero previous native outputs")
    profile = preparation["profile"]
    study.check_record(profile)
    links = [study.record(parent_path), study.record(parent_path.with_suffix(".yaml.identity"))]
    sources = [study.record(path) for path in (Path(__file__), OUTPUT / "test_repair.py", OUTPUT / "REPAIR_PLAN.md")]
    return repaired_payload(parent, profile, links, sources), preparation


def main():
    payload, preparation = prepare_repair()
    producer_path = OUTPUT / "frozen-producer-method.yaml"
    identity = study.freeze_payload(payload, producer_path)
    print({"producer_identity": identity, "repair": payload["repair"], "native_outputs_before_freeze": 0}, flush=True)
    jobs = {"registration_control": preparation["registration_control"], **{job["key"]: job for job in preparation["jobs"]}}
    for key in preparation["execution_order"]:
        print(study.preflight(producer_path, key), flush=True)
        job = jobs[key]
        log = Path(job["work"]) / "native.log"
        study.require(not log.exists(), "fresh native log required")
        print(f"NATIVE START {key}", flush=True)
        with log.open("w") as stream:
            process = subprocess.run(job["command"], cwd=study.ROOT, env={**os.environ, **job["environment"]}, stdout=stream, stderr=subprocess.STDOUT, check=False)
        outputs = [study.record(path) for path in job["outputs"] if Path(path).is_file()]
        write_yaml(
            Path(job["work"]) / "execution-receipt.yaml",
            {"method_identity": identity, "job": key, "command": job["command"], "exit_code": process.returncode, "log": study.record(log), "outputs": outputs},
        )
        study.require(process.returncode == 0 and len(outputs) == len(job["outputs"]), "native replay failed; inspect preserved receipt")
        print(f"NATIVE COMPLETE {key} artifacts={len(outputs)}", flush=True)
    payload = study.bind_evaluation(producer_path)
    payload["sources"] = study.unique_records([*payload["sources"], *[study.record(path) for path in (Path(__file__), OUTPUT / "test_repair.py", OUTPUT / "REPAIR_PLAN.md")]])
    payload["native_outputs_decoded_before_freeze"] = 0
    evaluation_path = OUTPUT / "frozen-support-method.yaml"
    identity = study.freeze_payload(payload, evaluation_path)
    print({"support_identity": identity, "native_values_decoded": 0}, flush=True)
    print(study.check(evaluation_path), flush=True)


if __name__ == "__main__":
    main()
