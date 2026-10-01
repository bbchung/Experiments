"""Root-only fixed mapping producer and separately frozen coverage evaluation."""

import os
import subprocess
from pathlib import Path

import coverage as study
from astra.io import digest, read_yaml, write_yaml
from run_pilot_pipeline import freeze


def main():
    own = study.record(Path(__file__))
    print(study.prepare(), flush=True)
    prep = read_yaml(study.PREPARATION)
    profile = read_yaml(study.PROFILE)
    producer_path = study.OUTPUT / "frozen-method.yaml"
    producer = freeze(
        producer_path,
        {
            "schema": "h15-fixed-mapping-producer-method-v1",
            "preparation": study.record(study.PREPARATION),
            "preparation_identity": digest(prep),
            "profile_identity": digest(profile),
            "sources": study.unique_records([*prep["sources"], own]),
            "inputs": study.unique_records(
                [
                    *prep["inputs"],
                    *[prep[k] for k in ("script", "profile", "binary", "guide", "shared_source_capsule", "runtime_snapshot")],
                    *prep["runtime_dependencies"],
                    *[j["config"] for j in prep["jobs"]],
                    study.record(study.PREPARATION),
                ]
            ),
            "outputs_decoded_before_freeze": 0,
            "models_permitted": 0,
        },
    )
    for job in prep["jobs"]:
        day = job["day"]
        print(study.preflight(producer_path, day), flush=True)
        if not job["requires_replay"]:
            print(f"SKIP {day}: frozen missing-input rule", flush=True)
            continue
        work = Path(job["work"])
        log = work / "native.log"
        study.require(not log.exists(), "fresh native log required")
        print(f"NATIVE START {day}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(job["command"], cwd=study.ROOT, env={**os.environ, **job["environment"]}, stdout=stream, stderr=subprocess.STDOUT, check=False)
        outputs = [study.record(Path(p)) for p in job["outputs"] if Path(p).is_file()]
        write_yaml(
            work / "execution-receipt.yaml",
            {"method_identity": producer["identity"], "job": day, "command": job["command"], "exit_code": result.returncode, "log": study.record(log), "outputs": outputs},
        )
        study.require(result.returncode == 0 and len(outputs) == len(job["outputs"]), "native mapping replay failed; inspect receipt")
        print(f"NATIVE COMPLETE {day} artifacts={len(outputs)}", flush=True)
    print(study.bind_evaluation(producer_path), flush=True)
    ep = read_yaml(study.EVALUATION_PROFILE)
    evaluation_path = study.OUTPUT / "frozen-evaluation-method.yaml"
    freeze(
        evaluation_path,
        {
            "schema": "h15-fixed-mapping-evaluation-method-v1",
            "profile": study.record(study.EVALUATION_PROFILE),
            "profile_identity": digest(ep),
            "sources": study.unique_records([*producer["sources"], study.record(study.PARENT / "evaluate_native_support.py")]),
            "inputs": study.unique_records(
                [
                    *producer["inputs"],
                    ep["producer"],
                    ep["producer_anchor"],
                    *ep["status_receipts"],
                    *[c["data"] for c in ep["cells"] if c["status"] == "observed_native"],
                    study.record(study.EVALUATION_PROFILE),
                ]
            ),
            "outputs_decoded_before_freeze": 0,
            "models_permitted": 0,
        },
    )
    print(study.evaluate(evaluation_path, study.OUTPUT / "coverage-validation.yaml"), flush=True)


if __name__ == "__main__":
    main()
