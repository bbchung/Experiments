"""Root-only H16 producer/support execution, with one process hash cache."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import native_pilot as study
from astra.io import read_yaml, write_yaml


def main():
    own = study.record(Path(__file__))
    result = study.prepare_metadata(study.ROOT / "build/Release/src/app/coco")
    payload = result.pop("producer_payload")
    payload["sources"] = study.unique_records([*payload["sources"], own])
    payload["native_outputs_decoded_before_freeze"] = 0
    producer_path = study.OUTPUT / "frozen-producer-method.yaml"
    identity = study.freeze_payload(payload, producer_path)
    print({**result, "producer_identity": identity}, flush=True)
    preparation = read_yaml(study.OUTPUT / "preparation-receipt.yaml")
    jobs = {
        "registration_control": preparation["registration_control"],
        **{job["key"]: job for job in preparation["jobs"]},
    }
    for key in preparation["execution_order"]:
        print(study.preflight(producer_path, key), flush=True)
        job = jobs[key]
        log = Path(job["work"]) / "native.log"
        study.require(not log.exists(), "fresh native log required")
        print(f"NATIVE START {key}", flush=True)
        with log.open("w") as stream:
            process = subprocess.run(
                job["command"],
                cwd=study.ROOT,
                env={**os.environ, **job["environment"]},
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        outputs = [study.record(path) for path in job["outputs"] if Path(path).is_file()]
        write_yaml(
            Path(job["work"]) / "execution-receipt.yaml",
            {
                "method_identity": identity,
                "job": key,
                "command": job["command"],
                "exit_code": process.returncode,
                "log": study.record(log),
                "outputs": outputs,
            },
        )
        study.require(process.returncode == 0 and len(outputs) == len(job["outputs"]), "native replay failed; inspect preserved receipt")
        print(f"NATIVE COMPLETE {key} artifacts={len(outputs)}", flush=True)
    payload = study.bind_evaluation(producer_path)
    payload["sources"] = study.unique_records([*payload["sources"], own])
    payload["native_outputs_decoded_before_freeze"] = 0
    evaluation_path = study.OUTPUT / "frozen-support-method.yaml"
    identity = study.freeze_payload(payload, evaluation_path)
    print({"support_identity": identity, "native_values_decoded": 0}, flush=True)
    print(study.check(evaluation_path), flush=True)


if __name__ == "__main__":
    main()
