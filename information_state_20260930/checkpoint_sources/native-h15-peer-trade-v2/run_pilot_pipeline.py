"""Root-only serialized native pilot, with producer and support freezes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import evaluate_native_support as support
import prepare_native as producer
from astra.io import digest, read_yaml, write_yaml


def freeze(path, body):
    producer.require(not path.exists() and not path.with_suffix(path.suffix + ".identity").exists(), "fresh canonical method required")
    body = {**body, "status": "frozen"}
    body["identity"] = digest(body)
    write_yaml(path, body)
    path.with_suffix(path.suffix + ".identity").write_text(body["identity"] + "\n")
    print(f"FROZEN {body['schema']} {body['identity']}", flush=True)
    return body


def main():
    root = producer.OUTPUT
    own = producer.record(Path(__file__))
    print(producer.prepare(), flush=True)
    preparation_path = root / "preparation-receipt.yaml"
    preparation = read_yaml(preparation_path)
    profile = read_yaml(producer.PROFILE)
    producer_path = root / "frozen-method.yaml"
    frozen = freeze(
        producer_path,
        {
            "schema": "h15-native-producer-method-v2",
            "preparation": producer.record(preparation_path),
            "preparation_identity": digest(preparation),
            "profile_identity": digest(profile),
            "sources": producer.unique_records([*preparation["sources"], own]),
            "inputs": producer.unique_records(
                [
                    *preparation["inputs"],
                    *[preparation[k] for k in ("script", "profile", "native_plan", "config_method_plan", "producer", "binary", "guide", "runtime_snapshot", "ldd")],
                    *preparation["runtime_dependencies"],
                    *[j["config"] for j in preparation["jobs"]],
                    preparation["registration_control"]["config"],
                    producer.record(preparation_path),
                ]
            ),
            "execution_order": list(producer.DAYS),
            "models_permitted": 0,
            "labels_permitted": False,
            "outputs_before_freeze": 0,
        },
    )
    jobs = {j["day"]: j for j in preparation["jobs"]}
    for day in preparation["execution_order"]:
        print(producer.preflight(producer_path, day), flush=True)
        job = jobs[day]
        if not job["requires_replay"]:
            print(f"SKIP {day}: frozen missing-input rule", flush=True)
            continue
        work = Path(job["work"])
        log = work / "native.log"
        producer.require(not log.exists(), "fresh native log required")
        print(f"NATIVE START {day}", flush=True)
        with log.open("w") as stream:
            execution = subprocess.run(job["command"], cwd=producer.ROOT, env={**os.environ, **job["environment"]}, stdout=stream, stderr=subprocess.STDOUT, check=False)
        outputs = [producer.record(Path(p)) for p in job["outputs"] if Path(p).is_file()]
        write_yaml(
            work / "execution-receipt.yaml",
            {"method_identity": frozen["identity"], "job": day, "command": job["command"], "exit_code": execution.returncode, "log": producer.record(log), "outputs": outputs},
        )
        producer.require(execution.returncode == 0 and len(outputs) == len(job["outputs"]), f"native {day} failed; inspect preserved execution receipt")
        status = read_yaml(work / "status" / f"{day}.yaml")
        producer.require(status.get("status") == "completed" and status.get("fatal_error") is False, "native status is not completed")
        print(f"NATIVE COMPLETE {day} artifacts={len(outputs)}", flush=True)
    print(support.bind_profile(), flush=True)
    evaluation_profile = read_yaml(support.EVALUATION_PROFILE)
    evaluation_path = root / "frozen-support-evaluation-method.yaml"
    links = evaluation_profile["producer_links"]
    freeze(
        evaluation_path,
        {
            "schema": "h15-native-support-evaluation-method-v2",
            "profile": producer.record(support.EVALUATION_PROFILE),
            "profile_identity": digest(evaluation_profile),
            "sources": producer.unique_records(
                [*frozen["sources"], producer.record(Path(support.__file__)), producer.record(support.TEST_SOURCE), producer.record(support.EVALUATION_PLAN)]
            ),
            "inputs": producer.unique_records(
                [
                    *frozen["inputs"],
                    *links,
                    *evaluation_profile["status_receipts"],
                    *[c[folder] for c in evaluation_profile["cells"] if c["status"] == "observed_native" for folder in ("data", "events")],
                    *[p[role] for p in evaluation_profile["registration_control"] for role in ("original", "produced")],
                    producer.record(support.EVALUATION_PROFILE),
                ]
            ),
            "outputs_decoded_before_freeze": 0,
            "raw_streams_decoded": 0,
            "labels_permitted": False,
            "models_permitted": 0,
        },
    )
    print(support.evaluate(evaluation_path, root / "source-support-validation.yaml"), flush=True)


if __name__ == "__main__":
    main()
