"""Root-only genuinely truncated native receive-prefix proof."""

import os
import subprocess
from pathlib import Path

import prepare_receive_prefix as study
from astra.io import digest, read_yaml, write_yaml
from run_pilot_pipeline import freeze


def main():
    own = study.record(Path(__file__))
    print(study.bind_plan(), flush=True)
    bound_path = study.OUTPUT / "bound-plan.yaml"
    bound = read_yaml(bound_path)
    selection_path = study.OUTPUT / "frozen-selection-method.yaml"
    freeze(
        selection_path,
        {
            "schema": "h15-real-receive-prefix-selection-method-v1",
            "plan": study.record(bound_path),
            "profile_identity": digest(study.CONTRACT),
            "sources": study.unique_records([*bound["sources"], own]),
            "inputs": study.unique_records([*bound["inputs"], study.record(bound_path)]),
        },
    )
    print(study.prepare(selection_path), flush=True)
    preparation = read_yaml(study.OUTPUT / "preparation-receipt.yaml")
    if not preparation["genuine_straddle_found"]:
        print("NO GENUINE STRADDLE; native proof is not established", flush=True)
        return
    print(study.bind_comparison(selection_path), flush=True)
    comparison_bound_path = study.OUTPUT / "bound-comparison.yaml"
    comparison_bound = read_yaml(comparison_bound_path)
    comparison_path = study.OUTPUT / "frozen-comparison-method.yaml"
    method = freeze(
        comparison_path,
        {
            "schema": "h15-real-receive-prefix-comparison-method-v1",
            "plan": study.record(comparison_bound_path),
            "sources": study.unique_records([*comparison_bound["sources"], own]),
            "inputs": study.unique_records([*comparison_bound["inputs"], study.record(comparison_bound_path)]),
        },
    )
    work = Path(preparation["config"]["path"]).parent
    log = work / "native.log"
    study.require(not log.exists(), "fresh native prefix log required")
    print("NATIVE PREFIX START", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            preparation["command"], cwd=study.producer.ROOT, env={**os.environ, **preparation["environment"]}, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    outputs = [study.record(Path(p)) for p in preparation["prefix_outputs"] if Path(p).is_file()]
    write_yaml(
        work / "execution-receipt.yaml",
        {
            "method_identity": method["identity"],
            "job": "receive_prefix",
            "command": preparation["command"],
            "exit_code": result.returncode,
            "log": study.record(log),
            "outputs": outputs,
        },
    )
    study.require(result.returncode == 0 and len(outputs) == len(preparation["prefix_outputs"]), "native receive-prefix replay failed; inspect execution receipt")
    print(study.check(comparison_path), flush=True)


if __name__ == "__main__":
    main()
