"""Root-only frozen comparison of the completed genuine native prefix."""

from pathlib import Path

import validate_receive_prefix as study
from run_pilot_pipeline import freeze


def main():
    bound = study.bind_comparison()
    print({"bound": True, "partitions": bound["partitions"]}, flush=True)
    bound_path = study.OUTPUT / "bound-comparison.yaml"
    method_path = study.OUTPUT / "frozen-comparison-method.yaml"
    freeze(
        method_path,
        {
            "schema": study.METHOD_SCHEMA,
            "plan": study.record(bound_path),
            "sources": study.unique_records([*bound["sources"], study.record(Path(__file__))]),
            "inputs": study.unique_records([*bound["inputs"], study.record(bound_path)]),
        },
    )
    print(study.check(method_path), flush=True)


if __name__ == "__main__":
    main()
