"""Root freezes the source census before reading the raw-message witnesses."""

from pathlib import Path

import census as study
from run_pilot_pipeline import freeze


def main():
    bound_path = study.OUTPUT / "bound-plan.yaml"
    study.raw.require(not bound_path.exists(), "fresh native census plan required")
    bound = study.bind()
    study.raw.write_yaml(bound_path, bound)
    method_path = study.OUTPUT / "frozen-method.yaml"
    freeze(
        method_path,
        {
            "schema": "h16-actual-auction-anchor-census-method-v1",
            "plan": study.raw.record(bound_path),
            "profile_identity": study.raw.digest(study.CONTRACT),
            "sources": study.raw.unique_records([*bound["sources"], study.raw.record(Path(__file__))]),
            "inputs": study.raw.unique_records([*bound["inputs"], study.raw.record(bound_path)]),
        },
    )
    print(study.census(method_path), flush=True)


if __name__ == "__main__":
    main()
