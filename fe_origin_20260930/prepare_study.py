"""Prepare frozen train/tune type projections while train-only PMQ runs."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from study_data import census, numeric_domain, project, project_nominal

from AstraResearch.contracts import ArtifactRef
from AstraResearch.io import ContractError, read_yaml, write_yaml
from AstraResearch.kernels.funnel import numeric_features
from AstraResearch.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.yaml").exists():
        raise ContractError("Preparation cannot write into an already fitting study")
    profile = read_yaml(args.material / "profile.yaml")
    bindings = read_yaml(args.material / "baseline-state.yaml")["bindings"]
    if not all(role in bindings for role in ("train", "tune")):
        raise ContractError("Train and tune materials are required")
    store = Store(Path(profile["paths"]["store"]))
    train = store.resolve(ArtifactRef(**bindings["train"]))
    numeric, nominal = numeric_features(train)
    numeric = numeric_domain(train, numeric, args.output)
    for role in ("train", "tune"):
        dataset = train if role == "train" else store.resolve(ArtifactRef(**bindings[role]))
        matrix, rows, health = project(dataset, numeric, args.output / role)
        census(rows, args.output, role)
        if nominal:
            strings = project_nominal(dataset, list(nominal), args.output / role, rows)
            del strings
        print(role, len(rows), "native origins prepared", flush=True)
        del matrix, rows, health
        gc.collect()
    write_yaml(
        args.output / "preparation.yaml",
        {"bindings": {role: bindings[role] for role in ("train", "tune")}, "numeric_columns": len(numeric), "forward_read": False, "model_fitted": False},
    )


if __name__ == "__main__":
    main()
