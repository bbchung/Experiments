"""Prepare/freeze V2, then use the existing AstraResearch.Experiments.signal representation runner.

The four arms and their bounded horizon follow-up use the V1 training/scoring
implementation verbatim. This helper only prepares a new representation and
closes its scientific source/cache dependencies before the first fit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...io import ContractError, file_hash, read_yaml, write_yaml
from ..representation_research import contract
from ..representation_research import runner as inherited
from . import renewal

JUDGE_FIELDS = (
    "data_floor",
    "horizons",
    "primary_horizon",
    "label",
    "sampling",
    "splits",
    "development_exposure",
    "sealed_oos",
    "selector",
    "universe",
    "training",
    "metrics",
    "acceptance",
)
COHORT_FIELDS = ("row_keys_roles_sha256", "rows_by_role", "labels_sha256")
ARMS = {
    "baseline": {"base": "current_nominal", "families": []},
    "old_context": {"base": "current_nominal", "families": ["graph", "session", "occupation"]},
    "renewal": {"base": "current_nominal", "families": ["renewal"]},
    "core": {"base": "core", "families": ["renewal"]},
}


def assert_inherited_judge(profile, parent):
    for name in JUDGE_FIELDS:
        if profile.get(name) != parent.get(name):
            raise ContractError(f"V2 changes frozen scientific judge field {name}; research methodology separately first")
    if profile.get("arms") != ARMS or list(profile["arms"]) != list(ARMS) or profile.get("baseline") != "baseline":
        raise ContractError("V2 requires its four predeclared baseline/complement/replacement arms")
    if profile["compute"]["primary_fits"] != 4 or profile["compute"]["secondary_fits_maximum"] != 6:
        raise ContractError("V2 fits must be bounded to four primary and six inherited follow-ups")


def parent_contract(profile):
    parent_path = Path(profile["inheritance"]["frozen_contract"])
    document = contract.verify(parent_path)
    if document["identity"] != profile["inheritance"]["identity"]:
        raise ContractError("V2 inherited method/cache identity changed")
    assert_inherited_judge(profile, document["profile"])
    return document


def hardlink(source, destination):
    """Bind a local immutable cache name without physically duplicating data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ContractError("Shared cache aliases must not be retargetable symlinks")
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ContractError(f"Shared cache identity changed: {destination}")
    else:
        os.link(source, destination)


def prepare_features(profile):
    parent = parent_contract(profile)
    original = Path(parent["profile"]["paths"]["output"]) / "shared"
    shared = inherited.output(profile) / "shared"
    counts, inputs = [], []
    for role in ("train", "tune", "forward"):
        for family in ("core", "graph", "session", "occupation"):
            filename = f"{role}-{family}.parquet"
            hardlink(original / filename, shared / filename)
        hardlink(original / f"mid-{role}.npy", shared / f"mid-{role}.npy")
        hardlink(original / f"mid-{role}.yaml", shared / f"mid-{role}.yaml")
        rows = pd.read_parquet(Path(profile["paths"]["parent_study"]) / role / "rows.parquet", columns=["day", "symbol", "SampleTime"])
        mids = np.load(shared / f"mid-{role}.npy", mmap_mode="r")
        if len(mids) != len(rows):
            raise ContractError("Renewal origin mid metadata is misaligned")
        rows["OriginMidPrice"] = mids
        inputs.append(rows)
        counts.append(len(rows))
    frame = pd.concat(inputs, ignore_index=True)
    features = renewal.build(frame, profile)
    if not features.index.equals(frame.index) or len(features) != len(frame):
        raise ContractError("Renewal representation changed native row identity")
    offsets = np.r_[0, np.cumsum(counts)]
    for role, start, stop in zip(("train", "tune", "forward"), offsets[:-1], offsets[1:], strict=True):
        features.iloc[start:stop].reset_index(drop=True).to_parquet(shared / f"{role}-renewal.parquet", index=False)
    write_yaml(
        shared / "features.yaml",
        {
            "columns": features.columns.tolist(),
            "rows": dict(zip(("train", "tune", "forward"), counts, strict=True)),
            "labels_used": False,
            "prior": "strictly_completed_prior_days",
            "physical_grid": features.attrs["half_cent_validation"],
        },
    )


def assert_same_cohort(parent, child):
    for field in COHORT_FIELDS:
        if parent[field] != child[field]:
            raise ContractError(f"V2 cohort differs from V1 in {field}")


def preflight(profile):
    # An interrupted rerun must not leave a prior success eligible for freeze.
    (inherited.output(profile) / "preflight.yaml").unlink(missing_ok=True)
    parent = parent_contract(profile)
    receipt = inherited.preflight(profile)
    previous = read_yaml(Path(parent["profile"]["paths"]["output"]) / "preflight.yaml")
    control = {entry["role"]: entry["comparison"] for entry in previous["entries"] if entry["arm"] == "baseline"}
    for entry in receipt["entries"]:
        assert_same_cohort(control[entry["role"]], entry["comparison"])
    receipt["inherited_judge_identical"] = True
    receipt["inherited_native_cohort_identical"] = True
    write_yaml(inherited.output(profile) / "preflight.yaml", receipt)
    return receipt


def source_files(profile, parent):
    experiment = Path(profile["_profile_path"]).parent
    return (
        [Path(item["path"]) for item in parent["sources"]]
        + sorted(Path(__file__).parent.glob("*.py"))
        + [
            experiment / "V2_PLAN.md",
            experiment / "H9_NOTES.md",
            experiment / "BOUNDARY_VALIDATION.md",
            experiment / "signal-v2.yaml",
        ]
    )


def freeze(profile):
    parent = parent_contract(profile)
    root = inherited.output(profile)
    receipt = read_yaml(root / "preflight.yaml")
    required = ("passed", "all_arms_identical_keys_roles_labels", "inherited_judge_identical", "inherited_native_cohort_identical")
    if not all(receipt.get(key) is True for key in required) or receipt.get("catboost_fits") != 0:
        raise ContractError("V2 requires its passed no-fit inherited-judge/cohort preflight")
    # Inherit all native material bindings, and freeze the original recipe,
    # manifest and identity anchor as well as the newly bound local shared cache.
    inputs = {item["path"]: Path(item["path"]) for item in parent["inputs"]}
    parent_path = Path(profile["inheritance"]["frozen_contract"])
    for path in (Path(parent["profile_source"]["path"]), parent_path, parent_path.with_name(parent_path.name + ".identity"), root / "preflight.yaml"):
        inputs[str(path)] = path
    for path in (root / "shared").iterdir():
        if path.is_file():
            inputs[str(path)] = path
    versions = {m.__name__: m.__version__ for m in (inherited.catboost, np, pd, inherited.pyarrow, inherited.scipy)}
    environment = root / "environment.yaml"
    write_yaml(environment, versions)
    inputs[str(environment)] = environment
    return contract.freeze(profile, root / "frozen-contract.yaml", source_files(profile, parent), inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("stage", choices=("features", "preflight", "freeze", "verify", "run"))
    args = parser.parse_args()
    profile = contract.load_profile(args.profile)
    stages = {
        "features": prepare_features,
        "preflight": preflight,
        "freeze": freeze,
        "verify": lambda p: contract.verify(inherited.output(p) / "frozen-contract.yaml"),
        "run": inherited.run,
    }
    stages[args.stage](profile)


if __name__ == "__main__":
    main()
