"""Freeze and verify the evaluation contract (the judge) before any FE comparison.

The identity covers every choice that can change a comparison outcome other
than the feature arms: data floor, universe rule and its screened result,
roles, session sampling, labels, training formulation, the judge section,
the cohort rows/labels and the Python sources that fit and score. Rounds are
scored only when the live identity equals the frozen record.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...io import ContractError, read_yaml, write_yaml
from . import dataset
from . import profile as P

SECTIONS = ("data_floor", "universe", "dates", "session", "labels", "training", "judge")
SOURCES = ("profile.py", "dataset.py", "train.py", "judge.py", "compare.py", "freeze.py")


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(profile) -> dict:
    here = Path(__file__).parent
    record = {
        "sections": P.section_digest(profile, SECTIONS),
        "universe_result": file_sha(P.work(profile) / "universe" / "universe.yaml"),
        "cohort": file_sha(dataset.cohort_path(profile)),
        "sources": {name: file_sha(here / name) for name in SOURCES},
        "metrics_source": file_sha(here.parent / "signal" / "metrics.py"),
    }
    record["identity"] = hashlib.sha256(P.canonical(record)).hexdigest()
    return record


def frozen_path(profile) -> Path:
    return Path(profile["_path"]).parent / "judge-frozen.yaml"


def freeze(profile, note: str):
    path = frozen_path(profile)
    if path.exists():
        raise ContractError(f"{path} exists; a changed judge needs a new study version, not an overwrite")
    record = {**identity(profile), "note": note}
    write_yaml(path, record)
    return record


def verify(profile) -> str:
    path = frozen_path(profile)
    if not path.exists():
        raise ContractError("The judge is not frozen; methodology must be frozen before FE comparison")
    frozen = read_yaml(path)
    live = identity(profile)
    if live["identity"] != frozen["identity"]:
        diff = {k for k in ("sections", "universe_result", "cohort", "metrics_source") if live[k] != frozen[k]}
        diff |= {f"source:{k}" for k in live["sources"] if live["sources"][k] != frozen["sources"].get(k)}
        raise ContractError(f"Live judge identity differs from the frozen record: {sorted(diff)}")
    return frozen["identity"]
