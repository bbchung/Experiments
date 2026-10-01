"""Explicit frozen-plan runner for a four-cell, label-free H13 support proxy.

Importing this module does not read market files. An actual run is refused
unless a canonical frozen contract binds the entire prospective profile,
running source closure, existing receipts and every exact source artifact.
"""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from AstraResearch.io import digest, file_hash, read_yaml, write_yaml

from . import raw
from .support import FIXED_CELLS, Contract, Origin, cell_summary, evaluate_gates, replay, share_price_key

NATIVE_KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")
SCHEMA = "h13-acceptance-support-method-v1"
PACKAGE = Path(__file__).resolve().parent
SOURCE_PROOF = (
    PACKAGE.parents[1] / "tests/test_representation_acceptance_support.py",
    PACKAGE.parents[1] / "Experiments/information_state_20260930/H13_SUPPORT_PLAN.md",
)


def _hash(path: Path) -> str:
    return file_hash(path)


def _check_record(record, *, allow_missing=False):
    path = Path(record["path"])
    if not path.is_absolute() or len(record["sha256"]) != 64:
        raise ValueError("absolute artifact path and SHA256 required")
    if not path.is_file() and allow_missing:
        return None
    if not path.is_file() or _hash(path) != record["sha256"]:
        raise ValueError(f"missing or changed bound artifact: {path}")
    return path


def validate_frozen_plan(profile_path: Path, contract_path: Path):
    profile = read_yaml(profile_path)
    contract = read_yaml(contract_path)
    anchor_path = contract_path.with_suffix(contract_path.suffix + ".identity")
    anchor = anchor_path.read_text().strip()
    identity = digest({key: value for key, value in contract.items() if key != "identity"})
    if contract.get("schema") != SCHEMA or contract.get("status") != "frozen" or contract.get("identity") != anchor or anchor != identity:
        raise ValueError("canonical frozen method and identity sidecar required")
    if contract.get("profile_identity") != digest(profile):
        raise ValueError("prospective profile differs from frozen method")
    _check_record(contract["profile"])
    if Path(contract["profile"]["path"]).resolve() != profile_path.resolve():
        raise ValueError("method binds a different profile path")
    sources = {str(Path(record["path"]).resolve()): record for record in contract["sources"]}
    required_sources = [*PACKAGE.glob("*.py"), Path(__file__).resolve().parents[2] / "io.py", *SOURCE_PROOF]
    if not all(str(path.resolve()) in sources for path in required_sources):
        raise ValueError("frozen method must bind every running package source and AstraResearch.io")
    for record in sources.values():
        _check_record(record)
    proof = {str(Path(record["path"]).resolve()): record for record in profile["source_proof"]}
    if set(proof) != {str(path.resolve()) for path in SOURCE_PROOF} or any(sources.get(path, {}).get("sha256") != record["sha256"] for path, record in proof.items()):
        raise ValueError("essential prospective plan/test source proof omitted or changed")
    bound_inputs = {str(Path(record["path"]).resolve()): record for record in contract["inputs"]}
    cell_inputs = [cell[key] for cell in profile["cells"] for key in ("raw", "basic_info", "parent_origins", "expected_native_keys")]
    requested = [*profile["provenance_inputs"], *cell_inputs]
    for record in requested:
        if bound_inputs.get(str(Path(record["path"]).resolve()), {}).get("sha256") != record["sha256"]:
            raise ValueError("frozen method omitted or replaced a prospective input identity")
    # Metadata and hashes are verified before any source decoding or origin read.
    cell_paths = {str(Path(record["path"]).resolve()) for record in cell_inputs}
    for path, record in bound_inputs.items():
        _check_record(record, allow_missing=path in cell_paths)
    cells = profile["cells"]
    if [(cell["day"], cell["symbol"]) for cell in cells] != list(FIXED_CELLS):
        raise ValueError("fixed chronological four-cell profile required")
    if profile.get("schema") != "h13-acceptance-support-profile-v1":
        raise ValueError("wrong prospective support profile schema")
    expected_contract = {
        "sampling_seconds": 30,
        "expected_native_rows": 1381,
        "expected_sampled_rows": 461,
        "max_book_age_micros": 5_000_000,
        "recent_support_age_micros": 300_000_000,
        "minimum_completions_per_cell": 5,
        "minimum_distinct_recent_sampled_landmarks_per_cell": 5,
        "minimum_completions_each_direction": 5,
        "minimum_provisional_or_unilateral_origins": 20,
        "minimum_observed_fixed_cells": 1,
    }
    if any(type(value) is not int for value in profile["contract"].values()) or profile["contract"] != expected_contract:
        raise ValueError("fixed support/sampling gates changed")
    return profile, contract


def read_origins(cell):
    native_path = Path(cell["parent_origins"]["path"])
    table = pq.read_table(native_path, columns=list(NATIVE_KEYS), use_threads=False)
    if table.num_rows != 1381 or any(table[name].type != pa.int64() or table[name].null_count for name in NATIVE_KEYS):
        raise ValueError("parent origin row count, int64 keys or null contract changed")
    keys = list(zip(*(table[name].to_pylist() for name in NATIVE_KEYS), strict=True))
    # H10 expected_keys_sha256 hashes its PARQUET artifact, not the key bytes.
    expected_path = _check_record(cell["expected_native_keys"])
    expected = table if expected_path.resolve() == native_path.resolve() else pq.read_table(expected_path, columns=list(NATIVE_KEYS), use_threads=False)
    if expected.num_rows != 1381 or any(not table[name].equals(expected[name]) for name in NATIVE_KEYS):
        raise ValueError("original complete native keys differ from bound preparation")
    if any(right[0] <= left[0] for left, right in pairwise(keys)) or any(
        not 0 < sample < 2**53 or not 0 <= book <= sample or not 0 <= sequence < 2**53 for sample, book, sequence in keys
    ):
        raise ValueError("parent source-key order or boundary contract invalid")
    sampled = [Origin(*key) for key in keys if key[0] % 30_000_000 == 0]
    if len(sampled) != 461 or any(right.time - left.time != 30_000_000 for left, right in pairwise(sampled)):
        raise ValueError("original 30s origin subset changed; no synthetic grid permitted")
    return sampled


def read_contract(cell):
    with Path(cell["basic_info"]["path"]).open(newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["symbol"] == cell["symbol"]]
    if not selected:
        return None
    if len(selected) != 1:
        raise ValueError("fixed ordinary-share contract duplicated")
    contract = Contract(float(selected[0]["limit_up"]), float(selected[0]["limit_down"]))
    if share_price_key(contract.limit_up) < share_price_key(contract.limit_down):
        raise ValueError("ordinary-share limit range invalid")
    return contract


def run(profile_path: Path, contract_path: Path, output: Path):
    profile, frozen = validate_frozen_plan(profile_path, contract_path)
    if output.exists():
        raise ValueError("fresh output path required; never replace a prior support receipt")
    summaries, details = [], []
    for cell in profile["cells"]:
        # No fallback path, symbol or day is ever selected.
        if any(not Path(cell[key]["path"]).is_file() for key in ("raw", "basic_info", "parent_origins", "expected_native_keys")):
            summaries.append({"day": cell["day"], "symbol": cell["symbol"], "status": "missing_skip_no_replacement"})
            continue
        contract = read_contract(cell)
        if contract is None:
            summaries.append({"day": cell["day"], "symbol": cell["symbol"], "status": "missing_contract_skip_no_replacement"})
            continue
        origins = read_origins(cell)
        state, snapshots = replay(raw.events(Path(cell["raw"]["path"]), cell["symbol"]), origins, contract)
        summaries.append(cell_summary(cell["day"], cell["symbol"], state, snapshots))
        details.append({"day": cell["day"], "symbol": cell["symbol"], "snapshots": snapshots})
    # Stat-keyed hash caching avoids repeated bytes when sources are unchanged;
    # a producer/input/profile edit during streaming rejects before publication.
    validate_frozen_plan(profile_path, contract_path)
    result = {
        "schema": "h13-acceptance-support-proxy-v1",
        "method_identity": frozen["identity"],
        "profile_identity": digest(profile),
        "method": {"path": str(contract_path.resolve()), "sha256": _hash(contract_path)},
        "method_anchor": {
            "path": str(contract_path.with_suffix(contract_path.suffix + ".identity").resolve()),
            "sha256": _hash(contract_path.with_suffix(contract_path.suffix + ".identity")),
        },
        "inputs": frozen["inputs"],
        "sources": frozen["sources"],
        "cells": summaries,
        "support_gate": evaluate_gates(summaries),
        "native_support_proved": False,
        "native_prefix_and_parity_required_before_fe": True,
        "labels_read": False,
        "model_fits": 0,
        "no_predictive_admission": True,
    }
    output.mkdir(parents=True)
    write_yaml(output / "support-receipt.yaml", result)
    write_yaml(output / "origin-snapshots.yaml", details)
    return result
