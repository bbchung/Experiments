"""Frozen scientific inputs and paired populations for representation research.

This module does not train or select features. Call ``verify`` immediately before
each fit, and compare population signatures before accepting an arm's evidence.
Different experiment stores share the same GPU lock.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import numbers
import operator
import os
import struct
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ... import io
from ...io import ContractError, atomic_bytes, digest, file_hash, lock, read_yaml, write_yaml

SCHEMA = "astra-representation-v1"
GLOBAL_GPU_LOCK = Path("/home/bb/workspace/coco_dev/AstraResearch/runs/.locks/catboost-gpu0.lock")
ROLES = ("train", "es", "calibration", "development")
_PROFILE_KEYS = {"_profile_path", "_profile_sha256"}


def _date(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        raise ContractError(f"{name} must be a quoted YYYYMMDD date")
    try:
        datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as error:
        raise ContractError(f"Invalid {name}: {value}") from error
    return value


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a mapping")
    return value


def _days(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ContractError(f"{name} must be an explicit {'nonempty ' if nonempty else ''}date list")
    days = [_date(day, name) for day in value]
    if days != sorted(set(days)):
        raise ContractError(f"{name} dates must be unique and chronological")
    return days


def scientific_profile(profile: dict) -> dict:
    """Return the complete frozen document without loader bookkeeping."""
    return {key: copy.deepcopy(value) for key, value in profile.items() if key not in _PROFILE_KEYS}


def validate_profile(profile: dict) -> None:
    p = _mapping(profile, "profile")
    if p.get("schema") != SCHEMA or not isinstance(p.get("id"), str) or not p["id"]:
        raise ContractError(f"Profile requires schema {SCHEMA} and an id")
    floor = _date(p.get("data_floor"), "data_floor")
    as_of = _date(p.get("freeze_as_of"), "freeze_as_of")
    if floor < "20260101":
        raise ContractError("Research data_floor must be >= 20260101")
    horizons = p.get("horizons")
    if (
        not isinstance(horizons, list)
        or any(type(h) is not int or h <= 0 for h in horizons)
        or horizons != sorted(set(horizons))
        or not {60, 120, 180, 300}.issubset(horizons)
        or any(h < 300 and h not in {60, 120, 180} for h in horizons)
        or p.get("primary_horizon") != 300
    ):
        raise ContractError("Horizons require chronological 60/120/180/300, optional longer horizons, and primary_horizon 300")
    label = _mapping(p.get("label"), "label")
    required_label = {"kind": "native_mid_tick_endpoint", "threshold_ticks": 5, "inclusive": True, "tail_support": "observed_endpoint", "trading_aware": False}
    if any(label.get(key) != value or type(label.get(key)) is not type(value) for key, value in required_label.items()):
        raise ContractError("Labels must be native observed endpoint mid-tick moves with inclusive +/- 5 tick events and pure-signal semantics")
    sampling = _mapping(p.get("sampling"), "sampling")
    interval = sampling.get("interval_seconds")
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ContractError("Sampling interval_seconds must be positive and finite")
    if sampling.get("shared_rows") is not True or sampling.get("complete_horizons") is not True:
        raise ContractError("This contract requires shared rows with complete endpoint support at every horizon")
    splits = _mapping(p.get("splits"), "splits")
    if set(splits) != set(ROLES):
        raise ContractError(f"Splits must define exactly {ROLES}")
    previous = floor
    all_days = []
    for role in ROLES:
        days = _days(splits[role], f"splits.{role}")
        if days[0] < floor or (all_days and days[0] <= previous):
            raise ContractError("Split roles must be disjoint and chronologically train < es < calibration < development")
        previous = days[-1]
        all_days.extend(days)
    exposure = _mapping(p.get("development_exposure"), "development_exposure")
    observed = _date(exposure.get("observed_through"), "development_exposure.observed_through")
    exposed = _days(exposure.get("days"), "development_exposure.days")
    if exposure.get("purpose") != "development_only" or not set(all_days).issubset(exposed) or exposed[-1] > observed or observed > as_of:
        raise ContractError("Development exposure must declare every split day and cannot claim final OOS")
    sealed = _mapping(p.get("sealed_oos"), "sealed_oos")
    not_before = _date(sealed.get("not_before"), "sealed_oos.not_before")
    future = _days(sealed.get("days"), "sealed_oos.days", nonempty=False)
    if sealed.get("mode") != "future_only" or not_before <= max(as_of, observed) or any(day < not_before for day in future):
        raise ContractError("Sealed OOS must reserve new future dates after freeze_as_of and all development exposure")
    selector = _mapping(p.get("selector"), "selector")
    if selector.get("fit_role") != "train" or selector.get("use_oos") is not False:
        raise ContractError("Feature and symbol eligibility selectors must fit on train only")
    training = _mapping(p.get("training"), "training")
    params = _mapping(training.get("params"), "training.params")
    if training.get("model") != "CatBoost" or training.get("es_role") != "es" or training.get("calibration_role") != "calibration":
        raise ContractError("Training requires fixed CatBoost and separate es/calibration roles")
    required_params = {"task_type", "devices", "loss_function", "iterations", "depth", "learning_rate", "random_seed", "thread_count"}
    if not required_params.issubset(params) or params["task_type"] != "GPU" or str(params["devices"]) != "0":
        raise ContractError("Fixed CatBoost GPU0 parameters must be explicit")
    for key in ("iterations", "depth", "thread_count"):
        if type(params[key]) is not int or params[key] <= 0:
            raise ContractError(f"training.params.{key} must be a positive integer")
    if type(params["random_seed"]) is not int or not isinstance(params["loss_function"], str) or not params["loss_function"]:
        raise ContractError("CatBoost random_seed and loss_function must be fixed")
    rate = params["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ContractError("CatBoost learning_rate must be positive and finite")
    for field in ("metrics", "acceptance"):
        if not _mapping(p.get(field), field):
            raise ContractError(f"{field} must be explicit and nonempty before feature comparison")


def load_profile(path: Path | str) -> dict:
    path = Path(path).resolve()
    before = file_hash(path)
    profile = read_yaml(path)
    validate_profile(profile)
    if _PROFILE_KEYS.intersection(profile):
        raise ContractError("Reserved profile loader fields may not appear in YAML")
    if file_hash(path) != before:
        raise ContractError("Profile changed while loading")
    return {**profile, "_profile_path": str(path), "_profile_sha256": before}


def _files(paths: Sequence[Path | str] | Mapping[str, Path | str]) -> list[dict]:
    pairs = paths.items() if isinstance(paths, Mapping) else ((str(i), path) for i, path in enumerate(paths))
    records = []
    for name, filename in pairs:
        path = Path(filename).resolve()
        if not path.is_file():
            raise ContractError(f"Frozen source/input must exist: {path}")
        records.append({"name": str(name), "path": str(path), "sha256": file_hash(path), "size": path.stat().st_size})
    return sorted(records, key=lambda entry: (entry["name"], entry["path"]))


def freeze(profile: dict, path: Path | str, sources: Sequence[Path | str], inputs: Sequence[Path | str] | Mapping[str, Path | str]) -> dict:
    """Create once, or verify an exactly identical resume; never rewrite a freeze."""
    path = Path(path).resolve()
    validate_profile(profile)
    source_path = Path(profile.get("_profile_path", ""))
    if not source_path.is_file() or file_hash(source_path) != profile.get("_profile_sha256"):
        raise ContractError("Freeze requires an unchanged load_profile document")
    if scientific_profile(profile) != read_yaml(source_path):
        raise ContractError("In-memory profile differs from the profile document")
    if not sources or not inputs:
        raise ContractError("Freeze requires explicit scientific sources and input artifacts")
    records = _files({str(Path(p).resolve()): p for p in [Path(__file__), Path(io.__file__), *sources]})
    document = {
        "schema": SCHEMA,
        "profile": scientific_profile(profile),
        "profile_source": _files({"profile": source_path})[0],
        "sources": records,
        "inputs": _files(inputs),
        "gpu_lock": str(GLOBAL_GPU_LOCK.resolve()),
    }
    document["identity"] = digest(document)
    with lock(path.with_name(path.name + ".lock")):
        if path.exists():
            existing = verify(path)
            if existing != document:
                raise ContractError("Resume differs from the immutable scientific freeze; create a new comparison")
            return existing
        if profile["freeze_as_of"] != datetime.now(UTC).strftime("%Y%m%d"):
            raise ContractError("A new freeze must declare today's actual freeze_as_of date")
        anchor = path.with_name(path.name + ".identity")
        if anchor.exists():
            raise ContractError("A scientific freeze identity already exists; do not recreate or replace a missing manifest")
        write_yaml(path, document)
        atomic_bytes(anchor, (document["identity"] + "\n").encode("ascii"))
        return verify(path)


def verify(path: Path | str) -> dict:
    """Reject profile, scientific source, input identity, or manifest drift."""
    document = read_yaml(Path(path))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ContractError("Invalid representation freeze schema")
    body = {key: value for key, value in document.items() if key != "identity"}
    if document.get("identity") != digest(body):
        raise ContractError("Frozen manifest identity changed")
    anchor = Path(path).with_name(Path(path).name + ".identity")
    if not anchor.is_file() or anchor.read_text(encoding="ascii").strip() != document["identity"]:
        raise ContractError("Frozen manifest differs from its immutable identity anchor")
    validate_profile(document.get("profile"))
    if document.get("gpu_lock") != str(GLOBAL_GPU_LOCK.resolve()):
        raise ContractError("Frozen GPU lock must use the global GPU0 path")
    try:
        for record in [document["profile_source"], *document["sources"], *document["inputs"]]:
            source = Path(record["path"])
            if not source.is_file() or source.stat().st_size != record["size"] or file_hash(source) != record["sha256"]:
                raise ContractError(f"Frozen source/input changed: {source}")
        if read_yaml(Path(document["profile_source"]["path"])) != document["profile"]:
            raise ContractError("Frozen profile and document differ")
    except (KeyError, TypeError) as error:
        raise ContractError("Malformed frozen source/input records") from error
    return document


def assert_train_only(profile: dict, days: Iterable[str], *, purpose: str = "selector") -> None:
    days = list(days)
    if not days or not set(days).issubset(profile["splits"]["train"]):
        raise ContractError(f"{purpose} fitting may use train dates only")


def comparison_signature(profile: dict, row_keys: Iterable[Sequence], labels: Mapping[int, Iterable], roles: Iterable[str]) -> dict:
    """Hash exact ordered keys, roles and finite endpoint labels for every arm.

    Keys are (day, symbol, SampleTime) or the full native observation identity
    (day, symbol, SampleTime, SampleBookTime, SampleBookSeq), sorted
    lexicographically. Every time/sequence must be an exact integer; book state
    cannot occur after sampling. Labels contain native endpoint delta-mid ticks,
    not trading outcomes or bps.
    """
    validate_profile(profile)
    horizons = profile["horizons"]
    if set(labels) != set(horizons):
        raise ContractError("Comparison labels must cover exactly the frozen horizons")
    iterators = [iter(labels[h]) for h in horizons]
    row_hasher = hashlib.sha256()
    label_hashers = {h: hashlib.sha256() for h in horizons}
    counts, last = Counter(), None
    role_days = {day: role for role, days in profile["splits"].items() for day in days}
    sentinel = object()
    for key, role, *values in itertools.zip_longest(row_keys, roles, *iterators, fillvalue=sentinel):
        if any(value is sentinel for value in (key, role, *values)):
            raise ContractError("Comparison key/role/label lengths differ")
        if not isinstance(key, Sequence) or len(key) not in (3, 5):
            raise ContractError("Comparison row keys must be (day, symbol, SampleTime[, SampleBookTime, SampleBookSeq])")
        day, symbol, *observation = key
        day = _date(day, "row day")
        if not isinstance(symbol, str) or not symbol:
            raise ContractError("Row symbols must be nonempty strings")
        try:
            if any(isinstance(value, bool) for value in observation):
                raise TypeError
            observation = [operator.index(value) for value in observation]
        except TypeError as error:
            raise ContractError("Sample and book time/sequence must be exact integers") from error
        if len(observation) == 3 and (observation[1] > observation[0] or observation[2] < 0):
            raise ContractError("Book time cannot exceed SampleTime and SampleBookSeq must be nonnegative")
        key = (day, symbol, *observation)
        if last is not None and key <= last:
            raise ContractError("Row keys must be unique and sorted chronologically by day/symbol/time")
        if role_days.get(day) != role:
            raise ContractError("Row role differs from the frozen chronological split")
        last = key
        counts[role] += 1
        row_hasher.update(json.dumps([*key, role], ensure_ascii=True, separators=(",", ":")).encode() + b"\n")
        for horizon, value in zip(horizons, values, strict=True):
            if isinstance(value, (bool, str)) or not isinstance(value, numbers.Real):
                raise ContractError("Endpoint labels must be numeric native mid-tick changes")
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ContractError("Endpoint labels must be finite native mid-tick changes") from error
            if not math.isfinite(value):
                raise ContractError("Endpoint labels require complete finite horizon support")
            label_hashers[horizon].update(struct.pack("!d", 0.0 if value == 0.0 else value))
    if not counts:
        raise ContractError("Cannot compare empty populations")
    signature = {
        "contract": digest(scientific_profile(profile)),
        "row_keys_roles_sha256": row_hasher.hexdigest(),
        "rows_by_role": dict(counts),
        "labels_sha256": {h: hasher.hexdigest() for h, hasher in label_hashers.items()},
    }
    return {**signature, "identity": digest(signature)}


def assert_same_comparison(baseline: dict, candidate: dict) -> None:
    for signature in (baseline, candidate):
        if signature.get("identity") != digest({key: value for key, value in signature.items() if key != "identity"}):
            raise ContractError("Invalid comparison signature identity")
    if baseline != candidate:
        raise ContractError("Baseline and candidate must have identical frozen rows, roles, labels and evaluation procedure")


def _check_device() -> None:
    try:
        result = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise ContractError(f"Cannot verify GPU0 device ownership: {error}") from error
    pids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(pid != str(os.getpid()) for pid in pids):
        raise ContractError(f"GPU0 is occupied by another compute process: {', '.join(pids)}")


@contextmanager
def gpu_lock(path: Path | str | None = None, *, blocking: bool = True, check_device: bool = False) -> Iterator[None]:
    """Always lock GLOBAL_GPU_LOCK; optional path is the freeze to verify.

    ``path`` never chooses a store-local lock, so independent runs serialize.
    Every CatBoost fit (including CPU-only test runs when used) follows this guard.
    """
    with lock(GLOBAL_GPU_LOCK, blocking=blocking):
        if path is not None:
            verify(path)
        if check_device:
            _check_device()
        yield
