"""Study profile: paths, universe rule, chronological roles and the frozen judge.

Every scientific choice lives in the versioned profile. Roles are derived from
the observed trading calendar: a date without the reference market file is
skipped, never filled, and never shifts a role boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from ...io import ContractError, read_yaml

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("schema", "study", "data_floor", "paths", "universe", "dates", "session", "labels", "training", "judge")
ROLE_NAMES = ("dev_train", "dev_es", "dev_val", "final_train", "final_es", "oos")


def load(path) -> dict:
    path = Path(path).resolve()
    profile = read_yaml(path)
    if not isinstance(profile, dict) or profile.get("schema") != "astra-fe-v2":
        raise ContractError(f"{path} is not an astra-fe-v2 profile")
    missing = [key for key in REQUIRED if key not in profile]
    if missing:
        raise ContractError(f"FE profile {path} lacks {missing}")
    profile = copy.deepcopy(profile)
    profile["_path"] = str(path)
    for key, value in profile["paths"].items():
        if isinstance(value, str) and value and not os.path.isabs(value):
            profile["paths"][key] = str((path.parent / value).resolve())
    floor = str(profile["data_floor"])
    for key, value in profile["dates"].items():
        if isinstance(value, str) and value.isdigit() and value < floor:
            raise ContractError(f"dates.{key}={value} precedes the data floor {floor}")
    window = profile["universe"]["screen_window"]
    if str(window[0]) < floor:
        raise ContractError("The universe screen precedes the data floor")
    return profile


def work(profile) -> Path:
    path = Path(profile["paths"]["work"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def calendar_days(profile) -> list[str]:
    cal = read_yaml(Path(profile["paths"]["calendar"]))
    closed = {str(d) for d in cal.get("unscheduled_closures") or []}
    return sorted(str(d) for d in cal["trading_days"] if str(d) not in closed)


def observed_days(profile, first: str, last: str) -> list[str]:
    """Scheduled trading days in [first, last] with a contract table and reference market data."""
    reference = Path(profile["paths"]["market_data"]) / profile["dates"]["reference_symbol"]
    contracts = Path(profile["paths"]["contracts"])
    have_md = {p.name[:8] for p in reference.iterdir()} if reference.is_dir() else set()
    have_contract = {p.name[:8] for p in contracts.iterdir()}
    return [d for d in calendar_days(profile) if first <= d <= last and d in have_md and d in have_contract]


def roles(profile) -> dict[str, list[str]]:
    """Chronological roles from explicit date boundaries (inclusive)."""
    cfg = profile["dates"]
    days = observed_days(profile, str(cfg["first"]), str(cfg["last"]))
    out = {}
    for name in ROLE_NAMES:
        lo, hi = (str(x) for x in cfg[name])
        out[name] = [d for d in days if lo <= d <= hi]
        if not out[name]:
            raise ContractError(f"Role {name} [{lo}, {hi}] has no observed trading day")
    if max(out["dev_val"]) >= min(out["oos"]) or max(out["final_es"]) >= min(out["oos"]):
        raise ContractError("Development and final fitting roles must precede the OOS role")
    if max(out["dev_train"]) >= min(out["dev_es"]) or max(out["dev_es"]) >= min(out["dev_val"]):
        raise ContractError("Development roles must be chronological")
    if max(out["final_train"]) >= min(out["final_es"]):
        raise ContractError("Final roles must be chronological")
    return out


def universe(profile) -> dict:
    path = work(profile) / "universe" / "universe.yaml"
    if not path.is_file():
        raise ContractError("Run the universe screen before using symbols")
    return read_yaml(path)


def symbols(profile, group="all") -> list[str]:
    """all | development | holdout | dev_a | dev_b (a fixed hash split of development)."""
    u = universe(profile)
    if group == "all":
        return sorted(u["development"] + u["holdout"])
    if group in ("dev_a", "dev_b"):
        from .screen import split_holdout

        a, b = split_holdout(u["development"], 0.5, f"{profile['universe']['holdout']['salt']}:devx")
        return a if group == "dev_a" else b
    return list(u[group])


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def section_digest(profile, keys) -> str:
    return hashlib.sha256(canonical({k: profile[k] for k in keys})).hexdigest()


def dump(value) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
