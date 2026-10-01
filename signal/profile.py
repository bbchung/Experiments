"""Signal profile loading and the fixed date plan."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = ("name", "paths", "universe", "dates", "session", "target", "features", "model", "evaluation")


class SignalError(RuntimeError):
    pass


def load(path) -> dict:
    path = Path(path)
    profile: dict = dict(yaml.safe_load(path.read_text()) or {})
    missing = [k for k in REQUIRED if k not in profile]
    if missing:
        raise SignalError(f"signal profile {path} lacks {missing}")
    profile = copy.deepcopy(profile)
    base = path.resolve().parent
    for key, value in profile["paths"].items():
        if isinstance(value, str) and value and not os.path.isabs(value):
            profile["paths"][key] = str((base / value).resolve())
    target = profile["target"]
    if target.get("kind") != "mid_ticks" or not target.get("horizons") or not target.get("threshold_ticks"):
        raise SignalError("signal target must be mid_ticks with horizons and threshold_ticks")
    return profile


def work(profile) -> Path:
    path = Path(profile["paths"]["work"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def calendar_days(profile) -> list[str]:
    cal: dict = dict(yaml.safe_load(Path(profile["paths"]["calendar"]).read_text()) or {})
    closed = {str(d) for d in cal.get("unscheduled_closures") or []}
    return sorted(str(d) for d in cal["trading_days"] if str(d) not in closed)


def observed_days(profile) -> list[str]:
    """Trading days with a contract file and market data for a reference symbol.

    Missing days are skipped, never filled; the split is laid over what exists.
    """
    reference = profile["dates"].get("reference_symbol", "2330")
    market = Path(profile["paths"]["market_data"]) / reference
    contracts = Path(profile["paths"]["contracts"])
    have_md = {p.name[:8] for p in market.iterdir()} if market.is_dir() else set()
    have_contract = {p.name[:8] for p in contracts.iterdir()}
    first = str(profile["dates"].get("first", "00000000"))
    last = str(profile["dates"]["last"])
    return [d for d in calendar_days(profile) if first <= d <= last and d in have_md and d in have_contract]


def date_plan(profile) -> dict:
    """Chronological roles, oldest first: screen | train | purge | es | purge | test || sealed."""
    cfg = profile["dates"]
    days = observed_days(profile)
    sealed_n, dev_n = int(cfg["sealed_days"]), int(cfg["dev_days"])
    screen_cfg = profile["universe"].get("screen", {})
    screen_n = int(screen_cfg.get("days", 0))
    if len(days) < sealed_n + dev_n:
        raise SignalError(f"only {len(days)} observed days; the plan needs {sealed_n + dev_n}")
    sealed = days[len(days) - sealed_n :] if sealed_n else []
    dev = days[len(days) - sealed_n - dev_n : len(days) - sealed_n]
    purge, train_n, es_n = int(cfg["purge_days"]), int(cfg["train_days"]), int(cfg["early_stopping_days"])
    plan = {
        # An explicit screen window reproduces a frozen selection; it must precede dev.
        "screen": [d for d in calendar_days(profile) if str(screen_cfg["window"][0]) <= d <= str(screen_cfg["window"][1])]
        if "window" in screen_cfg
        else days[max(0, len(days) - sealed_n - dev_n - screen_n) : len(days) - sealed_n - dev_n],
        "dev": dev,
        "train": dev[:train_n],
        "es": dev[train_n + purge : train_n + purge + es_n],
        "test": dev[train_n + purge + es_n + purge :],
        "sealed": sealed,
    }
    if not plan["test"]:
        raise SignalError("the development window leaves no test days")
    if plan["screen"] and plan["screen"][-1] >= dev[0]:
        raise SignalError("the universe screen must end before the development window")
    return plan
