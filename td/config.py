"""Explicit, bounded protocol. No target-dependent feature or sample selection."""

from __future__ import annotations

import copy
import datetime as dt
import math

from ...io import ContractError, digest

DEFAULTS = {
    "version": 1,
    "truth": {"horizons_seconds": [30, 60, 120], "up_ticks": [1.0, 2.0], "down_ticks": [1.0, 2.0]},
    "targets": {"families": ["fixed_return", "threshold", "triple_barrier", "mfe", "mae", "excursion", "first_passage"], "threshold_ticks": [1.0, 2.0], "lambdas": [0.5, 1.0]},
    "proxy": {"iterations": 80, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "seed": 42},
    "discovery": {"enabled": True, "sample_fraction": 0.05, "max_rows_per_partition": 100, "iterations": 8, "depth": 3, "max_promoted": 3},
    "splits": {
        "outer_train_days": 35,
        "outer_test_days": 5,
        "outer_folds": 3,
        "inner_train_days": 10,
        "calibration_days": 4,
        "validation_days": 4,
        "inner_folds": 3,
        "purge_days": 1,
    },
    "economics": {"horizon_seconds": 60, "round_trip_ticks": 2.0, "margin_ticks": 0.0, "bins": 5, "equivalence_ticks": 0.01},
    "gates": {
        "min_rows": 100,
        "min_class_count": 10,
        "min_triggers": 30,
        "min_days": 3,
        "min_symbols": 2,
        "min_coverage": 0.001,
        "positive_fold_fraction": 0.67,
        "positive_day_fraction": 0.5,
        "positive_symbol_fraction": 0.5,
        "max_symbol_share": 0.7,
        "null_alpha": 0.1,
    },
    "budget": {"max_candidates": 128, "max_fits": 20000, "null_repeats": 19, "shortlist": 3},
}


def protocol(document):
    cfg = copy.deepcopy(DEFAULTS)
    if not isinstance(document, dict) or set(document) - {*cfg, "data"}:
        raise ContractError("Unknown TD configuration section")
    for key, value in document.items():
        if key == "data":
            cfg[key] = copy.deepcopy(value)
        elif isinstance(cfg[key], dict):
            if not isinstance(value, dict) or set(value) - set(cfg[key]):
                raise ContractError(f"Unknown TD {key} setting")
            cfg[key].update(value)
        else:
            cfg[key] = value
    if cfg["version"] != 1:
        raise ContractError("Unsupported TD version")
    discovery = cfg["discovery"]
    if type(discovery["enabled"]) is not bool:
        raise ContractError("TD discovery.enabled must be boolean")
    if type(discovery["sample_fraction"]) not in (int, float) or not 0 < discovery["sample_fraction"] <= 1:
        raise ContractError("TD discovery.sample_fraction must be in (0, 1]")
    for key in ["max_rows_per_partition", "iterations", "depth", "max_promoted"]:
        if type(discovery[key]) is not int or discovery[key] < 1:
            raise ContractError(f"TD discovery.{key} must be a positive integer")
    if discovery["iterations"] % 2 or discovery["depth"] > 10:
        raise ContractError("TD discovery needs an even tree budget and depth <=10")
    for group in ["proxy", "splits", "economics", "gates", "budget"]:
        for key, value in cfg[group].items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ContractError(f"Invalid TD {group}.{key}")
    for group, keys in {
        "proxy": ["iterations", "depth", "seed"],
        "splits": list(cfg["splits"]),
        "economics": ["horizon_seconds", "bins"],
        "gates": ["min_rows", "min_class_count", "min_triggers", "min_days", "min_symbols"],
        "budget": list(cfg["budget"]),
    }.items():
        if any(type(cfg[group][key]) is not int or cfg[group][key] < (0 if key == "seed" else 1) for key in keys):
            raise ContractError(f"TD {group} counts must be positive integers")
    if cfg["splits"]["inner_folds"] < 2 or cfg["splits"]["outer_folds"] < 2 or cfg["proxy"]["iterations"] % 2:
        raise ContractError("TD needs >=2 inner/outer folds and an even per-candidate tree budget")
    if cfg["proxy"]["learning_rate"] <= 0 or cfg["proxy"]["depth"] > 10:
        raise ContractError("Invalid bounded proxy learner")
    if discovery["enabled"] and (discovery["iterations"] > cfg["proxy"]["iterations"] or discovery["depth"] > cfg["proxy"]["depth"]):
        raise ContractError("TD screening cannot exceed the confirmation model budget")
    for key in ["min_coverage", "positive_fold_fraction", "positive_day_fraction", "positive_symbol_fraction", "max_symbol_share", "null_alpha"]:
        if not 0 < cfg["gates"][key] <= 1:
            raise ContractError(f"Invalid TD fraction {key}")
    for section, key in [("truth", "horizons_seconds"), ("truth", "up_ticks"), ("truth", "down_ticks"), ("targets", "threshold_ticks"), ("targets", "lambdas")]:
        grid = cfg[section][key]
        if not isinstance(grid, list) or not grid or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in grid) or len(set(grid)) != len(grid):
            raise ContractError(f"Invalid TD grid {section}.{key}")
        cfg[section][key] = sorted(grid)
    if any(type(h) is not int or h > 86400 for h in cfg["truth"]["horizons_seconds"]) or any(
        v > 1000000 or v != int(v) for v in [*cfg["truth"]["up_ticks"], *cfg["truth"]["down_ticks"]]
    ):
        raise ContractError("Truth horizons must be integer seconds <=86400 and barriers positive integer ticks <=1000000")
    if cfg["economics"]["horizon_seconds"] not in cfg["truth"]["horizons_seconds"]:
        raise ContractError("Common economic horizon must be in the truth grid")
    families = cfg["targets"]["families"]
    if not families or len(set(families)) != len(families) or set(families) - set(DEFAULTS["targets"]["families"]):
        raise ContractError("Unknown or duplicate target families; arbitrary formulas are not supported")
    return cfg


def validate_data(data):
    required = {"profile", "dates", "symbols", "modules", "features", "sample_interval_seconds"}
    optional = {"otc_info_root", "otc_symbols", "excluded_partitions", "native_symbols_per_job"}
    if not isinstance(data, dict) or not required <= set(data) or set(data) - required - optional:
        raise ContractError(f"TD data requires {sorted(required)}; optional {sorted(optional)}")
    for key in ["dates", "symbols", "features"]:
        values = data[key]
        if not values or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
            raise ContractError(f"TD data.{key} must be unique nonempty strings")
    if data["dates"] != sorted(data["dates"]) or any(dt.date.fromisoformat(d).strftime("%Y%m%d") != d for d in data["dates"]):
        raise ContractError("TD dates must be chronological YYYYMMDD")
    if type(data["sample_interval_seconds"]) is not int or data["sample_interval_seconds"] <= 0:
        raise ContractError("TD sample interval must be positive integer seconds")
    if not data["modules"] or any(f.startswith("truth.") or len(f.split(".")) != 4 for f in data["features"]):
        raise ContractError("TD requires explicit native feature descriptors, never truth/metadata columns")
    if "otc_info_root" in data and (not isinstance(data["otc_info_root"], str) or not data["otc_info_root"].strip()):
        raise ContractError("TD otc_info_root must be a nonempty path")
    if ("otc_info_root" in data) != ("otc_symbols" in data):
        raise ContractError("TD OTC metadata requires an explicit otc_symbols list")
    if "otc_symbols" in data:
        symbols = data["otc_symbols"]
        if not isinstance(symbols, list) or not symbols or any(s not in data["symbols"] for s in symbols) or len(set(symbols)) != len(symbols):
            raise ContractError("TD otc_symbols must be a unique subset of the fixed panel")
    size = data.get("native_symbols_per_job", len(data["symbols"]))
    if type(size) is not int or size <= 0:
        raise ContractError("TD native_symbols_per_job must be a positive integer")
    excluded = data.get("excluded_partitions", [])
    if not isinstance(excluded, list):
        raise ContractError("TD excluded_partitions must be a list")
    seen = set()
    for row in excluded:
        if (
            not isinstance(row, dict)
            or set(row) != {"day", "symbol", "reason"}
            or row["day"] not in data["dates"]
            or row["symbol"] not in data["symbols"]
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
        ):
            raise ContractError("TD exclusions must name configured day/symbol partitions and an explicit reason")
        key = (row["day"], row["symbol"])
        if key in seen:
            raise ContractError("Duplicate TD excluded partition")
        seen.add(key)
    if any(all((day, symbol) in seen for symbol in data["symbols"]) for day in data["dates"]):
        raise ContractError("TD exclusion removes an entire configured date; revise the protocol explicitly")
    return digest({"features": data["features"], "modules": data["modules"]})
