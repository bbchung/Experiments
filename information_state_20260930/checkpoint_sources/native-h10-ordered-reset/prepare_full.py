"""Prepare the fixed 161-day native H10 producer; never replay or train."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
PROFILE = OUTPUT.parent / "v2/profile.yaml"
CALENDAR = ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"
RAW = Path("/mnt/data0/marketdata/tse/kgi/stock")
BASIC = Path("/mnt/data0/contract/tse/stock")
FIELDS = (
    "direction",
    "log_work_strength",
    "signed_after_containment_share",
    "signed_after_repair_share",
    "delayed_repair_fraction",
    "signed_renewed_work",
    "mid_progress_5ticks",
    "phase",
)
DIAGNOSTICS = (
    "processed_cluster_exchange_time",
    "processed_cluster_available_time",
    "anchor_price",
    "renewed_price",
    "cumulative_complete_chain_count",
    "cumulative_buy_complete_chain_count",
    "cumulative_sell_complete_chain_count",
)


def read(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record(path):
    return {"path": str(path), "sha256": digest(path), "size": path.stat().st_size}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def full_config(original):
    target = next(group for group in original["Modules"] if any(item["Desc"] == "DatasetWriter.0" for item in group["Decl"]))
    target_items = {item["Desc"]: item for item in target["Decl"]}
    source_writer = target_items["DatasetWriter.0"]["Spec"]
    source_filter = target_items["TwseFilter.0"]
    assert source_writer["Subscribe"] == [{"Book": [target["Gid"]]}]
    assert source_filter["Spec"]["Subscribe"] == [{"Book": [target["Gid"]], "Trade": [target["Gid"]]}]
    assert source_filter["Spec"]["RequireTradable"] is False
    assert source_writer["PeriodicSampler"] == {"StartTime": "091000", "UntilTime": "130000", "SampleInterval": "10s"}
    groups = []
    for group in original["Modules"]:
        if not group["Gid"]:
            groups.append(copy.deepcopy(group))
            continue
        symbol = group["Gid"]
        chosen = {item["Desc"]: item for item in group["Decl"]}
        filter_module = copy.deepcopy(source_filter)
        filter_module["Spec"]["Subscribe"] = [{"Book": [symbol], "Trade": [symbol]}]
        metadata = [
            {"Feature": f"CurrentBook.0.{field}.0@{symbol}", "Name": name}
            for field, name in (
                ("book_mid_price", "OriginMidPrice"),
                ("book_mid_ticks", "OriginMidTicks"),
                ("book_bid_ticks", "OriginBidTicks"),
                ("book_ask_ticks", "OriginAskTicks"),
            )
        ]
        metadata += [
            {"Feature": f"DemandRepairRenewal.{i}.{field}.0@{symbol}", "Name": f"H10_{variant}_{field}"} for i, variant in ((0, "ordered"), (1, "reset")) for field in DIAGNOSTICS
        ]
        modules = [filter_module, copy.deepcopy(chosen["CurrentBook.0"])]
        modules += [
            {
                "Desc": f"DemandRepairRenewal.{i}",
                "Spec": {
                    "MaxBookAge": "5s",
                    "MaxDemandGap": "30s",
                    "ResetAtTouchRenewal": bool(i),
                    "Dep": {"Book": [f"TwseFilter.0@{symbol}"], "Trade": [f"TwseFilter.0@{symbol}"]},
                },
            }
            for i in (0, 1)
        ]
        modules += [
            {
                "Desc": "DatasetWriter.0",
                "Spec": {
                    "Subscribe": [{"Book": [symbol]}],
                    "Exports": [f"DemandRepairRenewal.{i}.{field}.0@{symbol}" for i in (0, 1) for field in FIELDS],
                    "MetadataExports": metadata,
                    "PeriodicSampler": copy.deepcopy(source_writer["PeriodicSampler"]),
                    "OutputPath": "${cwd}/data/${trading_date}/" + symbol + "/values.parquet",
                    "Format": "parquet",
                    "UseTmp": True,
                    "EmitSampleContext": True,
                    "RequireAlphaFactorExports": True,
                },
            }
        ]
        groups.append({"Gid": symbol, "Decl": modules})
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def main():
    destination = OUTPUT / "full-preparation-receipt.yaml"
    if destination.exists() or (OUTPUT / "full").exists():
        raise RuntimeError("Fresh full producer root and receipt required; no overwrite or resume")
    started = time.monotonic()
    profile = read(PROFILE)
    days = sorted({str(day) for role in ("train", "es", "calibration", "development") for day in profile["splits"][role]})
    assert len(days) == 161 and min(days) >= "20260101"
    symbols = profile["universe"]["symbols"]
    assert len(symbols) == 16
    preparation = read(OUTPUT / "preparation-receipt.yaml")
    original_path = Path(preparation["pilot_jobs"][0]["original_config"])
    assert digest(original_path) == preparation["pilot_jobs"][0]["original_config_sha256"]
    original = read(original_path)
    assert [group["Gid"] for group in original["Modules"] if group["Gid"]] == symbols
    config = full_config(original)
    assert [group["Gid"] for group in config["Modules"]] == [group["Gid"] for group in original["Modules"]]
    source_build = read(OUTPUT / "producer-source-build.yaml")
    binary = Path(source_build["binary"])
    assert digest(binary) == source_build["binary_sha256"]
    runtime_path = OUTPUT / "runtime-snapshot.yaml"
    runtime = read(runtime_path)
    assert all(digest(Path(dep["path"])) == dep["sha256"] for dep in runtime["dependencies"])
    raw_inputs, basic_inputs, jobs, missing = [], [], [], []
    higher_priority_absent = []
    for day in days:
        basic = BASIC / (day + ".csv")
        if not basic.is_file():
            missing.append({"day": day, "scope": "whole_day", "reason": "missing_basic_info"})
            continue
        basic_inputs.append(record(basic))
        present = []
        for symbol in symbols:
            candidates = [RAW / symbol / (day + ext) for ext in (".bin.zst", ".bin", ".csv.zst", ".csv")]
            selected = next((path for path in candidates if path.is_file()), None)
            if selected is None:
                missing.append({"day": day, "symbol": symbol, "reason": "missing_raw_skip_no_backfill"})
                continue
            higher_priority_absent.extend(str(path) for path in candidates[: candidates.index(selected)])
            raw_inputs.append({"day": day, "symbol": symbol, **record(selected)})
            present.append(symbol)
        if not present:
            continue
        work = OUTPUT / "full" / day
        day_config = copy.deepcopy(config)
        for group in day_config["Modules"]:
            if group["Gid"] and group["Gid"] not in present:
                group["Decl"] = [item for item in group["Decl"] if item["Desc"] == "CurrentBook.0"]
        write(work / "config.yaml", day_config)
        jobs.append(
            {
                "day": day,
                "symbols": present,
                "work": str(work),
                "config": record(work / "config.yaml"),
                "command": [str(binary), "-d", day, "-C", str(work), "--trading-calendar", str(CALENDAR), "--run-status-dir", str(work / "status"), str(work / "config.yaml")],
                "environment": {"LD_LIBRARY_PATH": str(OUTPUT / "runtime"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                "outputs": [str(work / "data" / day / symbol / "values.parquet") for symbol in present],
            }
        )
    receipt = {
        "schema": "h10-native-full-preparation-v1",
        "script": record(Path(__file__)),
        "reference_profile": record(PROFILE),
        "calendar": record(CALENDAR),
        "original_template": record(original_path),
        "binary": record(binary),
        "source_build": record(OUTPUT / "producer-source-build.yaml"),
        "runtime_snapshot": record(runtime_path),
        "runtime_dependencies": runtime["dependencies"],
        "pilot_validation": record(OUTPUT / "independent-native-validation.yaml"),
        "jobs": jobs,
        "raw_inputs": raw_inputs,
        "basic_info_inputs": basic_inputs,
        "raw_higher_priority_paths_must_remain_absent": higher_priority_absent,
        "missing": missing,
        "fixed_role_days": days,
        "universe_order": symbols,
        "compressed_raw_bytes": sum(item["size"] for item in raw_inputs),
        "prepare_seconds": time.monotonic() - started,
        "native_replays_launched": 0,
        "model_fits": 0,
        "labels_read": False,
        "full_replay_authorized": False,
        "execution_requirements": [
            "Root requires independently validated frozen component provenance and passed native support before launch",
            "Run exactly one day job at a time using immutable binary and runtime; reject changed/added source precedence",
            "First Jan19 monolithic replay is also the full artifact benchmark; validate all16 parent keys/midbits before remaining days",
            "Existing outputs must never be overwritten; missing cells skip with no symbol/date replacement",
            "Full material contains only H10 pairs and metadata, not regenerated original labels or baseline features",
        ],
    }
    write(destination, receipt)
    print("PREPARED_ONLY", len(jobs), "days", len(raw_inputs), "raw inputs", receipt["compressed_raw_bytes"], "bytes", round(receipt["prepare_seconds"], 2), "seconds")


if __name__ == "__main__":
    main()
