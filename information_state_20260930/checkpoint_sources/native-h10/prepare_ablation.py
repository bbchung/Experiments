"""Prepare ordered/reset native pilots in one pass; never replay."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent.parent / "native-h10-ordered-reset"
STORE = ROOT / "AstraResearch/runs/fe-origin-store/objects"
PROFILE = ROOT / "AstraResearch/runs/information_state_20260930/v2/profile.yaml"
CALENDAR = ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"
PRESERVED = ROOT / "AstraResearch/runs/information_state_20260930/native-baseline/coco"
NEW_BINARY = ROOT / "build/Release/src/app/coco"
RAW = Path("/mnt/data0/marketdata/tse/kgi/stock")
KEYS = ["SampleTime", "SampleBookTime", "SampleBookSeq"]
ALPHAS = [
    "direction",
    "log_work_strength",
    "signed_after_containment_share",
    "signed_after_repair_share",
    "delayed_repair_fraction",
    "signed_renewed_work",
    "mid_progress_5ticks",
    "phase",
]
DIAGNOSTICS = [
    "processed_cluster_exchange_time",
    "processed_cluster_available_time",
    "anchor_price",
    "renewed_price",
    "cumulative_complete_chain_count",
    "cumulative_buy_complete_chain_count",
    "cumulative_sell_complete_chain_count",
]
EXPORT_IDENTITIES = {
    "20260119": "9c10e19edefa61c37bd7ba4c5c3d2d932eef9eb4a20815d4a1595d381b8bff1d",
    "20260120": "16036d233eb02c68622d5c9c54916e301c82161e1c0cd877f47e6eb729798d0b",
    "20260203": "348afede1668fc4c26d9afb0cbb6eafb0fef431a8924158236eafdb6da5832d5",
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))


def source(day, symbol):
    parent = STORE / EXPORT_IDENTITIES[day] / day / symbol
    return parent / "config.yaml", parent / "data" / day / symbol / "values.parquet"


def metadata(symbol):
    result = [
        {"Feature": f"CurrentBook.0.{field}.0@{symbol}", "Name": name}
        for field, name in [
            ("book_mid_price", "OriginMidPrice"),
            ("book_mid_ticks", "OriginMidTicks"),
            ("book_bid_ticks", "OriginBidTicks"),
            ("book_ask_ticks", "OriginAskTicks"),
        ]
    ]
    result += [
        {"Feature": f"DemandRepairRenewal.{module_id}.{field}.0@{symbol}", "Name": "H10_" + name + "_" + field}
        for module_id, name in [(0, "ordered"), (1, "reset")]
        for field in DIAGNOSTICS
    ]
    return result


def pilot_config(original, symbol):
    groups = []
    for group in original["Modules"]:
        if group["Gid"] == "":
            groups.append(copy.deepcopy(group))
            continue
        current_book = next(d for d in group["Decl"] if d["Desc"] == "CurrentBook.0")
        if group["Gid"] != symbol:
            groups.append({"Gid": group["Gid"], "Decl": [copy.deepcopy(current_book)]})
            continue
        filter_module = next(d for d in group["Decl"] if d["Desc"] == "TwseFilter.0")
        original_writer = next(d["Spec"] for d in group["Decl"] if d["Desc"] == "DatasetWriter.0")
        assert original_writer["PeriodicSampler"] == {"StartTime": "091000", "UntilTime": "130000", "SampleInterval": "10s"}
        writer = {
            "Subscribe": copy.deepcopy(original_writer["Subscribe"]),
            "Exports": [f"DemandRepairRenewal.{module_id}.{field}.0@{symbol}" for module_id in (0, 1) for field in ALPHAS],
            "MetadataExports": metadata(symbol),
            "PeriodicSampler": copy.deepcopy(original_writer["PeriodicSampler"]),
            "OutputPath": "${cwd}/data/${trading_date}/" + symbol + "/values.parquet",
            "Format": "parquet",
            "UseTmp": True,
            "EmitSampleContext": True,
            "RequireAlphaFactorExports": True,
        }
        event_writer = copy.deepcopy(writer)
        del event_writer["PeriodicSampler"]
        event_writer["BookFlipSampler"] = {"Mode": "all_book", "MinInterval": "0s", "StartTime": "091000", "UntilTime": "130000"}
        event_writer["OutputPath"] = "${cwd}/events/${trading_date}/" + symbol + "/values.parquet"
        groups.append(
            {
                "Gid": symbol,
                "Decl": [
                    copy.deepcopy(filter_module),
                    copy.deepcopy(current_book),
                    {
                        "Desc": "DemandRepairRenewal.0",
                        "Spec": {
                            "MaxBookAge": "5s",
                            "MaxDemandGap": "30s",
                            "ResetAtTouchRenewal": False,
                            "Dep": {"Book": [f"TwseFilter.0@{symbol}"], "Trade": [f"TwseFilter.0@{symbol}"]},
                        },
                    },
                    {
                        "Desc": "DemandRepairRenewal.1",
                        "Spec": {
                            "MaxBookAge": "5s",
                            "MaxDemandGap": "30s",
                            "ResetAtTouchRenewal": True,
                            "Dep": {"Book": [f"TwseFilter.0@{symbol}"], "Trade": [f"TwseFilter.0@{symbol}"]},
                        },
                    },
                    {"Desc": "DatasetWriter.0", "Spec": writer},
                    {"Desc": "DatasetWriter.1", "Spec": event_writer},
                ],
            }
        )
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def command(binary, day, work):
    return [str(binary), "-d", day, "-C", str(work), "--trading-calendar", str(CALENDAR), "--run-status-dir", str(work / "status"), str(work / "config.yaml")]


def main():
    destination = OUTPUT / "preparation-receipt.yaml"
    if destination.exists():
        raise RuntimeError("Native H10 preparation already exists; do not implicitly overwrite")
    profile = read(PROFILE)
    eligibility = read(OUTPUT.parent / "h10-pilot/preregistered-v1/eligibility-receipt.yaml")
    symbols = eligibility["selected_symbols"]
    assert symbols == ["2308", "2317"]
    binary_receipt = read(PRESERVED.parent / "receipt.yaml")
    assert digest(PRESERVED) == binary_receipt["sha256"]
    jobs = []
    for day in ("20260119", "20260120"):
        for symbol in symbols:
            original_config, native = source(day, symbol)
            original = read(original_config)
            work = OUTPUT / "pilot" / day / symbol
            write(work / "config.yaml", pilot_config(original, symbol))
            keys = pq.read_table(native, columns=KEYS, use_threads=False)
            assert len(keys) == 1381
            pq.write_table(keys, work / "expected-origin-keys.parquet", compression="zstd")
            jobs.append(
                {
                    "day": day,
                    "symbol": symbol,
                    "work": str(work),
                    "command_after_build": command(NEW_BINARY, day, work),
                    "config_sha256": digest(work / "config.yaml"),
                    "original_config": str(original_config),
                    "original_config_sha256": digest(original_config),
                    "original_native": str(native),
                    "original_native_stat_bytes": native.stat().st_size,
                    "expected_keys_sha256": digest(work / "expected-origin-keys.parquet"),
                    "expected_rows": len(keys),
                }
            )
    controls = []
    for symbol in ("2609", "2337"):
        day = "20260203"
        original_config, native = source(day, symbol)
        immutable_hash = digest(native)
        pair = {
            "day": day,
            "symbol": symbol,
            "original_config": str(original_config),
            "original_config_sha256": digest(original_config),
            "original_native": str(native),
            "original_native_sha256": immutable_hash,
        }
        for version, binary in (("preserved", PRESERVED), ("new", NEW_BINARY)):
            work = OUTPUT / "parity" / version / symbol
            work.mkdir(parents=True, exist_ok=True)
            (work / "config.yaml").write_bytes(original_config.read_bytes())
            pair[version] = {"work": str(work), "command": command(binary, day, work), "config_sha256": digest(work / "config.yaml")}
            assert pair[version]["config_sha256"] == pair["original_config_sha256"]
        controls.append(pair)
    days = sorted({day for dates in profile["splits"].values() for day in dates})
    sizes = []
    missing = []
    for day in days:
        for symbol in profile["universe"]["symbols"]:
            raw = RAW / symbol / (day + ".csv.zst")
            if raw.is_file():
                sizes.append(raw.stat().st_size)
            else:
                missing.append({"day": day, "symbol": symbol})
    write(
        destination,
        {
            "schema": "h10-native-ordered-reset-pilot-preparation-v2",
            "script_sha256": digest(Path(__file__)),
            "frozen_reference_profile": str(PROFILE),
            "frozen_reference_profile_sha256": digest(PROFILE),
            "calendar_sha256": digest(CALENDAR),
            "preserved_binary": str(PRESERVED),
            "preserved_binary_sha256": binary_receipt["sha256"],
            "parent_byte_parity_source_correspondence": "unproven_until_preserved_replay_matches_original_then_new_matches_preserved_and_original",
            "native_replays_launched": 0,
            "labels_read": False,
            "model_fits": 0,
            "pilot_jobs": jobs,
            "parity_pairs": controls,
            "full_feature_only_cost": {
                "source": "stat_only_exact_frozen_role_dates_not_raw_scan",
                "role_days": len(days),
                "universe_symbols": len(profile["universe"]["symbols"]),
                "available_raw_files": len(sizes),
                "missing": missing,
                "compressed_bytes_once_per_day_all_symbols": sum(sizes),
                "compressed_bytes_target_sharded_repeated_clock_universe": sum(sizes) * len(profile["universe"]["symbols"]),
                "recommended": "one_sequential_native_job_per_day_with_all16_ordered_and_reset_H10_instances_and_writers_read_raw_once_reuse_parent_labels_baseline",
                "runtime": "measure_four_cell_pilot_before_commitment_no_full_replay_authorized_by_preparation",
            },
            "checks_after_replay": [
                "all_native_completion_statuses_ok",
                "each_periodic_cell_exact1381rows_and_key_bits_equal_original_expected_keys",
                "OriginMidPrice_float64_bits_match_original_native_column",
                "metadata_available_time_never_exceeds_origin_time_and_exchange_phase_order_is_causal",
                "censored_numeric_isNaN_inactive_numeric_is0_phase_matches_semantics",
                "event_counters_cover_same_receive_scope_without_claiming_independent_proxy_parity",
                "original_original_config_bytes_equals_preserved_config_equals_new_config",
                "preserved_parquet_bytes_equals_original_before_attributing_new_vs_preserved_differences_to_H10",
                "new_parquet_bytes_equals_preserved_and_original",
            ],
        },
    )
    print("prepared four ordered/reset TRAIN pilots, two old/new parity pairs; no replay; compressed bytes", sum(sizes), "days", len(days))


if __name__ == "__main__":
    main()
