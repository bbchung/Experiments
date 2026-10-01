"""Read-only native integrity/support audit; no replay, training or FE ranking."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
KEYS = ["SampleTime", "SampleBookTime", "SampleBookSeq"]
NUMERIC = ["direction", "log_work_strength", "signed_after_containment_share", "signed_after_repair_share", "delayed_repair_fraction", "signed_renewed_work", "mid_progress_5ticks"]
PHASES = {"warmup", "inactive", "contained", "repaired", "renewed", "resumed", "stale", "censored"}


def read(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array(table, name):
    return table[name].to_numpy(zero_copy_only=False)


def same_bits(left, right):
    if left.type != right.type or left.null_count != right.null_count or not left.is_null().equals(right.is_null()):
        return False
    if pa.types.is_float64(left.type):
        return np.array_equal(left.to_numpy(zero_copy_only=False).view(np.uint64), right.to_numpy(zero_copy_only=False).view(np.uint64))
    return left.equals(right)


def phase_checks(table, module_id, mask):
    phase = np.asarray(table[f"DemandRepairRenewal.{module_id}.phase.0"].to_pylist())
    numeric = np.column_stack([array(table, f"DemandRepairRenewal.{module_id}.{field}.0") for field in NUMERIC])
    active = np.isin(phase, ["contained", "repaired", "renewed", "resumed"])
    inactive = phase == "inactive"
    warmup = phase == "warmup"
    missing = np.isin(phase, ["stale", "censored"])
    resumed = (phase == "resumed") & mask
    return {
        "phases": dict(Counter(phase.tolist())),
        "all_phase_names_supported": set(phase).issubset(PHASES),
        "inactive_numeric_exact_positive_zero": bool((numeric[inactive].view(np.uint64) == 0).all()),
        "warmup_numeric_negative_infinity": bool(np.isneginf(numeric[warmup]).all()),
        "stale_and_censored_numeric_nan": bool(np.isnan(numeric[missing]).all()),
        "active_numeric_finite": bool(np.isfinite(numeric[active]).all()),
        "active_direction_known": bool(np.isin(numeric[active, 0], [-1.0, 1.0]).all()),
        "shares_and_repair_bounded": bool((np.abs(numeric[active, 2:4]) <= 1.0).all() and (numeric[active, 4] >= 0.0).all() and (numeric[active, 4] < 1.0).all()),
        "work_strength_nonnegative": bool((numeric[active, 1] >= 0.0).all()),
        "renewed_work_oriented_by_direction": bool((numeric[active, 5] * numeric[active, 0] >= 0.0).all()),
        "resumed_30s_origins": int(resumed.sum()),
        "resumed_buy_30s_origins": int((resumed & (numeric[:, 0] == 1.0)).sum()),
        "resumed_sell_30s_origins": int((resumed & (numeric[:, 0] == -1.0)).sum()),
    }


def metadata_checks(table, name):
    sample = array(table, "SampleTime")
    exchange = array(table, f"H10_{name}_processed_cluster_exchange_time")
    available = array(table, f"H10_{name}_processed_cluster_available_time")
    counts = np.column_stack(
        [array(table, f"H10_{name}_{field}") for field in ["cumulative_complete_chain_count", "cumulative_buy_complete_chain_count", "cumulative_sell_complete_chain_count"]]
    )
    return {
        "available_not_after_origin": bool(np.isfinite(available).all() and (available >= 0).all() and (available <= sample).all()),
        "exchange_not_after_available": bool(np.isfinite(exchange).all() and (exchange >= 0).all() and (exchange <= available).all()),
        "counters_integer_monotone": bool(np.isfinite(counts).all() and (counts >= 0).all() and (counts == np.floor(counts)).all() and (np.diff(counts, axis=0) >= 0).all()),
        "counters_side_sum": bool((counts[:, 0] == counts[:, 1] + counts[:, 2]).all()),
        "counter_at_first_origin": counts[0].astype(int).tolist(),
        "counter_at_last_origin": counts[-1].astype(int).tolist(),
        "counter_delta_scoped_origins": (counts[-1] - counts[0]).astype(int).tolist(),
    }


def event_summary(path):
    selected = ["SampleTime", "DemandRepairRenewal.0.phase.0", "DemandRepairRenewal.0.direction.0", "H10_ordered_processed_cluster_available_time"]
    counter = Counter()
    available_valid = True
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=selected, use_threads=False):
        phase = np.asarray(batch.column(1).to_pylist())
        sign = batch.column(2).to_numpy(zero_copy_only=False)
        times = batch.column(0).to_numpy(zero_copy_only=False)
        available = batch.column(3).to_numpy(zero_copy_only=False)
        available_valid &= bool((available <= times).all())
        counter.update(phase.tolist())
        counter["resumed_buy_book_rows"] += int(((phase == "resumed") & (sign == 1.0)).sum())
        counter["resumed_sell_book_rows"] += int(((phase == "resumed") & (sign == -1.0)).sum())
    return {"phase_and_direction_counts": dict(counter), "available_not_after_book_row": available_valid}


def control(pair):
    original = Path(pair["original_native"])
    outputs = {version: Path(pair[version]["work"]) / "data" / pair["day"] / pair["symbol"] / "values.parquet" for version in ("preserved", "new")}
    schema = pq.read_schema(original)
    labels = [name for name in schema.names if name.startswith(("mid_return_ticks[", "sticky_return_ticks[", "mid_endpoint."))]
    columns = [*KEYS, "OriginMidPrice", *labels]
    reference = pq.read_table(original, columns=columns, use_threads=False)
    tables = {version: pq.read_table(path, columns=columns, use_threads=False) for version, path in outputs.items()}
    original_hash = digest(original)
    hashes = {version: digest(path) for version, path in outputs.items()}
    checks = {version: {name: same_bits(reference[name], table[name]) for name in columns} for version, table in tables.items()}
    return {
        "day": pair["day"],
        "symbol": pair["symbol"],
        "original_sha256_matches_preparation": original_hash == pair["original_native_sha256"],
        "original_sha256": original_hash,
        "replay_sha256": hashes,
        "preserved_new_byte_invariance": hashes["preserved"] == hashes["new"],
        "historical_original_byte_correspondence": all(value == original_hash for value in hashes.values()),
        "raw_key_mid_and_label_bit_checks": checks,
        "raw_key_mid_and_all_label_bits_match": all(all(fields.values()) for fields in checks.values()),
        "semantic_scope": "control_keys_raw_mid_and_all16labels_only_no_feature_fitting_or_ranking",
    }


def main():
    preparation = read(OUTPUT / "preparation-receipt.yaml")
    screen_binding = read(OUTPUT / "screen-binding.yaml")
    screen = Path(screen_binding["screen"])
    assert digest(screen) == screen_binding["screen_sha256"]
    assert digest(OUTPUT / "preparation-receipt.yaml") == screen_binding["preparation_receipt_sha256"]
    for job in preparation["pilot_jobs"]:
        work = Path(job["work"])
        status = work / "status" / (job["day"] + ".yaml")
        if not status.exists() or read(status)["status"] != "completed":
            raise RuntimeError("Root native replay is not complete; never audit partial outputs")
    destination = OUTPUT / "independent-native-validation.yaml"
    if destination.exists():
        raise RuntimeError("Independent validation already exists; no implicit overwrite")
    cells = []
    for job in preparation["pilot_jobs"]:
        work = Path(job["work"])
        produced = work / "data" / job["day"] / job["symbol"] / "values.parquet"
        events = work / "events" / job["day"] / job["symbol"] / "values.parquet"
        table = pq.read_table(produced, use_threads=False)
        original = pq.read_table(job["original_native"], columns=[*KEYS, "OriginMidPrice"], use_threads=False)
        expected = pq.read_table(work / "expected-origin-keys.parquet", use_threads=False)
        samples = array(table, "SampleTime")
        mask = samples.astype(np.int64) % 30_000_000 == 0
        row_checks = {
            "completed_without_fatal_error": read(work / "status" / (job["day"] + ".yaml")).get("fatal_error") is False,
            "rows_1381": len(table) == job["expected_rows"] == 1381,
            "keys_exact_original": all(same_bits(table[name], original[name]) for name in KEYS),
            "keys_exact_prepared": all(same_bits(table[name], expected[name]) for name in KEYS),
            "raw_mid_float64": table["OriginMidPrice"].type == pa.float64(),
            "raw_mid_bits_exact_original": same_bits(table["OriginMidPrice"], original["OriginMidPrice"]),
            "source_book_not_after_origin": bool((array(table, "SampleBookTime") <= samples).all()),
            "evaluation_grid_461": int(mask.sum()) == 461,
            "no_label_columns": not any(name.startswith(("mid_return_ticks[", "sticky_return_ticks[", "mid_endpoint.")) for name in table.column_names),
            "config_unchanged": digest(work / "config.yaml") == job["config_sha256"],
            "expected_key_artifact_unchanged": digest(work / "expected-origin-keys.parquet") == job["expected_keys_sha256"],
            "original_config_unchanged": digest(Path(job["original_config"])) == job["original_config_sha256"],
        }
        cells.append(
            {
                "day": job["day"],
                "symbol": job["symbol"],
                "produced_sha256": digest(produced),
                "events_sha256": digest(events),
                "row_checks": row_checks,
                "ordered": phase_checks(table, 0, mask),
                "reset": phase_checks(table, 1, mask),
                "ordered_metadata": metadata_checks(table, "ordered"),
                "reset_metadata": metadata_checks(table, "reset"),
                "events": event_summary(events),
            }
        )
    controls = [control(pair) for pair in preparation["parity_pairs"]]
    native_buy = sum(cell["ordered_metadata"]["counter_at_last_origin"][1] for cell in cells)
    native_sell = sum(cell["ordered_metadata"]["counter_at_last_origin"][2] for cell in cells)
    sampled_buy = sum(cell["ordered"]["resumed_buy_30s_origins"] for cell in cells)
    sampled_sell = sum(cell["ordered"]["resumed_sell_30s_origins"] for cell in cells)
    support = {
        "all_four_fixed_cells": len(cells) == 4,
        "each_cell_native_completed_chain": all(cell["ordered_metadata"]["counter_at_last_origin"][0] >= 1 for cell in cells),
        "native_both_directions": native_buy > 0 and native_sell > 0,
        "each_cell_resumed_on_fixed30s_grid": all(cell["ordered"]["resumed_30s_origins"] >= 1 for cell in cells),
        "sampled_resumed_both_directions": sampled_buy > 0 and sampled_sell > 0,
    }
    integrity = all(all(cell["row_checks"].values()) for cell in cells)
    integrity &= all(control["original_sha256_matches_preparation"] and control["raw_key_mid_and_all_label_bits_match"] for control in controls)
    for cell in cells:
        for key in ("ordered", "reset"):
            integrity &= all(value for value in cell[key].values() if isinstance(value, bool))
        for key in ("ordered_metadata", "reset_metadata"):
            integrity &= all(value for value in cell[key].values() if isinstance(value, bool))
        integrity &= cell["events"]["available_not_after_book_row"]
    historical_parity = all(control["historical_original_byte_correspondence"] for control in controls)
    invariant = all(control["preserved_new_byte_invariance"] for control in controls)
    document = {
        "schema": "h10-independent-native-validation-v1",
        "script_sha256": digest(Path(__file__)),
        "screen_sha256": screen_binding["screen_sha256"],
        "preparation_receipt_sha256": digest(OUTPUT / "preparation-receipt.yaml"),
        "producer_source_build_receipt_sha256": digest(OUTPUT / "producer-source-build.yaml"),
        "pilot_labels_read": False,
        "control_labels_read_for_bit_parity_only": True,
        "model_fits": 0,
        "native_replays_launched": 0,
        "cells": cells,
        "controls": controls,
        "support_checks": support,
        "native_feature_integrity": bool(integrity),
        "preserved_new_invariance": invariant,
        "historical_original_byte_correspondence": historical_parity,
        "original_screen_pass": bool(integrity and invariant and historical_parity and all(support.values())),
        "descriptive_support_conditional_on_component_provenance": bool(integrity and invariant and all(support.values())),
        "predictive_admission": False,
        "scoped_counter_delta_note": "Report first/last/delta separately; fixed-day full-prefix episodes may begin before0910 and retained counters do not equal resetting proxy52",
    }
    destination.write_text(yaml.safe_dump(document, sort_keys=False))
    print("integrity", integrity, "invariance", invariant, "historical", historical_parity, "support", support)
    for cell in cells:
        print(
            cell["day"],
            cell["symbol"],
            "native",
            cell["ordered_metadata"]["counter_at_last_origin"],
            "30s resumed",
            cell["ordered"]["resumed_30s_origins"],
            "buy/sell",
            cell["ordered"]["resumed_buy_30s_origins"],
            cell["ordered"]["resumed_sell_30s_origins"],
        )


if __name__ == "__main__":
    main()
