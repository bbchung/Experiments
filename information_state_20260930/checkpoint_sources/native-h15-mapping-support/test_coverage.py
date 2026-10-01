"""Synthetic mapping/cohort/source-boundary fixtures; no market outputs opened."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
coverage = import_module("coverage")
native = coverage.native
DAY, SYMBOL = coverage.DAYS[0], "2454"


def replace(table, name, values, *, dtype=None, role=None):
    field = table.schema.field(name)
    array = pa.array(values, type=field.type if dtype is None else dtype)
    field = pa.field(name, array.type, metadata=field.metadata if role is None else {b"coco.role": role})
    return table.set_column(table.schema.get_field_index(name), field, array)


def table(*, history=True):
    count = 1381
    start, _ = coverage.session_bounds(DAY)
    times = start + np.arange(count, dtype=np.int64) * 10_000_000
    fields, arrays = [], []
    categories = ("waiting", "unresponsive", "quote", "quote")
    for name in native.REQUIRED_COLUMNS:
        dtype = pa.int64() if name in native.KEYS else pa.string() if name in native.CATEGORICAL_COLUMNS else pa.float64()
        if name in native.KEYS:
            values = times if name == "SampleTime" else times - 1000 if name == "SampleBookTime" else np.arange(count) + 1
        elif name in native.CATEGORICAL_COLUMNS:
            values = [categories[native.CATEGORICAL_COLUMNS.index(name)]] * count
        elif name in native.NUMERIC_COLUMNS:
            values = np.full(count, np.nan)
        else:
            values = np.full(count, 39.975 if name == "OriginMidPrice" else 0.0)
        role = b"time" if name == "SampleTime" else b"context" if name in native.KEYS else b"feature" if name in native.ALPHA_COLUMNS else b"metadata"
        fields.append(pa.field(name, dtype, metadata={b"coco.role": role}))
        arrays.append(pa.array(values, type=dtype))
    result = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    if not history:
        return result
    peer_a, baseline_a = start - 60_000_000, start - 61_000_000
    facts = {
        "processed_peer_available_time": peer_a,
        "processed_peer_exchange_time": peer_a + 3_600_000_000,
        "processed_target_available_time": baseline_a,
        "processed_target_exchange_time": baseline_a + 3_600_000_000,
        "source_mark_id": 5,
        "peer_source_available_time": peer_a,
        "peer_source_exchange_time": peer_a + 3_600_000_000,
        "qualified_mark_available_time": peer_a + 1,
        "target_baseline_available_time": baseline_a,
        "target_baseline_receive_time": baseline_a - 1,
        "target_baseline_exchange_time": baseline_a + 3_600_000_000,
        "cumulative_source_mark_count": 5,
        "cumulative_qualified_mark_count": 2,
        "source_anchor_mid": 39.975,
        "source_confirmed_mid": 40.025,
        "target_baseline_mid": 99.975,
        "mark_is_qualified": 1,
    }
    for name, value in facts.items():
        result = replace(result, "H15_" + name, np.full(count, value))
    values = (np.full(count, 0.2), np.full(count, 0.4), np.zeros(count), np.zeros(count), np.zeros(count), np.ones(count), (times - peer_a) / 300_000_000.0)
    for name, column in zip(native.NUMERIC_COLUMNS, values, strict=True):
        result = replace(result, name, column)
    return result


def parent(table):
    return table.select([*native.KEYS, "OriginMidPrice"])


def cells(*, observed=True):
    return [
        {
            "day": day,
            "symbol": symbol,
            "peer": coverage.MAPPING[symbol],
            "status": "observed_native" if observed else "missing_fixed_input_skip",
            "integrity_passed": True,
            "session_marks": 10,
            "session_up_marks": 5,
            "session_down_marks": 5,
            "distinct_recent_sampled_marks": 5,
            "ordered_response_support_origins": 20,
        }
        for day, symbol in coverage.FIXED_CELLS
    ]


class MappingGateTest(unittest.TestCase):
    def test_exact_full_mapping_and_each_present_gate(self):
        self.assertEqual(coverage.MAPPING["2330"], "2317")
        self.assertTrue(all(peer == "2330" for symbol, peer in coverage.MAPPING.items() if symbol != "2330"))
        fixture = cells()
        self.assertTrue(coverage.gates(fixture)["passed"])
        fixture[8]["session_marks"], fixture[8]["session_up_marks"], fixture[8]["session_down_marks"] = 4, 2, 2
        result = coverage.gates(fixture)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["aggregate5_up"])
        self.assertFalse(result["checks"]["each_present5_session_marks"])
        fixture = cells()
        fixture[8]["distinct_recent_sampled_marks"] = 4
        self.assertFalse(coverage.gates(fixture)["passed"])

    def test_missing_is_descriptive_without_vacuous_pass_or_replacement(self):
        fixture = cells(observed=False)
        self.assertFalse(coverage.gates(fixture)["passed"])
        fixture[0] = cells()[0]
        result = coverage.gates(fixture)
        self.assertTrue(result["passed"])
        self.assertEqual(result["missing_cells"], 31)
        self.assertFalse(result["all_fixed_cells_observed"])
        with self.assertRaises(ValueError):
            coverage.gates(fixture[:-1])

    def test_identity_peer_order_types_and_direction_total_fail_closed(self):
        mutations = (
            lambda c: c.reverse(),
            lambda c: c[0].update(peer="2454"),
            lambda c: c[0].update(session_marks=True),
            lambda c: c[0].update(distinct_recent_sampled_marks=5.0),
            lambda c: c[0].update(session_marks=9),
            lambda c: c[0].update(ordered_response_support_origins=-1),
            lambda c: c[0].update(status="dropped_after_counts"),
        )
        for mutation in mutations:
            fixture = cells()
            mutation(fixture)
            with self.assertRaises(ValueError):
                coverage.gates(fixture)


class NativeMappingCellTest(unittest.TestCase):
    def test_new_target_without_mutating_frozen_pair_universe(self):
        original_fixed = copy.deepcopy(native.FIXED_CELLS)
        fixture = table()
        result = coverage.evaluate_cell(DAY, SYMBOL, fixture, parent(fixture))
        self.assertEqual(result["peer"], "2330")
        self.assertEqual(result["support_rows"], 461)
        self.assertEqual(result["distinct_recent_sampled_marks"], 1)
        self.assertEqual(result["distinct_prior_session_recent_marks"], 1)
        self.assertEqual(result["session_marks"], 0)
        self.assertEqual(native.FIXED_CELLS, original_fixed)
        with self.assertRaises(ValueError):
            coverage.evaluate_cell("20260121", SYMBOL, fixture, parent(fixture))

    def test_unknown_work_does_not_remove_fresh_native_history(self):
        fixture = table()
        fixture = replace(fixture, native.CATEGORICAL_COLUMNS[0], ["invalid"] * len(fixture))
        fixture = replace(fixture, native.CATEGORICAL_COLUMNS[1], ["unknown"] * len(fixture))
        for name in native.NUMERIC_COLUMNS[3:5]:
            fixture = replace(fixture, name, np.full(len(fixture), np.nan))
        result = coverage.evaluate_cell(DAY, SYMBOL, fixture, parent(fixture))
        self.assertEqual(result["distinct_recent_sampled_marks"], 1)
        self.assertEqual(result["ordered_response_support_origins"], 0)

    def test_source_id_clock_price_and_work_are_fixed_across_origins(self):
        fixture = table()
        for name in ("H15_source_anchor_mid", "H15_peer_source_exchange_time", native.NUMERIC_COLUMNS[1]):
            values = fixture[name].to_numpy().copy()
            values[100:] += 1.0 if "exchange_time" in name else 0.005
            with self.assertRaisesRegex(ValueError, "same sourceID"):
                coverage.evaluate_cell(DAY, SYMBOL, replace(fixture, name, values), parent(fixture))

    def test_counter_jumps_do_not_require_every_intermediate_id(self):
        fixture = table()
        start, _ = coverage.session_bounds(DAY)
        peer_a = start + 29_000_000
        facts = {
            "source_mark_id": 7,
            "cumulative_source_mark_count": 7,
            "cumulative_qualified_mark_count": 4,
            "session_qualified_mark_count": 2,
            "session_up_qualified_mark_count": 1,
            "session_down_qualified_mark_count": 1,
            "peer_source_available_time": peer_a,
            "peer_source_exchange_time": peer_a + 3_600_000_000,
            "qualified_mark_available_time": peer_a + 1,
            "target_baseline_available_time": peer_a - 1_000_000,
            "target_baseline_receive_time": peer_a - 1_000_001,
            "target_baseline_exchange_time": peer_a - 1_000_000 + 3_600_000_000,
            "processed_peer_available_time": peer_a,
            "processed_peer_exchange_time": peer_a + 3_600_000_000,
            "processed_target_available_time": peer_a - 1_000_000,
            "processed_target_exchange_time": peer_a - 1_000_000 + 3_600_000_000,
        }
        for name, value in facts.items():
            values = fixture["H15_" + name].to_numpy().copy()
            values[3:] = value
            fixture = replace(fixture, "H15_" + name, values)
        ages = fixture[native.NUMERIC_COLUMNS[6]].to_numpy().copy()
        ages[3:] = (fixture["SampleTime"].to_numpy()[3:] - peer_a) / 300_000_000.0
        fixture = replace(fixture, native.NUMERIC_COLUMNS[6], ages)
        result = coverage.evaluate_cell(DAY, SYMBOL, fixture, parent(fixture))
        self.assertEqual(result["session_marks"], 2)
        self.assertEqual(result["distinct_recent_sampled_marks"], 2)

    def test_source_receive_future_rejected_exchange_offset_admissible(self):
        fixture = table()
        result = coverage.evaluate_cell(DAY, SYMBOL, fixture, parent(fixture))
        for name in ("processed_peer_exchange_time", "processed_target_exchange_time", "peer_source_exchange_time", "target_baseline_exchange_time"):
            fixture = replace(fixture, "H15_" + name, fixture["H15_" + name].to_numpy() - 7_200_000_000)
        self.assertEqual(coverage.evaluate_cell(DAY, SYMBOL, fixture, parent(fixture)), result)
        fixture = replace(fixture, "H15_processed_peer_available_time", fixture["SampleTime"].to_numpy() + 1)
        with self.assertRaises(ValueError):
            coverage.evaluate_cell(DAY, SYMBOL, fixture, parent(fixture))

    def test_original_exact_mid_bits_key_dtype_and_native_schema(self):
        fixture = table()
        zero = replace(fixture, "OriginMidPrice", np.zeros(len(fixture)))
        reference = replace(parent(zero), "OriginMidPrice", np.full(len(fixture), -0.0))
        with self.assertRaisesRegex(ValueError, "uint64bits"):
            coverage.evaluate_cell(DAY, SYMBOL, zero, reference)
        with self.assertRaises(ValueError):
            coverage.evaluate_cell(DAY, SYMBOL, fixture, replace(parent(fixture), "SampleTime", fixture["SampleTime"], dtype=pa.float64()))
        bad = (
            fixture.append_column("FutureMoveLabel", pa.array(np.zeros(len(fixture)))),
            replace(fixture, "H15_source_mark_id", fixture["H15_source_mark_id"], role=b"feature"),
            replace(fixture, "SampleBookSeq", [None] + list(range(2, len(fixture) + 1))),
        )
        for candidate in bad:
            with self.assertRaises(ValueError):
                coverage.evaluate_cell(DAY, SYMBOL, candidate, parent(fixture))


class FixedInputMetadataTest(unittest.TestCase):
    def test_full_receipt_flat_dates_match_immutable_pair_train_dates(self):
        full = {"fixed_role_days": [*coverage.DAYS, "20260121"], "universe_order": list(coverage.SYMBOLS)}
        profile = {"fixed_train_days": list(coverage.DAYS)}
        coverage.validate_full_metadata(full, profile)
        for dates in ({"train": list(coverage.DAYS)}, list(reversed(coverage.DAYS)), [*coverage.DAYS, coverage.DAYS[-1]], [20260119, 20260120]):
            with self.assertRaises(ValueError):
                coverage.validate_full_metadata({**full, "fixed_role_days": dates}, profile)
        with self.assertRaises(ValueError):
            coverage.validate_full_metadata(full, {"fixed_train_days": ["20260120", "20260121"]})
        with self.assertRaises(ValueError):
            coverage.validate_full_metadata({**full, "universe_order": list(reversed(coverage.SYMBOLS))}, profile)

    def test_missing_carrier_skips_whole_daily_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = {}
            for symbol in coverage.SYMBOLS:
                path = directory / f"{symbol}.bin.zst"
                path.write_bytes(b"synthetic hash metadata only")
                raw[(DAY, symbol)] = coverage.record(path)
            basic = directory / "basic.csv"
            basic.write_text("symbol\n" + "\n".join(coverage.SYMBOLS) + "\n")
            parents = []
            for symbol in coverage.SYMBOLS:
                path = directory / f"{symbol}.parquet"
                path.write_bytes(b"synthetic file identity only, not Parquet")
                parents.append({"day": DAY, "symbol": symbol, "parent_origins": coverage.record(path)})
            Path(raw[(DAY, "2454")]["path"]).unlink()
            selected, missing = coverage.select_day(DAY, raw, coverage.record(basic), parents)
            self.assertEqual(selected, [])
            self.assertEqual(missing[0]["symbol"], "2454")

    def test_missing_parent_skips_only_writer_and_duplicate_contract_rejects(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = {}
            parents = []
            for symbol in coverage.SYMBOLS:
                path = directory / f"{symbol}.bin.zst"
                path.write_bytes(b"synthetic")
                raw[(DAY, symbol)] = coverage.record(path)
                path = directory / f"{symbol}.parquet"
                path.write_bytes(b"synthetic")
                parents.append({"day": DAY, "symbol": symbol, "parent_origins": coverage.record(path)})
            basic = directory / "basic.csv"
            basic.write_text("symbol\n" + "\n".join(coverage.SYMBOLS) + "\n")
            Path(parents[2]["parent_origins"]["path"]).unlink()
            selected, missing = coverage.select_day(DAY, raw, coverage.record(basic), parents)
            self.assertEqual(selected, [symbol for symbol in coverage.SYMBOLS if symbol != "2454"])
            self.assertEqual(missing[0]["reason"], "missing_original_parent_skip_only_target_writer")
            basic.write_text(basic.read_text() + "2330\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                coverage.select_day(DAY, raw, coverage.record(basic), parents)

    def test_configured_output_cannot_be_relabeled_missing_before_decode(self):
        fixed = [{"day": day, "symbol": symbol, "peer": coverage.MAPPING[symbol], "status": "missing_fixed_input_skip"} for day, symbol in coverage.FIXED_CELLS]
        profile = {
            "evaluation_contract": coverage.EVALUATION_CONTRACT,
            "producer": {"path": "/synthetic/producer.yaml", "sha256": "x"},
            "producer_anchor": {"path": "/synthetic/producer.yaml.identity", "sha256": "x"},
            "producer_identity": "producer",
            "parents": [],
            "cells": fixed,
        }
        method = {"profile": {"path": str(coverage.EVALUATION_PROFILE), "sha256": "x"}, "profile_identity": coverage.digest(profile)}
        producer = {"identity": "producer"}
        jobs = [{"day": day, "symbols": list(coverage.SYMBOLS), "work": f"/synthetic/{day}"} for day in coverage.DAYS]
        with (
            patch.object(coverage, "canonical_method", return_value=method),
            patch.object(coverage, "check_record"),
            patch.object(coverage, "read_yaml", return_value=profile),
            patch.object(coverage, "producer_context", return_value=(producer, {"jobs": jobs}, {"parents": []})),
            self.assertRaisesRegex(ValueError, "configured writer skipped"),
        ):
            coverage.evaluation_context(Path("/synthetic/evaluation.yaml"))


if __name__ == "__main__":
    unittest.main()
