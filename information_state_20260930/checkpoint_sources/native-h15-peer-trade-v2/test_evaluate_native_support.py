"""Synthetic H15 native-contract fixtures; no producer outputs are opened."""

from __future__ import annotations

import copy
import sys
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
evaluation = import_module("evaluate_native_support")
DAY, SYMBOL = evaluation.FIXED_CELLS[0]


def replace(table, name, values, *, dtype=None, role=None):
    field = table.schema.field(name)
    array = pa.array(values, type=field.type if dtype is None else dtype)
    field = pa.field(name, array.type, metadata=field.metadata if role is None else {b"coco.role": role})
    return table.set_column(table.schema.get_field_index(name), field, array)


def table(count=3, *, history=False, periodic=True):
    start, _ = evaluation.session_bounds(DAY)
    times = start + np.arange(count, dtype=np.int64) * 10_000_000
    arrays, fields = [], []
    categories = ("waiting", "unresponsive", "quote", "quote")
    for name in evaluation.REQUIRED_COLUMNS:
        dtype = pa.int64() if name in evaluation.KEYS else pa.string() if name in evaluation.CATEGORICAL_COLUMNS else pa.float64()
        if name in evaluation.KEYS:
            values = times if name == "SampleTime" or (not periodic and name == "SampleBookTime") else times - 1000 if name == "SampleBookTime" else np.arange(count) + 1
        elif name in evaluation.CATEGORICAL_COLUMNS:
            values = [categories[evaluation.CATEGORICAL_COLUMNS.index(name)]] * count
        elif name in evaluation.NUMERIC_COLUMNS:
            values = np.full(count, np.nan)
        else:
            values = np.full(count, 39.975 if name == "OriginMidPrice" else 0.0)
        arrays.append(pa.array(values, type=dtype))
        role = b"time" if name == "SampleTime" else b"context" if name in evaluation.KEYS else b"feature" if name in evaluation.ALPHA_COLUMNS else b"metadata"
        fields.append(pa.field(name, dtype, metadata={b"coco.role": role}))
    native = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    if not history:
        return native
    peer_a = start - 60_000_000
    baseline_a = peer_a - 1_000_000
    for name, value in {
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
    }.items():
        native = replace(native, "H15_" + name, np.full(count, value))
    values = (np.full(count, 0.2), np.full(count, 0.4), np.zeros(count), np.zeros(count), np.zeros(count), np.ones(count), (times - peer_a) / 300_000_000.0)
    for name, value in zip(evaluation.NUMERIC_COLUMNS, values, strict=True):
        native = replace(native, name, value)
    return native


def references(native):
    return native.select([*evaluation.KEYS, "OriginMidPrice"]), native.select(evaluation.KEYS)


def inputs(native=None):
    native = table(1381, history=True) if native is None else native
    parent, keys = references(native)
    events = replace(native, "SampleBookTime", native["SampleTime"])
    return native, parent, keys, events


def later_mark(native, index=3):
    start, _ = evaluation.session_bounds(DAY)
    peer_a = start + index * 10_000_000 - 1_000_000
    values = {
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
    for name, value in values.items():
        column = native["H15_" + name].to_numpy().copy()
        column[index:] = value
        native = replace(native, "H15_" + name, column)
    ages = native[evaluation.NUMERIC_COLUMNS[6]].to_numpy().copy()
    ages[index:] = (native["SampleTime"].to_numpy()[index:] - peer_a) / 300_000_000.0
    return replace(native, evaluation.NUMERIC_COLUMNS[6], ages)


class NativeContractTest(unittest.TestCase):
    def test_exchange_clock_can_lead_receive_and_origin_without_changing_alpha(self):
        native = table(history=True)
        original = evaluation.validate_table(native, periodic=True)
        self.assertTrue(np.all(original["info"]["peer_source_exchange_time"] > original["times"]))
        for name in ("processed_peer_exchange_time", "processed_target_exchange_time", "peer_source_exchange_time", "target_baseline_exchange_time"):
            native = replace(native, "H15_" + name, native["H15_" + name].to_numpy() - 7_200_000_000)
        shifted = evaluation.validate_table(native, periodic=True)
        np.testing.assert_array_equal(original["numeric"].view(np.uint64), shifted["numeric"].view(np.uint64))

    def test_future_availability_tied_qualification_and_future_baseline_reject(self):
        native = table(history=True)
        bad = (
            replace(native, "H15_processed_peer_available_time", native["SampleTime"].to_numpy() + 1),
            replace(native, "H15_qualified_mark_available_time", native["H15_peer_source_available_time"]),
            replace(native, "H15_target_baseline_available_time", native["H15_peer_source_available_time"]),
            replace(native, "H15_source_mark_id", [5.0, 5.5, 5.0]),
        )
        for candidate in bad:
            with self.assertRaises(ValueError):
                evaluation.validate_table(candidate, periodic=True)

    def test_schema_dtype_null_info_role_and_exact_zero_are_fail_closed(self):
        native = table()
        bad = (
            replace(native, "SampleTime", native["SampleTime"].to_numpy(), dtype=pa.float64()),
            replace(native, "SampleBookSeq", [1, None, 3]),
            replace(native, "H15_source_mark_id", [0.0] * 3, role=b"feature"),
            replace(native, "H15_ordered_response_support", [-0.0, 0.0, 0.0]),
            native.append_column("FutureMoveLabel", pa.array([0.0] * 3)),
        )
        for candidate in bad:
            with self.assertRaises(ValueError):
                evaluation.validate_table(candidate, periodic=True)

    def test_qualified_fresh_history_with_unknown_target_work_counts_unique(self):
        native = table(1381, history=True)
        native = replace(native, evaluation.CATEGORICAL_COLUMNS[0], ["invalid"] * len(native))
        native = replace(native, evaluation.CATEGORICAL_COLUMNS[1], ["unknown"] * len(native))
        for name in evaluation.NUMERIC_COLUMNS[3:5]:
            native = replace(native, name, np.full(len(native), np.nan))
        result = evaluation.evaluate_cell(DAY, SYMBOL, *inputs(native))
        self.assertEqual(result["distinct_recent_sampled_marks"], 1)
        self.assertEqual(result["session_marks"], 0)

    def test_stale_history_keeps_work_age_but_hides_each_affected_view(self):
        native = table(history=True)
        native = replace(native, evaluation.CATEGORICAL_COLUMNS[2], ["stale"] * 3)
        native = replace(native, evaluation.NUMERIC_COLUMNS[5], [np.nan] * 3)
        result = evaluation.validate_table(native, periodic=True)
        self.assertTrue(np.isfinite(result["numeric"][:, (1, 6)]).all())
        self.assertFalse(result["fresh_qualified"].any())
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(native, evaluation.NUMERIC_COLUMNS[3], [np.nan] * 3), periodic=True)
        with self.assertRaisesRegex(ValueError, "completed native warmup"):
            evaluation.validate_table(replace(native, evaluation.NUMERIC_COLUMNS[5], [-np.inf] * 3), periodic=True)
        warmup = table()
        for name in evaluation.NUMERIC_COLUMNS:
            warmup = replace(warmup, name, [-np.inf] * 3)
        evaluation.validate_table(warmup, periodic=True)
        with self.assertRaisesRegex(ValueError, "sentinels cannot be mixed"):
            evaluation.validate_table(replace(warmup, evaluation.NUMERIC_COLUMNS[0], [np.nan] * 3), periodic=True)

    def test_all_book_ties_preserve_sequence_and_paired_work_state(self):
        native = table(periodic=False)
        time = native["SampleTime"][0].as_py()
        native = replace(native, "SampleTime", [time, time, time + 1])
        native = replace(native, "SampleBookTime", [time, time, time + 1])
        evaluation.validate_table(native, periodic=False)
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(native, "SampleBookSeq", [1, 1, 2]), periodic=False)
        native = table(history=True)
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(native, evaluation.NUMERIC_COLUMNS[3], [np.nan] * 3), periodic=True)


class CellSupportTest(unittest.TestCase):
    def test_pre_session_carry_in_repeated_id_does_not_earn_session_or_multiple_marks(self):
        result = evaluation.evaluate_cell(DAY, SYMBOL, *inputs())
        self.assertEqual(result["session_marks"], 0)
        self.assertEqual(result["distinct_recent_sampled_marks"], 1)
        self.assertEqual(result["distinct_prior_session_recent_marks"], 1)
        self.assertEqual(result["support_rows"], 461)

    def test_unobserved_ids_and_counter_jumps_are_allowed_but_same_source_facts_fixed(self):
        native = later_mark(table(1381, history=True))
        result = evaluation.evaluate_cell(DAY, SYMBOL, *inputs(native))
        self.assertEqual(result["session_marks"], 2)
        self.assertEqual(result["session_up_marks"], 1)
        self.assertEqual(result["session_down_marks"], 1)
        self.assertEqual(result["distinct_recent_sampled_marks"], 2)
        periodic, parent, keys, events = inputs(native)
        for name in ("H15_source_anchor_mid", "H15_peer_source_exchange_time", evaluation.NUMERIC_COLUMNS[1]):
            changed = replace(events, name, events[name].to_numpy() + 0.001 if name != "H15_peer_source_exchange_time" else events[name].to_numpy() + 1)
            with self.assertRaises(ValueError):
                evaluation.evaluate_cell(DAY, SYMBOL, periodic, parent, keys, changed)

    def test_parent_keys_and_mid_uint64_including_signed_zero_are_exact(self):
        native = table(1381)
        native = replace(native, "OriginMidPrice", np.zeros(len(native)))
        periodic, parent, keys, events = inputs(native)
        evaluation.evaluate_cell(DAY, SYMBOL, periodic, parent, keys, events)
        with self.assertRaises(ValueError):
            evaluation.evaluate_cell(DAY, SYMBOL, periodic, replace(parent, "OriginMidPrice", np.full(len(parent), -0.0)), keys, events)
        with self.assertRaises(ValueError):
            evaluation.evaluate_cell(DAY, SYMBOL, periodic, parent, replace(keys, "SampleBookSeq", keys["SampleBookSeq"].to_numpy() + 1), events)

    def test_support_requires_native_predicate_not_distinct_category_vocabularies(self):
        native = table(1381, history=True)
        result = evaluation.evaluate_cell(DAY, SYMBOL, *inputs(native))
        self.assertTrue(np.all(native[evaluation.CATEGORICAL_COLUMNS[0]].to_numpy() != native[evaluation.CATEGORICAL_COLUMNS[2]].to_numpy()))
        self.assertEqual(result["ordered_response_support_origins"], 0)
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(native, "H15_ordered_response_support", np.ones(len(native))), periodic=True)


class GateAndFreezeTest(unittest.TestCase):
    def cells(self):
        return [
            {
                "day": day,
                "symbol": symbol,
                "status": "observed_native",
                "integrity_passed": True,
                "session_marks": 5,
                "session_up_marks": 3,
                "session_down_marks": 2,
                "distinct_recent_sampled_marks": 5,
                "ordered_response_support_origins": 5,
            }
            for day, symbol in evaluation.FIXED_CELLS
        ]

    def test_fixed_gates_skip_absence_without_vacuity_and_reject_numeric_aliases(self):
        self.assertTrue(evaluation.gates(self.cells())["passed"])
        for value in (True, 5.0, -1):
            cells = self.cells()
            cells[0]["distinct_recent_sampled_marks"] = value
            with self.assertRaises(ValueError):
                evaluation.gates(cells)
        cells = [{"day": c["day"], "symbol": c["symbol"], "status": "missing_fixed_input_skip"} for c in self.cells()]
        self.assertFalse(evaluation.gates(cells)["passed"])
        with self.assertRaises(ValueError):
            evaluation.gates(list(reversed(self.cells())))

    def test_configured_writer_cannot_be_discarded_before_any_output_decode(self):
        def node(path):
            return {"path": str(Path(path).resolve()), "sha256": "0" * 64, "size": 0}

        source_paths = (
            Path(evaluation.__file__),
            Path(evaluation.__file__).with_name("prepare_native.py"),
            evaluation.ROOT / "AstraResearch/astra/io.py",
            evaluation.EVALUATION_PLAN,
            evaluation.TEST_SOURCE,
        )
        sources = [node(path) for path in source_paths]
        profile = {
            "schema": "h15-native-support-evaluation-profile-v2",
            "evaluation_contract": copy.deepcopy(evaluation.EVALUATION_CONTRACT),
            "source_proof": sources[-2:],
            "producer_identity": "producer",
            "parents": [],
            "cells": [{"day": day, "symbol": symbol, "status": "missing_fixed_input_skip"} for day, symbol in evaluation.FIXED_CELLS],
        }
        method = {"profile": node(evaluation.EVALUATION_PROFILE), "profile_identity": evaluation.digest(profile), "sources": sources}
        producer = {"identity": "producer", "sources": []}
        preparation = {"jobs": [{"day": day, "symbols": list(evaluation.TARGETS)} for day in evaluation.DAYS]}
        producer_profile = {"parents": []}
        with (
            patch.object(evaluation, "canonical_method", return_value=method),
            patch.object(evaluation, "read_yaml", return_value=profile),
            patch.object(evaluation, "check_record", return_value=Path("/synthetic")),
            patch.object(evaluation, "producer_context", return_value=(producer, preparation, producer_profile, [])),
            self.assertRaisesRegex(ValueError, "cannot be silently skipped"),
        ):
            evaluation.validate_evaluation_freeze(Path("/synthetic/frozen.yaml"))


if __name__ == "__main__":
    unittest.main()
