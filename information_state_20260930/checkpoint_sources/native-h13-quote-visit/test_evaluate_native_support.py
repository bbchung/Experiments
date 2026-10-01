"""Synthetic native support-contract fixtures; never inspect producer outputs."""

from __future__ import annotations

import copy
import sys
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parent))
evaluation = import_module("evaluate_native_support")

DAY = "20260119"
SYMBOL = "2308"


def table(*, count=3, periodic=True, history=False):
    start, _ = evaluation.session_bounds(DAY)
    times = start + np.arange(count, dtype=np.int64) * 10_000_000
    arrays = {}
    for name in evaluation.REQUIRED_COLUMNS:
        if name in evaluation.KEYS:
            values = times if name == "SampleTime" else times - 1000 if name == "SampleBookTime" else np.arange(count, dtype=np.int64) + 1
            if not periodic and name == "SampleBookTime":
                values = times
            arrays[name] = pa.array(values, type=pa.int64())
        elif name == evaluation.ALPHA_COLUMNS[0]:
            arrays[name] = pa.array(["bilateral" if history else "inactive"] * count, type=pa.string())
        elif name in evaluation.ALPHA_COLUMNS:
            arrays[name] = pa.array(np.full(count, np.nan), type=pa.float64())
        else:
            arrays[name] = pa.array(np.full(count, 39.975 if name == "OriginMidPrice" else 0.0), type=pa.float64())
    result = pa.table(arrays)
    fields = []
    for field in result.schema:
        role = b"time" if field.name == "SampleTime" else b"context" if field.name in evaluation.KEYS else b"feature" if field.name in evaluation.ALPHA_COLUMNS else b"metadata"
        fields.append(field.with_metadata({b"coco.role": role}))
    result = pa.Table.from_arrays(result.columns, schema=pa.schema(fields))
    if history:
        available = np.full(count, start - 60_000_000, dtype=np.float64)
        result = replace(result, "H13_processed_cluster_available_time", available)
        result = replace(result, "H13_processed_cluster_exchange_time", available + 3_600_000_000)
        result = replace(result, "H13_completed_landmark_available_time", available)
        result = replace(result, "H13_completed_landmark_exchange_time", available + 3_600_000_000)
        for name in (
            "completed_landmark_id",
            "current_visit_id",
            "cumulative_completed_landmark_count",
            "cumulative_up_completed_landmark_count",
            "cumulative_quote_led_visit_count",
        ):
            result = replace(result, "H13_" + name, np.ones(count))
        for name, values in zip(evaluation.ALPHA_COLUMNS[1:], (np.ones(count), np.zeros(count), np.full(count, 0.25), (times - available) / 300_000_000.0), strict=True):
            result = replace(result, name, values)
    return result


def replace(original, name, values, *, dtype=None, role=None):
    field = original.schema.field(name)
    values = pa.array(values, type=field.type if dtype is None else dtype)
    field = pa.field(name, values.type, metadata=field.metadata if role is None else {b"coco.role": role})
    return original.set_column(original.schema.get_field_index(name), field, values)


def reference(periodic):
    return periodic.select([*evaluation.KEYS, "OriginMidPrice"]), periodic.select(evaluation.KEYS)


def full_cell(*, history=False):
    periodic = table(count=1381, history=history)
    events = replace(periodic, "SampleBookTime", periodic["SampleTime"])
    parent, keys = reference(periodic)
    return periodic, parent, keys, events


class NativeTableTest(unittest.TestCase):
    def test_valid_integer_native_table_and_independent_exchange_epoch_are_accepted(self):
        native = table(history=True)
        result = evaluation.validate_table(native, periodic=True)
        self.assertTrue(result["exposed"].all())
        self.assertTrue((result["info"]["processed_cluster_exchange_time"] > result["times"]).all())
        shifted = replace(native, "H13_processed_cluster_exchange_time", result["info"]["processed_cluster_exchange_time"] - 7_200_000_000)
        shifted = replace(shifted, "H13_completed_landmark_exchange_time", result["info"]["completed_landmark_exchange_time"] - 7_200_000_000)
        other = evaluation.validate_table(shifted, periodic=True)
        np.testing.assert_array_equal(result["numeric"].view(np.uint64), other["numeric"].view(np.uint64))

    def test_future_receive_availability_and_noninteger_info_are_rejected(self):
        native = table(history=True)
        times = native["SampleTime"].to_numpy()
        variants = (
            replace(native, "H13_processed_cluster_available_time", times + 1),
            replace(native, "H13_completed_landmark_available_time", times + 1),
            replace(native, "H13_current_visit_id", [1.0, 1.5, 1.0]),
            replace(native, "H13_processed_cluster_available_time", [float(2**53)] * 3),
        )
        for candidate in variants:
            with self.subTest(schema=candidate.schema), self.assertRaises(ValueError):
                evaluation.validate_table(candidate, periodic=True)

    def test_native_key_dtype_null_and_role_cannot_be_coerced(self):
        native = table()
        variants = (
            replace(native, "SampleTime", native["SampleTime"].to_pylist(), dtype=pa.float64()),
            replace(native, "SampleBookSeq", [1, None, 3]),
            replace(native, "H13_completed_landmark_id", [0.0] * 3, role=b"feature"),
        )
        for candidate in variants:
            with self.assertRaises(ValueError):
                evaluation.validate_table(candidate, periodic=True)

    def test_all_book_retains_receive_ties_but_requires_strict_source_sequences(self):
        native = table(periodic=False)
        time = native["SampleTime"][0].as_py()
        native = replace(native, "SampleTime", [time, time, time + 1])
        native = replace(native, "SampleBookTime", [time, time, time + 1])
        evaluation.validate_table(native, periodic=False)
        with self.assertRaises(ValueError):
            evaluation.validate_table(native, periodic=True)
        for sequences in ([1, 1, 2], [2, 1, 3]):
            with self.assertRaises(ValueError):
                evaluation.validate_table(replace(native, "SampleBookSeq", sequences), periodic=False)

    def test_zero_info_and_zero_anchor_identity_are_bit_exact(self):
        native = table()
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(native, "H13_current_visit_id", [-0.0, 0.0, 0.0]), periodic=True)
        native = table(history=True)
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(native, evaluation.ALPHA_COLUMNS[2], [-0.0, 0.0, 0.0]), periodic=True)

    def test_warmup_missing_historical_and_stale_states_are_distinct(self):
        native = table()
        warmup = replace(native, evaluation.ALPHA_COLUMNS[0], ["warmup"] * 3)
        for name in evaluation.ALPHA_COLUMNS[1:]:
            warmup = replace(warmup, name, [-np.inf] * 3)
        evaluation.validate_table(warmup, periodic=True)
        with self.assertRaises(ValueError):
            evaluation.validate_table(replace(warmup, evaluation.ALPHA_COLUMNS[1], [np.nan] * 3), periodic=True)
        stale = replace(table(history=True), evaluation.ALPHA_COLUMNS[0], ["stale"] * 3)
        with self.assertRaises(ValueError):
            evaluation.validate_table(stale, periodic=True)
        for name in evaluation.ALPHA_COLUMNS[1:]:
            stale = replace(stale, name, [np.nan] * 3)
        result = evaluation.validate_table(stale, periodic=True)
        self.assertFalse(result["exposed"].any())
        self.assertTrue((result["info"]["completed_landmark_id"] == 1).all())

    def test_hard_censor_cannot_keep_historical_ownership(self):
        native = replace(table(history=True), evaluation.ALPHA_COLUMNS[0], ["censored"] * 3)
        for name in evaluation.ALPHA_COLUMNS[1:]:
            native = replace(native, name, [np.nan] * 3)
        with self.assertRaises(ValueError):
            evaluation.validate_table(native, periodic=True)


class CellTest(unittest.TestCase):
    def evaluate(self, inputs):
        return evaluation.evaluate_cell(DAY, SYMBOL, *inputs)

    def test_pre_session_completed_fact_is_unique_support_but_not_session_completion(self):
        result = self.evaluate(full_cell(history=True))
        self.assertEqual(result["session_counter_at_start"], [1, 1, 0])
        self.assertEqual(result["session_completions"], 0)
        self.assertEqual(result["distinct_recent_sampled_landmarks"], 1)
        self.assertEqual(result["distinct_prior_session_recent_landmarks"], 1)
        self.assertEqual(result["distinct_within_session_recent_landmarks"], 0)
        self.assertEqual(result["support_rows"], 461)

    def test_repeated_same_landmark_is_not_five_distinct_landmarks(self):
        result = self.evaluate(full_cell(history=True))
        self.assertGreater(result["phase_counts_on_original30s"]["bilateral"], 5)
        self.assertEqual(result["distinct_recent_sampled_landmarks"], 1)

    def test_parent_origin_keys_and_mid_signed_zero_or_nan_payload_are_exact(self):
        periodic, parent, keys, events = full_cell()
        wrong_keys = replace(keys, "SampleBookSeq", np.arange(1381, dtype=np.int64) + 2)
        with self.assertRaises(ValueError):
            self.evaluate((periodic, parent, wrong_keys, events))
        for raw_bits, changed_bits in ((0x0000000000000000, 0x8000000000000000), (0x7FF8000000000001, 0x7FF8000000000002)):
            values = np.full(1381, raw_bits, dtype=np.uint64).view(np.float64)
            changed = np.full(1381, changed_bits, dtype=np.uint64).view(np.float64)
            native = replace(periodic, "OriginMidPrice", values)
            exact_parent, exact_keys = reference(native)
            self.evaluate((native, exact_parent, exact_keys, events))
            with self.assertRaises(ValueError):
                self.evaluate((native, replace(exact_parent, "OriginMidPrice", changed), exact_keys, events))

    def test_counter_disagreement_between_periodic_and_same_receive_event_is_rejected(self):
        periodic, parent, keys, events = full_cell(history=True)
        changed = replace(events, "H13_cumulative_quote_led_visit_count", np.full(1381, 2.0))
        with self.assertRaisesRegex(ValueError, "counter/source-phase"):
            self.evaluate((periodic, parent, keys, changed))

    def test_same_completed_id_cannot_change_frozen_strength(self):
        periodic, parent, keys, events = full_cell(history=True)
        changed = replace(events, evaluation.ALPHA_COLUMNS[3], np.full(1381, 0.3))
        with self.assertRaisesRegex(ValueError, "direction/strength changed"):
            self.evaluate((periodic, parent, keys, changed))

    def test_exact_origin_grid_is_not_recreated_or_extended(self):
        periodic, _, _, events = full_cell()
        shifted = replace(periodic, "SampleTime", periodic["SampleTime"].to_numpy() + 1)
        parent, keys = reference(shifted)
        with self.assertRaisesRegex(ValueError, "start/end10s"):
            self.evaluate((shifted, parent, keys, events))


class GatesTest(unittest.TestCase):
    @staticmethod
    def cells():
        return [
            {
                "day": day,
                "symbol": symbol,
                "status": "observed_native",
                "integrity_passed": True,
                "session_completions": 5,
                "session_up_completions": 3,
                "session_down_completions": 2,
                "distinct_recent_sampled_landmarks": 5,
                "provisional_or_unilateral_origins": 5,
            }
            for day, symbol in evaluation.FIXED_CELLS
        ]

    def test_gate_passes_only_fixed_cells_and_real_all_criteria(self):
        cells = self.cells()
        self.assertTrue(evaluation.gates(cells)["passed"])
        changed = copy.deepcopy(cells)
        changed[0].update(session_completions=4, session_up_completions=2, session_down_completions=2)
        self.assertFalse(evaluation.gates(changed)["passed"])
        for name in ("distinct_recent_sampled_landmarks", "provisional_or_unilateral_origins"):
            changed = copy.deepcopy(cells)
            changed[0][name] = 0
            self.assertFalse(evaluation.gates(changed)["passed"])
        changed = copy.deepcopy(cells)
        changed[0]["session_completions"] = 4
        with self.assertRaisesRegex(ValueError, "direction counts disagree"):
            evaluation.gates(changed)
        with self.assertRaises(ValueError):
            evaluation.gates(list(reversed(cells)))

    def test_skips_missing_fixed_inputs_without_replacements_or_vacuous_pass(self):
        cells = self.cells()
        for index in (1, 2, 3):
            cells[index] = {"day": cells[index]["day"], "symbol": cells[index]["symbol"], "status": "missing_fixed_input_skip"}
        self.assertFalse(evaluation.gates(cells)["passed"])
        cells[0] = {"day": cells[0]["day"], "symbol": cells[0]["symbol"], "status": "missing_fixed_input_skip"}
        result = evaluation.gates(cells)
        self.assertFalse(result["passed"])
        self.assertEqual(result["observed_cells"], 0)

    def test_counts_are_exact_nonnegative_native_ints_not_boolean_or_float_aliases(self):
        for value in (True, 5.0, 5.1, np.inf, -1):
            cells = self.cells()
            cells[0]["session_completions"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation.gates(cells)


if __name__ == "__main__":
    unittest.main()
