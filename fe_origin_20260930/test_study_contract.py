"""Check research population/domain contracts using small Parquet fixtures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from mechanism import permute_native_finite
from study import compact_arms
from study_data import HORIZONS, native_classes, nonoverlap, numeric_domain, project, project_nominal

from AstraResearch.contracts import ArtifactRef
from AstraResearch.io import ContractError, read_yaml
from AstraResearch.material import KEYS


class StudyContractTest(unittest.TestCase):
    def test_absent_extension_cannot_create_an_identical_nomination_candidate(self):
        arms = {
            "current": (["a", "b"], "classifier"),
            "current_regression": (["a", "b"], "regressor"),
            "current_binary": (["a", "b"], "binary"),
            "current_equal_day": (["a", "b"], "classifier"),
            "current_cross": (["b", "a"], "classifier"),
            "current_flow": (["a", "b", "flow"], "classifier"),
            "current_flow_cross": (["a", "b", "flow"], "classifier"),
            "current_flow_permuted_control": (["a", "b", "flow"], "classifier"),
            "current_nominal": (["a", "b"], "classifier"),
        }
        retained, aliases = compact_arms(arms, {"current_nominal"}, ["native_side"])
        self.assertEqual(aliases, {"current_cross": "current", "current_flow_cross": "current_flow"})
        self.assertEqual(set(retained), set(arms) - set(aliases))
        self.assertIn("current_equal_day", retained)
        self.assertIn("current_flow_permuted_control", retained)
        self.assertIn("current_nominal", retained)

    def fixture(self, directory):
        count = 105
        table = pd.DataFrame(
            {
                "good": np.arange(count, dtype=float),
                "special": np.r_[np.arange(count - 3, dtype=float), -np.inf, np.nan, np.inf],
                "overflow": np.r_[np.arange(count - 1, dtype=float), 1e100],
                "constant": np.ones(count),
                "nominal": ["bid", "ask", "none"] * 35,
                "SampleTime": np.arange(count, dtype=np.int64) * 10_000_000 + 100_000_000,
                "SampleBookTime": np.arange(count, dtype=np.int64) * 10_000_000 + 100_000_000,
                "SampleBookSeq": np.arange(count, dtype=np.int64),
            }
        )
        for horizon in HORIZONS:
            table[f"mid_return_ticks[{horizon}s]"] = np.resize([-5.0, 0.0, 5.0], count)
            table[f"mid_endpoint.up.5[{horizon}s]"] = np.resize([0.0, 0.0, 1.0], count)
            table[f"mid_endpoint.down.5[{horizon}s]"] = np.resize([1.0, 0.0, 0.0], count)
        table.to_parquet(directory / "values.parquet", index=False)
        dataset = SimpleNamespace(
            ref=ArtifactRef("feature_dataset", "a" * 64),
            metadata={
                "role": "train",
                "partitions": [{"day": "20260102", "symbol": "2330", "path": "values.parquet"}],
                "feature_columns": ["good", "special", "overflow", "constant", "nominal"],
                "label": "mid_return_ticks[300s]",
            },
            file=lambda source: directory / source["path"],
        )
        return dataset, table

    def test_train_domain_excludes_whole_column_preserves_every_origin_and_sentinel(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            dataset, original = self.fixture(directory)
            features = numeric_domain(dataset, ["good", "special", "overflow", "constant"], directory / "domain")
            self.assertEqual(features, ["good", "special"])
            report = read_yaml(directory / "domain/numeric-domain.yaml")
            self.assertEqual(report["excluded_overflow"]["overflow"]["train_rows"], 1)
            matrix, rows, _ = project(dataset, features, directory / "projection")
            self.assertEqual(len(matrix), len(original))
            self.assertTrue(rows[KEYS].equals(original[KEYS]))
            self.assertTrue(np.isneginf(matrix[-3, 1]))
            self.assertTrue(np.isnan(matrix[-2, 1]))
            self.assertTrue(np.isposinf(matrix[-1, 1]))
            cats = project_nominal(dataset, ["nominal"], directory / "projection", rows)
            self.assertEqual(cats.nominal.tolist(), original.nominal.tolist())
            self.assertEqual(pd.read_parquet(directory / "values.parquet").overflow.iloc[-1], 1e100)
            corrupted = np.load(directory / "projection/x.npy", mmap_mode="r+")
            corrupted[0, 0] = 999
            corrupted.flush()
            del corrupted
            with self.assertRaisesRegex(ContractError, "another native recipe"):
                project(dataset, features, directory / "projection")

    def test_future_numeric_domain_cannot_reselect_or_hide_new_overflow(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            dataset, table = self.fixture(directory)
            features = numeric_domain(dataset, ["good"], directory / "domain")
            dataset.metadata["role"] = "forward"
            with self.assertRaisesRegex(ContractError, "only inspect train"):
                numeric_domain(dataset, ["good"], directory / "forward-domain")
            table.loc[0, "good"] = 1e100
            table.to_parquet(directory / "values.parquet", index=False)
            with self.assertRaisesRegex(ContractError, "train-frozen"):
                project(dataset, features, directory / "forward")

    def test_native_endpoint_codebook_requires_exact_side_and_availability(self):
        rows = pd.DataFrame(
            {
                "mid_return_ticks[300s]": [-5.0, -4.9, 0.0, 4.9, 5.0, np.nan],
                "mid_endpoint.up.5[300s]": [0.0, 0.0, 0.0, 0.0, 1.0, np.nan],
                "mid_endpoint.down.5[300s]": [1.0, 0.0, 0.0, 0.0, 0.0, np.nan],
            }
        )
        classes, known = native_classes(rows, 300)
        np.testing.assert_array_equal(classes, [2, 0, 0, 0, 1, -1])
        np.testing.assert_array_equal(known, [True] * 5 + [False])
        rows.loc[4, "mid_endpoint.down.5[300s]"] = 1.0
        with self.assertRaisesRegex(ContractError, "disagree"):
            native_classes(rows, 300)
        rows.loc[4, "mid_endpoint.down.5[300s]"] = np.nan
        with self.assertRaisesRegex(ContractError, "states differ"):
            native_classes(rows, 300)

    def test_nonoverlap_keeps_original_time_slots_without_using_outcomes(self):
        times = np.array([100, 110, 130, 140, 100, 110, 130, 140], dtype=np.int64) * 1_000_000
        day = np.array(["20260102"] * 8)
        symbol = np.array(["2330"] * 4 + ["2317"] * 4)
        np.testing.assert_array_equal(nonoverlap(day, symbol, times, 30), [True, False, True, False] * 2)

    def test_value_permutation_retains_availability_and_native_joint_values(self):
        rows = pd.DataFrame({"day": ["20260102"] * 8, "symbol": ["2330"] * 4 + ["2317"] * 4})
        values = np.array([[1, 10], [2, 20], [-np.inf, np.nan], [3, 30], [4, 40], [5, 50], [np.inf, 60], [6, 60]], dtype=np.float32)
        result = permute_native_finite(values, rows)
        np.testing.assert_array_equal(result, permute_native_finite(values, rows))
        np.testing.assert_array_equal(np.isfinite(result), np.isfinite(values))
        np.testing.assert_array_equal(np.isnan(result), np.isnan(values))
        np.testing.assert_array_equal(result[[2, 6]], values[[2, 6]])
        for indices in ([0, 1, 3], [4, 5, 7]):
            self.assertEqual(sorted(map(tuple, result[indices])), sorted(map(tuple, values[indices])))


if __name__ == "__main__":
    unittest.main()
