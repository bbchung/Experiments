"""Check the boundary between study evidence and actual native model admission."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, FeaturesData, Pool
from study_models import admission, binary_heads

from AstraResearch.io import ContractError, digest, file_hash, read_yaml, write_yaml


class StudyModelsTest(unittest.TestCase):
    def fixture(self, directory):
        protocol = {
            "arms": {"binary": {"features": ["native_numeric"], "kind": "binary"}, "three_class": {"features": ["native_numeric"], "kind": "classifier"}},
            "primary_horizon": 300,
            "secondary_horizons": [60, 120, 180],
            "native_nominal_arms": ["binary"],
            "native_nominal_features": ["native_nominal"],
            "material_bindings": {"train": {"kind": "feature_dataset", "identity": "a" * 64}, "tune": {"kind": "feature_dataset", "identity": "b" * 64}},
        }
        write_yaml(directory / "protocol.yaml", protocol)
        features = ["native_numeric", "native_nominal"]
        values = FeaturesData(
            num_feature_data=np.arange(12, dtype=np.float32).reshape(-1, 1),
            cat_feature_data=np.array([["a"], ["b"]] * 6, dtype=object),
            num_feature_names=features[:1],
            cat_feature_names=features[1:],
        )
        for side in ("up", "down"):
            stem = f"binary-300-{side}"
            CatBoostClassifier(iterations=2, depth=2, thread_count=1, verbose=False, allow_writing_files=False).fit(Pool(values, label=[0, 1] * 6)).save_model(
                str(directory / f"{stem}.cbm")
            )
            label = f"mid_endpoint.{side}.5[300s]"
            write_yaml(
                directory / f"{stem}-fit.yaml",
                {
                    "recipe": {"protocol": digest(protocol), "horizon": 300, "arm": "binary", "native_endpoint_side": side},
                    "model_features": features,
                    "feature_digest": digest(features),
                    "loss": "Logloss",
                    "native_label": label,
                    "train_dataset": protocol["material_bindings"]["train"],
                    "tune_dataset": protocol["material_bindings"]["tune"],
                },
            )
            write_yaml(
                directory / f"{stem}-model-info.yaml",
                {
                    "model": {"kind": "classifier"},
                    "features": features,
                    "categorical_features": features[1:],
                    "target": {
                        "column": label,
                        "side": side,
                        "horizon_seconds": 300,
                        "contract": {"kind": "endpoint_tail", "large_move_ticks": 5, "costs": "excluded"},
                    },
                },
            )
        return protocol

    def test_actual_binary_input_order_and_target_metadata_are_checked(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            self.fixture(directory)
            _, numeric, nominal, models, _, _ = binary_heads(directory, "binary", 300)
            self.assertEqual(numeric, ["native_numeric"])
            self.assertEqual(nominal, ["native_nominal"])
            self.assertEqual(set(models), {"up", "down"})
            path = directory / "binary-300-down-model-info.yaml"
            info = read_yaml(path)
            info["target"]["side"] = "up"
            write_yaml(path, info)
            with self.assertRaisesRegex(ContractError, "Model-info"):
                binary_heads(directory, "binary", 300)

    def test_three_class_and_negative_controls_cannot_use_native_binary_route(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            protocol = self.fixture(directory)
            protocol["arms"]["binary_permuted_control"] = protocol["arms"]["binary"]
            write_yaml(directory / "protocol.yaml", protocol)
            for arm in ("three_class", "binary_permuted_control"):
                with self.assertRaisesRegex(ContractError, "ordinary.*binary"):
                    binary_heads(directory, arm, 300)

    def test_parity_alone_or_different_model_bytes_cannot_admit_a_baseline(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            protocol = self.fixture(directory)
            _, _, _, _, model_files, _ = binary_heads(directory, "binary", 300)
            write_yaml(directory / "completed.yaml", {"completed": True})
            write_yaml(directory / "primary-decision.yaml", {"challenger": "current", "predictive_gain_supported": False})
            decision = {"predictive_gain_supported": False, "checks": {"positive_daily_AP_bound": False}}
            write_yaml(directory / "objective-decisions.yaml", {"binary": decision})
            parity = {
                "mode": "predictor",
                "study_arm": "binary",
                "study_horizon": 300,
                "study_protocol_digest": digest(protocol),
                "model_sha256": {side: file_hash(path) for side, path in model_files.items()},
                "symbols": {"2330": {"directional_classifier": {"native_score_parity": True, "feature_bitwise_mismatches": {}, "native_label_bitwise_mismatches": {}}}},
            }
            with self.assertRaisesRegex(ContractError, "did not support"):
                admission(directory, "binary", parity, model_files)
            decision.update(predictive_gain_supported=True, checks={"positive_daily_AP_bound": True})
            write_yaml(directory / "objective-decisions.yaml", {"binary": decision})
            self.assertEqual(admission(directory, "binary", parity, model_files), decision)
            parity["model_sha256"]["up"] = "c" * 64
            with self.assertRaisesRegex(ContractError, "exact fitted models"):
                admission(directory, "binary", parity, model_files)


if __name__ == "__main__":
    unittest.main()
