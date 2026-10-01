"""Synthetic config/provenance failure modes; never prepare or replay real data."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("h15_native_v2_prepare", Path(__file__).with_name("prepare_native.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class NativeV2ConfigTest(unittest.TestCase):
    def original(self):
        return {"Users": [], "Modules": [{"Gid": s, "Decl": [{"Desc": "CurrentBook.0", "Spec": {}}]} for s in module.CARRIERS]}

    def test_both_filters_and_modules_require_valid_explicit_tautology(self):
        original = self.original()
        config = module.daily_config(original, list(module.TARGETS))
        module.validate_daily_config(config, original, list(module.TARGETS))
        selected = [d for g in config["Modules"] for d in g["Decl"] if d["Desc"] in ("TwseFilter.0", "PeerTradeInformation.0")]
        self.assertEqual(len(selected), 4)
        self.assertEqual({d["Spec"]["StatusFilter"] for d in selected}, {"TRIAL || !TRIAL"})
        for invalid in ("", "TRIAL", "!TRIAL", None):
            with self.subTest(expression=invalid):
                modified = copy.deepcopy(config)
                changed = next(d for g in modified["Modules"] for d in g["Decl"] if d["Desc"] == "TwseFilter.0")
                changed["Spec"]["StatusFilter"] = invalid
                with self.assertRaises(ValueError):
                    module.validate_daily_config(modified, original, list(module.TARGETS))

    def test_sampling_and_exports_cannot_drift_with_grammar_repair(self):
        original = self.original()
        for field, value in (("Exports", []), ("PeriodicSampler", {"SampleInterval": "30s"}), ("Labelers", ["future_return"])):
            config = module.daily_config(original, list(module.TARGETS))
            writer = next(d for g in config["Modules"] for d in g["Decl"] if d["Desc"] == "DatasetWriter.0")
            writer["Spec"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_daily_config(config, original, list(module.TARGETS))

    def test_missing_target_writer_still_keeps_both_source_filters(self):
        config = module.daily_config(self.original(), ["2308"])
        module.validate_daily_config(config, self.original(), ["2308"])
        self.assertEqual([g["Gid"] for g in config["Modules"]], list(module.CARRIERS))
        self.assertEqual(sum(d["Desc"] == "TwseFilter.0" for g in config["Modules"] for d in g["Decl"]), 2)
        self.assertEqual(sum(d["Desc"] == "PeerTradeInformation.0" for g in config["Modules"] for d in g["Decl"]), 1)

    def test_accept_all_masks_are_exact_for_every_possible_native_trial_table(self):
        self.assertEqual(module.ACCEPT_ALL, "TRIAL || !TRIAL")
        for truth_table in range(1 << 16):
            self.assertEqual(truth_table | ((~truth_table) & 0xFFFF), 0xFFFF)


class NativeV2ReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def node(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return module.record(path)

    def fixture(self):
        outputs = [self.node(f"control/out{i}", f"opaque-control-{i}") for i in range(4)]
        expected = [self.node(f"old/out{i}", f"opaque-control-{i}") for i in range(4)]
        config = self.node("control/config.yaml", "old-config")
        old_config = self.node("old/config.yaml", "old-config")
        old_binary = self.node("old/coco", "old-binary")
        command = ["/immutable/coco", "-d", "20260119"]
        original = {
            "work": str(self.root / "control"),
            "config": config,
            "old_config": old_config,
            "old_binary": old_binary,
            "command": command,
            "requires_replay": True,
            "outputs": [n["path"] for n in outputs],
            "required_byte_equal_outputs": expected,
        }
        receipt = {
            "method_identity": module.V1_IDENTITY,
            "job": "registration_control",
            "exit_code": 0,
            "command": command,
            "outputs": outputs,
            "log": self.node("control/native.log", "successful-control"),
        }
        path = self.root / "control/execution-receipt.yaml"
        module.write_yaml(path, receipt)
        failure = {"registration_control": {"whole_four_h13_parquet_bytes_equal": True, "execution": module.record(path)}}
        return {"registration_control": original}, failure, receipt, path

    def test_successful_existing_control_is_bound_but_has_no_replay_command(self):
        preparation, failure, _, _ = self.fixture()
        control, links = module.reused_control(preparation, failure)
        self.assertIs(control["requires_replay"], False)
        self.assertIs(control["reuse_existing_outputs"], True)
        self.assertEqual(control["command"], [])
        self.assertEqual(control["reused_command"], preparation["registration_control"]["command"])
        self.assertEqual(control["outputs"], preparation["registration_control"]["outputs"])
        self.assertTrue(all(n in links for n in preparation["registration_control"]["required_byte_equal_outputs"]))

    def test_resigned_changed_control_receipt_cannot_claim_success(self):
        for field, value in (("exit_code", 1), ("method_identity", "0" * 64), ("outputs", []), ("command", ["/other/coco"])):
            with self.subTest(field=field):
                preparation, failure, receipt, path = self.fixture()
                receipt[field] = value
                module.write_yaml(path, receipt)
                failure["registration_control"]["execution"] = module.record(path)
                with self.assertRaises(ValueError):
                    module.reused_control(preparation, failure)

    def test_changed_control_bytes_fail_even_when_execution_receipt_is_resigned(self):
        preparation, failure, receipt, path = self.fixture()
        receipt["outputs"][0] = self.node("control/out0", "different-control")
        module.write_yaml(path, receipt)
        failure["registration_control"]["execution"] = module.record(path)
        with self.assertRaises(ValueError):
            module.reused_control(preparation, failure)

    def test_canonical_identity_does_not_allow_mandatory_evidence_omission(self):
        mandatory = [self.node(name, name) for name in ("failure.yaml", "v1-method.yaml", "v1-method.yaml.identity", "new-profile.yaml", "grammar.cpp")]
        for omitted in mandatory:
            body = {"schema": "h15-native-producer-method-v2", "status": "frozen", "sources": [], "inputs": [n for n in mandatory if n != omitted]}
            body["identity"] = module.digest(body)
            path = self.root / "new-frozen.yaml"
            module.write_yaml(path, body)
            path.with_suffix(".yaml.identity").write_text(body["identity"])
            verified = module.canonical_method(path, "h15-native-producer-method-v2")
            with self.subTest(omitted=omitted["path"]), self.assertRaises(ValueError):
                module.require_closure(verified, "inputs", mandatory)

    def test_compiled_grammar_proof_cannot_be_moved_to_input_scope(self):
        node = self.node("compiled-parser.cpp", "immutable-grammar")
        with self.assertRaises(ValueError):
            module.require_closure({"sources": [], "inputs": [node]}, "sources", [node])

    def test_profile_repair_does_not_authorize_support_threshold_drift(self):
        original = {"contract": copy.deepcopy(module.CONTRACT), "fixed_train_days": list(module.DAYS), "native_plan": {"path": "old"}, "routing": "old"}
        with patch.object(module, "record", side_effect=lambda p: {"path": str(p), "sha256": "0" * 64}):
            profile = copy.deepcopy(original)
            profile.update(
                schema="h15-native-source-support-profile-v2",
                status="prospective_requires_root_freeze",
                status_filter_expression=module.ACCEPT_ALL,
                shared_h15_v1_producer_identity=module.V1_IDENTITY,
                native_plan=module.record(module.NATIVE_PLAN),
                config_method_plan=module.record(module.PLAN),
            )
            module.validate_profile(profile, original)
            profile["contract"]["minimum_ordered_response_support_origins"] = 19
            with self.assertRaises(ValueError):
                module.validate_profile(profile, original)

    def test_config_only_failure_cannot_hide_existing_feature_outputs(self):
        failure = {
            "schema": "h15-native-startup-failure-v1",
            "producer_identity": module.V1_IDENTITY,
            "binary_sha256": module.BINARY_SHA256,
            "frozen_old_artifacts_modified": False,
            "labels_read": False,
            "model_fits": 0,
            "native_h15_output_artifacts": 0,
            "native_h15_outputs_decoded": False,
            "remaining_day_not_launched": module.DAYS[1],
        }
        log = self.node("failed/native.log", "invalid StatusFilter expression '' at offset 0")
        execution = {"method_identity": module.V1_IDENTITY, "job": module.DAYS[0], "exit_code": 1, "outputs": [], "command": ["old-coco"], "log": log}
        path = self.root / "failed/execution.yaml"
        module.write_yaml(path, execution)
        failure["startup_failure"] = {
            "day": module.DAYS[0],
            "error": "invalid StatusFilter expression empty string",
            "execution": module.record(path),
            "log": log,
            "status": self.node("failed/status.yaml", "failed"),
        }
        preparation = {"jobs": [{"day": module.DAYS[0], "command": ["old-coco"]}]}
        module.validate_failed_startup(failure, preparation)
        execution["outputs"] = [self.node("failed/feature-output", "opaque-output")]
        module.write_yaml(path, execution)
        failure["startup_failure"]["execution"] = module.record(path)
        with self.assertRaises(ValueError):
            module.validate_failed_startup(failure, preparation)

    def test_claimed_empty_parser_failure_requires_actual_bound_log_evidence(self):
        failure = {
            "schema": "h15-native-startup-failure-v1",
            "producer_identity": module.V1_IDENTITY,
            "binary_sha256": module.BINARY_SHA256,
            "frozen_old_artifacts_modified": False,
            "labels_read": False,
            "model_fits": 0,
            "native_h15_output_artifacts": 0,
            "native_h15_outputs_decoded": False,
            "remaining_day_not_launched": module.DAYS[1],
        }
        log = self.node("failed/native.log", "different native error")
        execution = {"method_identity": module.V1_IDENTITY, "job": module.DAYS[0], "exit_code": 1, "outputs": [], "command": ["old-coco"], "log": log}
        path = self.root / "failed/execution.yaml"
        module.write_yaml(path, execution)
        failure["startup_failure"] = {
            "day": module.DAYS[0],
            "error": "invalid StatusFilter expression empty string",
            "execution": module.record(path),
            "log": log,
            "status": self.node("failed/status.yaml", "failed"),
        }
        with self.assertRaises(ValueError):
            module.validate_failed_startup(failure, {"jobs": [{"day": module.DAYS[0], "command": ["old-coco"]}]})


if __name__ == "__main__":
    unittest.main()
