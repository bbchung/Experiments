"""Synthetic lifecycle and exact native-bit tests; no research reads or replay."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("h15_prefix_v2", Path(__file__).with_name("validate_receive_prefix.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def native_table(times):
    values = {}
    for name in module.KEYS:
        data = times if name != "SampleBookSeq" else list(range(1, len(times) + 1))
        values[name] = pa.array(data, type=pa.int64())
    categories = {f"PeerTradeInformation.0.{name}.0" for name in module.producer.ALPHA_CATEGORICAL}
    for name in module.producer.ALPHA:
        key = f"PeerTradeInformation.0.{name}.0"
        values[key] = pa.array(["quote"] * len(times), type=pa.string()) if key in categories else pa.array([0.0] * len(times), type=pa.float64())
    for name in module.producer.INFO:
        values[f"H15_{name}"] = pa.array([0.0] * len(times), type=pa.float64())
    for name in ("OriginMidPrice", "OriginMidTicks", "OriginBidTicks", "OriginAskTicks"):
        values[name] = pa.array([100.0] * len(times), type=pa.float64())
    values["H15_processed_peer_exchange_time"] = pa.array([100.0] * len(times))
    values["H15_processed_peer_available_time"] = pa.array([29_000_000.0] * len(times))
    # A native uncomputable payload must survive bit-for-bit, without pandas.
    bits = np.asarray([0x7FF8000000000011] * len(times), dtype=np.uint64).view(np.float64)
    values["PeerTradeInformation.0.target_known_work_imbalance.0"] = pa.array(bits, from_pandas=False)
    return pa.table(values)


class PrefixComparisonTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.v1, self.output = self.root / "v1", self.root / "v2"
        self.output.mkdir()
        self.patches = [patch.object(module, "OUTPUT", self.output), patch.object(module, "V1", self.v1)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.selected = {
            "R": 29_999_999,
            "S": 30_000_000,
            "D": 30_000_002,
            "A": 30_000_010,
            "pending_exchange": 200,
            "target": "2308",
            "peer": "2317",
            "leg": "peer",
            "origin_keys": {"SampleTime": 30_000_000, "SampleBookTime": 30_000_000, "SampleBookSeq": 1},
        }
        self.receipt = {
            "selected": self.selected,
            "selection_identity": "fixed-selection",
            "binary": {"sha256": module.producer.BINARY_SHA256},
            "command": ["fixed-coco", "fixed-config"],
        }
        self.receipt["prefix_outputs"], self.receipt["full_outputs"] = [], []
        for partition in ("data", "events"):
            for target in module.producer.TARGETS:
                original = self.root / "original" / partition / target / "values.parquet"
                original.parent.mkdir(parents=True)
                times = [30_000_000, 60_000_000] if partition == "data" else [60_000_000]
                table = native_table(times)
                pq.write_table(table, original)
                truncated = self.v1 / "truncated" / partition / module.prefix.DAY / target / "values.parquet"
                if partition == "data":
                    truncated.parent.mkdir(parents=True)
                    pq.write_table(table.slice(0, 1), truncated)
                self.receipt["full_outputs"].append(module.record(original))
                self.receipt["prefix_outputs"].append(str(truncated))
        source = self.root / "compiled-native.cpp"
        source.write_text("immutable source proof")
        source_input = self.root / "cut.bin.zst"
        source_input.write_bytes(b"immutable cut bytes")
        self.sources, self.inputs = [module.record(source)], [module.record(source_input), *self.receipt["full_outputs"]]
        self.parent = patch.object(module, "parent_context", side_effect=lambda: (self.receipt, {"identity": "old-comparison"}, self.sources, self.inputs))
        self.parent.start()
        self.addCleanup(self.parent.stop)
        (self.v1 / "truncated/status").mkdir(parents=True)
        module.write_yaml(self.v1 / "truncated/status" / f"{module.prefix.DAY}.yaml", {"status": "completed", "fatal_error": False})
        self.log = self.v1 / "truncated/native.log"
        self.log.write_text("completed native job")
        self.execution = {
            "method_identity": "old-comparison",
            "job": "receive_prefix",
            "exit_code": 0,
            "command": self.receipt["command"],
            "log": module.record(self.log),
            "outputs": [module.record(Path(path)) for path in self.receipt["prefix_outputs"][:2]],
        }
        self.execution_path = self.v1 / "truncated/execution-receipt.yaml"
        module.write_yaml(self.execution_path, self.execution)

    def freeze(self, *, omit_scope=None, omit_path=None):
        bound_path = self.output / "bound-comparison.yaml"
        if not bound_path.exists():
            module.bind_comparison()
        bound = module.read_yaml(bound_path)
        body = {
            "schema": module.METHOD_SCHEMA,
            "status": "frozen",
            "plan": module.record(bound_path),
            "sources": copy.deepcopy(bound["sources"]),
            "inputs": [*copy.deepcopy(bound["inputs"]), module.record(bound_path)],
        }
        if omit_scope:
            body[omit_scope] = [n for n in body[omit_scope] if n["path"] != omit_path]
        body["identity"] = module.digest(body)
        path = self.output / "frozen-comparison-method.yaml"
        module.write_yaml(path, body)
        path.with_suffix(".yaml.identity").write_text(body["identity"] + "\n")
        return path

    def test_legitimate_empty_absent_events_complete_without_creating_empty_files(self):
        method = self.freeze()
        result = module.check(method)
        self.assertEqual(result["present_native_tables"], 2)
        self.assertEqual(result["absent_empty_events"], 2)
        validation = module.read_yaml(self.output / "validation.yaml")
        self.assertEqual(validation["schema"], module.VALIDATION_SCHEMA)
        self.assertTrue(validation["common_origin_present"])
        self.assertTrue(validation["pending_cluster_not_published_at_origin"])
        for proof in validation["tables"][2:]:
            self.assertTrue(proof["empty_original_key_population_proven"])
            self.assertEqual(proof["common_rows"], 0)
            self.assertEqual(proof["field_bit_checks"], {})
        self.assertFalse(any(Path(path).exists() for path in self.receipt["prefix_outputs"][2:]))

    def test_bind_reads_only_original_keys_and_metadata_not_alpha_values(self):
        real_reader = module.pq.read_table
        calls = []

        def key_reader(path, **kwargs):
            self.assertEqual(kwargs.get("columns"), list(module.KEYS))
            self.assertIn(str(path), [n["path"] for n in self.receipt["full_outputs"]])
            calls.append(str(path))
            return real_reader(path, **kwargs)

        with patch.object(module.pq, "read_table", side_effect=key_reader):
            bound = module.bind_comparison()
        self.assertEqual(len(calls), 4)
        self.assertFalse(bound["native_values_read"])
        self.assertEqual([p["expected_key_rows"] for p in bound["partitions"]], [1, 1, 0, 0])

    def test_absent_event_with_nonempty_original_population_rejects_before_value_read(self):
        path = Path(self.receipt["full_outputs"][2]["path"])
        pq.write_table(native_table([30_000_000]), path)
        self.receipt["full_outputs"][2] = module.record(path)
        self.inputs = [self.inputs[0], *self.receipt["full_outputs"]]
        with self.assertRaisesRegex(ValueError, "nonempty original event"):
            module.bind_comparison()
        self.assertFalse((self.output / "bound-comparison.yaml").exists())

    def test_periodic_absence_always_rejects_even_if_original_population_is_empty(self):
        Path(self.receipt["prefix_outputs"][0]).unlink()
        path = Path(self.receipt["full_outputs"][0]["path"])
        pq.write_table(native_table([60_000_000]), path)
        self.receipt["full_outputs"][0] = module.record(path)
        self.inputs = [self.inputs[0], *self.receipt["full_outputs"]]
        self.execution["outputs"] = self.execution["outputs"][1:]
        module.write_yaml(self.execution_path, self.execution)
        with self.assertRaisesRegex(ValueError, "missing periodic"):
            module.bind_comparison()

    def test_forged_job_method_command_exit_type_and_output_membership_reject(self):
        variants = [
            {"method_identity": "new-comparison"},
            {"job": "different-job"},
            {"command": ["other-coco"]},
            {"exit_code": False},
            {"outputs": self.execution["outputs"][:1]},
            {"outputs": self.execution["outputs"] + [self.execution["outputs"][0]]},
            {"log": {**self.execution["log"], "path": str(self.root / "forged.log")}},
        ]
        for change in variants:
            module.write_yaml(self.execution_path, {**self.execution, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.bind_comparison()
        module.write_yaml(self.execution_path, self.execution)

    def test_changed_receipt_hash_or_log_cannot_be_bound_as_executed_output(self):
        changed = copy.deepcopy(self.execution)
        changed["outputs"][0]["sha256"] = "0" * 64
        module.write_yaml(self.execution_path, changed)
        with self.assertRaisesRegex(ValueError, "changed/missing bound artifact"):
            module.bind_comparison()
        module.write_yaml(self.execution_path, self.execution)
        self.log.write_text("changed native log")
        with self.assertRaisesRegex(ValueError, "changed/missing bound artifact"):
            module.bind_comparison()

    def test_absent_event_appearance_after_freeze_rejects_before_native_values(self):
        method = self.freeze()
        path = Path(self.receipt["prefix_outputs"][2])
        path.parent.mkdir(parents=True)
        pq.write_table(native_table([]), path)
        with self.assertRaisesRegex(ValueError, "execution output list"):
            module.check(method)
        self.assertFalse((self.output / "validation.yaml").exists())

    def test_present_artifact_byte_drift_and_original_key_drift_reject(self):
        method = self.freeze()
        path = Path(self.receipt["prefix_outputs"][0])
        path.write_bytes(path.read_bytes() + b"drift")
        with self.assertRaisesRegex(ValueError, "changed/missing bound artifact"):
            module.check(method)
        self.assertFalse((self.output / "validation.yaml").exists())

    def test_canonicalized_manifest_omission_cannot_drop_source_cut_or_execution(self):
        for scope, path in (
            ("sources", self.sources[0]["path"]),
            ("inputs", self.inputs[0]["path"]),
            ("inputs", str(self.execution_path)),
            ("inputs", self.receipt["full_outputs"][2]["path"]),
        ):
            method = self.freeze(omit_scope=scope, omit_path=path)
            with self.subTest(scope=scope, path=path), self.assertRaisesRegex(ValueError, "omitted mandatory"):
                module.check(method)
        self.assertFalse((self.output / "validation.yaml").exists())

    def test_signed_bound_cannot_lie_about_empty_key_population_or_presence(self):
        module.bind_comparison()
        bound_path = self.output / "bound-comparison.yaml"
        original = module.read_yaml(bound_path)
        for change in ({"expected_key_rows": 1}, {"prefix_present": True}, {"expected_keys_identity": "0" * 64}):
            bound = copy.deepcopy(original)
            bound["partitions"][2].update(change)
            module.write_yaml(bound_path, bound)
            method = self.freeze()
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "KEY-only absence/presence"):
                module.check(method)

    def test_selected_S_is_mandatory_and_pending_cluster_cannot_publish(self):
        for missing, published in ((True, False), (False, True)):
            table = native_table([30_000_000, 60_000_000])
            if missing:
                self.selected["origin_keys"]["SampleBookSeq"] = 99
            else:
                self.selected["origin_keys"]["SampleBookSeq"] = 1
                table = table.set_column(table.schema.get_field_index("H15_processed_peer_exchange_time"), "H15_processed_peer_exchange_time", pa.array([200.0, 200.0]))
                pq.write_table(table, self.receipt["full_outputs"][0]["path"])
                self.receipt["full_outputs"][0] = module.record(Path(self.receipt["full_outputs"][0]["path"]))
                pq.write_table(table.slice(0, 1), self.receipt["prefix_outputs"][0])
                self.inputs = [self.inputs[0], *self.receipt["full_outputs"]]
                self.execution["outputs"][0] = module.record(Path(self.receipt["prefix_outputs"][0]))
                module.write_yaml(self.execution_path, self.execution)
            (self.output / "bound-comparison.yaml").unlink(missing_ok=True)
            method = self.freeze()
            with self.subTest(missing=missing, published=published), self.assertRaises(ValueError):
                module.check(method)
        self.assertFalse((self.output / "validation.yaml").exists())

    def test_full_native_values_compare_every_field_bit_including_sentinel_payload(self):
        table = pq.read_table(self.receipt["prefix_outputs"][0])
        bits = np.asarray([0x7FF8000000000012], dtype=np.uint64).view(np.float64)
        name = "PeerTradeInformation.0.target_known_work_imbalance.0"
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(bits, from_pandas=False))
        pq.write_table(table, self.receipt["prefix_outputs"][0])
        self.execution["outputs"][0] = module.record(Path(self.receipt["prefix_outputs"][0]))
        module.write_yaml(self.execution_path, self.execution)
        method = self.freeze()
        with self.assertRaisesRegex(ValueError, "native common-origin bits changed"):
            module.check(method)
        self.assertFalse((self.output / "validation.yaml").exists())

    def test_absent_event_requires_native_full_schema_and_integer_keys(self):
        path = Path(self.receipt["full_outputs"][2]["path"])
        table = native_table([60_000_000]).set_column(0, "SampleTime", pa.array([60_000_000.0]))
        pq.write_table(table, path)
        self.receipt["full_outputs"][2] = module.record(path)
        self.inputs = [self.inputs[0], *self.receipt["full_outputs"]]
        with self.assertRaisesRegex(ValueError, "schema changed"):
            module.bind_comparison()


if __name__ == "__main__":
    unittest.main()
