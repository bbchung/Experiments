"""Synthetic ABI/prefix/provenance tests; no actual data or replay."""

import copy
import ctypes as ct
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

spec = importlib.util.spec_from_file_location("h16_receive_prefix", Path(__file__).with_name("receive_prefix.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def native_record(symbol, receive, exchange, *, sequence=1, trade=False):
    header = module.reader.Header()
    header.seq, header.time, header.type = sequence, receive, b"T" if trade else b"B"
    payload = module.reader.Trade() if trade else module.reader.Book()
    payload.exchange, payload.symbol = exchange, symbol.encode()
    if trade:
        payload.quantity, payload.price, payload.status = 0, 0, 3
    else:
        payload.bid_depth = payload.ask_depth = 1
        payload.bid[0], payload.ask[0] = 100, 100.5
        payload.bid_qty[0] = payload.ask_qty[0] = 1
    return bytes(header) + bytes(payload)


def table_fixture():
    sample = np.array([90_000_000, 100_000_000, 110_000_000], dtype=np.int64)
    fields, arrays = [], []
    for name in module.pilot.COLUMNS:
        dtype = pa.int64() if name in module.KEYS else pa.string() if name in module.pilot.CAT_COLUMNS else pa.float64()
        role = b"time" if name == "SampleTime" else b"context" if name in module.KEYS else b"feature" if name in module.pilot.ALPHA_COLUMNS else b"metadata"
        values = (
            sample
            if name == "SampleTime"
            else sample - 1
            if name == "SampleBookTime"
            else np.arange(3)
            if name == "SampleBookSeq"
            else ["warmup"] * 3
            if name in module.pilot.CAT_COLUMNS
            else np.zeros(3)
        )
        if name == module.pilot.NUMERIC_COLUMNS[0]:
            values = np.array([0x7FF8000000000001, 0xFFF0000000000000, 0], dtype=np.uint64).view(np.float64)
        fields.append(pa.field(name, dtype, metadata={b"coco.role": role}))
        arrays.append(pa.array(values, type=dtype))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


class NativePrefixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(module.scan_index.cache_clear)
        self.addCleanup(module.cut_content_identity.cache_clear)
        self.addCleanup(module.source_prefix_hash.cache_clear)

    def spool(self, symbol, records):
        path = self.root / "spool" / f"{symbol}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(records))
        source = {"path": str(self.root / "original" / symbol / "20260119.bin.zst"), "sha256": "a" * 64}
        return module.scan_spool(module.record(path), symbol, source)

    def scans(self):
        scans = {}
        for symbol in module.CARRIERS:
            next_r = 110_000_000 if symbol == module.TARGET else 95_000_000 if symbol == "2330" else 120_000_000
            scans[symbol] = self.spool(symbol, [native_record(symbol, 80_000_000, 500_000_000), native_record(symbol, next_r, 500_000_001, sequence=2, trade=True)])
        return scans

    def test_native_abi_raw_exchange_and_zero_trade_are_preserved(self):
        self.assertEqual(module.reader.abi()["header_bytes"], 24)
        self.assertEqual(ct.sizeof(module.reader.Book), 152)
        self.assertEqual(ct.sizeof(module.reader.Trade), 56)
        data = native_record(module.TARGET, 80, 8_000, trade=True)
        row = module.reader.read_record(io.BytesIO(data), module.TARGET, 0, 1)
        self.assertEqual(row["exchange"], 8_000)
        self.assertEqual(row["bytes"], data)

    def test_truncated_header_payload_and_wrong_symbol_reject(self):
        data = native_record(module.TARGET, 80, 500)
        for invalid in (data[:8], data[:-1], native_record("2330", 80, 500)):
            with self.subTest(size=len(invalid)), self.assertRaises(ValueError):
                module.reader.read_record(io.BytesIO(invalid), module.TARGET, 0, 1)

    def test_first_strict_case_has_no_trial_quantity_eligibility(self):
        selected = module.select(self.scans(), [(90_000_000, 80_000_000, 1), (120_000_000, 110_000_000, 2)], 90_000_000)
        self.assertEqual((selected["R"], selected["S"], selected["D"], selected["A"]), (80_000_000, 90_000_000, 95_000_000, 110_000_000))
        self.assertEqual(selected["carrier_symbol"], "2330")
        self.assertEqual(selected["carrier_record"]["type"], "T")

    def test_no_global_before_target_is_unsupported_without_substitution(self):
        scans = self.scans()
        scans["2330"] = self.spool("2330", [native_record("2330", 80_000_000, 1), native_record("2330", 120_000_000, 2)])
        self.assertIsNone(module.select(scans, [(90_000_000, 80_000_000, 1)], 90_000_000))

    def test_native_tie_order_precedes_subscription_order(self):
        trade = {"receive": 95, "seq": 2, "type": "T"}
        book = {"receive": 95, "seq": 2, "type": "B"}
        source = {"path": "/fixed/source"}
        self.assertLess(module.native_order("3711", trade, source), module.native_order("2330", book, source))
        self.assertLess(module.native_order("2330", trade, source), module.native_order("3711", trade, source))
        self.assertLess(module.native_order("3711", {**book, "seq": 1}, source), module.native_order("2330", trade, source))

    def test_receive_regression_and_sameR_cut_preserve_all_raw_bytes(self):
        records = [
            native_record(module.TARGET, 80, 100, sequence=1),
            native_record(module.TARGET, 92, 101, sequence=2, trade=True),
            native_record(module.TARGET, 90, 102, sequence=3),
            native_record(module.TARGET, 95, 103, sequence=4),
        ]
        scan = self.spool(module.TARGET, records)
        self.assertEqual(list(scan["times"]), [80, 92, 92, 95])
        cut = module.reader.write_cut(scan, 92, self.root / "cut.bin.zst")
        size, actual_hash = module.cut_content_identity(cut["path"], cut["sha256"])
        expected = b"".join(records[:3])
        self.assertEqual(size, len(expected))
        self.assertEqual(actual_hash, module.hashlib.sha256(expected).hexdigest())
        self.assertEqual(cut["records"], 3)
        self.assertEqual(module.header_at(scan, 2)["raw_receive"], 90)
        self.assertEqual(module.header_at(scan, 2)["receive"], 92)

    def test_spool_index_shared_and_eof_creates_no_record(self):
        data = native_record(module.TARGET, 80, 100)
        scan = self.spool(module.TARGET, [data])
        before = module.scan_index.cache_info()
        again = module.scan_spool(scan["spool"], module.TARGET, scan["source"])
        self.assertEqual(module.scan_index.cache_info().hits, before.hits + 1)
        self.assertEqual(again["records"], 1)
        self.assertEqual(again["uncompressed_bytes"], len(data))
        self.assertIsNone(module.select({symbol: {**scan, "symbol": symbol} for symbol in module.CARRIERS}, [(90_000_000, 80, 1)], 90_000_000))

    def test_all36_bits_strings_schemas_and_exact_keys_compare(self):
        full = table_fixture()
        common = full.slice(0, 2)
        proof = {
            "schema_identity": module.schema_identity(full.schema),
            "expected_common_rows": 2,
            "expected_common_keys_identity": module.digest(module.reader.native_keys(common)),
        }
        result = module.compare_tables(full, common, proof, 100_000_000)
        self.assertEqual(len(result["field_bit_checks"]), 36)
        self.assertTrue(all(result["field_bit_checks"].values()))
        name = module.pilot.NUMERIC_COLUMNS[0]
        values = np.array([0x7FF8000000000002, 0xFFF0000000000000], dtype=np.uint64).view(np.float64)
        altered = common.set_column(common.schema.get_field_index(name), common.schema.field(name), pa.array(values))
        with self.assertRaisesRegex(ValueError, "bits changed"):
            module.compare_tables(full, altered, proof, 100_000_000)
        name = module.pilot.CAT_COLUMNS[0]
        altered = common.set_column(common.schema.get_field_index(name), common.schema.field(name), pa.array(["warmup", "indicative"]))
        with self.assertRaisesRegex(ValueError, "bits changed"):
            module.compare_tables(full, altered, proof, 100_000_000)

    def test_negative_zero_dtypes_roles_and_orphaned_keys_reject(self):
        full = table_fixture()
        common = full.slice(0, 2)
        proof = {
            "schema_identity": module.schema_identity(full.schema),
            "expected_common_rows": 2,
            "expected_common_keys_identity": module.digest(module.reader.native_keys(common)),
        }
        with self.assertRaisesRegex(ValueError, "invented/dropped/reordered"):
            module.compare_tables(full, common.slice(0, 1), proof, 100_000_000)
        name = module.pilot.NUMERIC_COLUMNS[1]
        altered = common.set_column(common.schema.get_field_index(name), common.schema.field(name), pa.array([-0.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "bits changed"):
            module.compare_tables(full, altered, proof, 100_000_000)
        field = pa.field(name, pa.float32(), metadata={b"coco.role": b"feature"})
        altered = common.set_column(common.schema.get_field_index(name), field, common[name].cast(pa.float32()))
        with self.assertRaisesRegex(ValueError, "dtype/semantic"):
            module.schema_identity(altered.schema)
        name = "H16_cycle_support"
        field = pa.field(name, pa.float64(), metadata={b"coco.role": b"feature"})
        altered = common.set_column(common.schema.get_field_index(name), field, common[name])
        with self.assertRaisesRegex(ValueError, "dtype/semantic"):
            module.schema_identity(altered.schema)

    def test_absence_requires_frozen_original_zero_key_count(self):
        self.assertTrue(module.absence_allowed({"absence_allowed": True, "expected_common_rows": 0}, False))
        for proof in (
            {"absence_allowed": True, "expected_common_rows": 1},
            {"absence_allowed": False, "expected_common_rows": 0},
            {"absence_allowed": True, "expected_common_rows": False},
        ):
            with self.subTest(proof=proof), self.assertRaises(ValueError):
                module.absence_allowed(proof, False)

    def test_selection_bind_cannot_read_headers_keys_or_values(self):
        context = {"sources": [], "inputs": [], "producer_identity": "a" * 64}
        with (
            patch.object(module, "OUTPUT", self.root),
            patch.object(module, "parent_context", return_value=context),
            patch.object(module.pq, "read_table", side_effect=AssertionError("no table before freeze")),
            patch.object(module, "scan_spool", side_effect=AssertionError("no header before freeze")),
        ):
            payload = module.bind_selection("producer", "support")
        self.assertEqual(payload["schema"], module.SELECTION_SCHEMA)
        bound = module.read_yaml(self.root / "bound-selection.yaml")
        self.assertIs(bound["raw_headers_read"], False)
        self.assertIs(bound["original_keys_read"], False)
        self.assertIs(bound["native_values_read"], False)

    def test_key_proofs_read_only_original_keys_and_schema_before_values(self):
        full = table_fixture()
        nodes = []
        for symbol in module.CARRIERS:
            path = self.root / "full" / symbol / "values.parquet"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"opaque native; never decoded")
            nodes.append(module.record(path))

        def key_only(path, *, columns, use_threads):
            self.assertEqual(columns, list(module.KEYS))
            self.assertIs(use_threads, False)
            return full.select(module.KEYS)

        with (
            patch.object(module, "OUTPUT", self.root),
            patch.object(module.pq, "read_schema", return_value=full.schema),
            patch.object(module.pq, "read_table", side_effect=key_only),
        ):
            proofs = module.key_proofs(nodes, 95_000_000)
        self.assertEqual(len(proofs), 16)
        self.assertTrue(all(proof["expected_common_rows"] == 1 and not proof["absence_allowed"] for proof in proofs))
        self.assertEqual(len({proof["schema_identity"] for proof in proofs}), 1)

    def test_canonical_selection_manifest_cannot_omit_science_or_raw_inputs(self):
        kernel = self.root / "kernel.py"
        raw = self.root / "original.bin.zst"
        kernel.write_text("science")
        raw.write_text("opaque raw")
        node, source = module.record(raw), module.record(kernel)
        context = {"producer_method": node, "support_method": node, "support_receipt": node, "sources": [source], "inputs": [node]}
        with patch.object(module, "OUTPUT", self.root), patch.object(module, "parent_context", return_value=context):
            payload = module.bind_selection("producer", "support")
            for scope, omitted in (("sources", source), ("inputs", node)):
                changed = copy.deepcopy(payload)
                changed[scope] = [item for item in changed[scope] if item != omitted]
                path = self.root / f"{scope}.yaml"
                module.freeze_payload(changed, path)
                with self.subTest(scope=scope), self.assertRaisesRegex(ValueError, "mandatory"):
                    module.verify_selection(path)

    def test_all16_cut_content_validation_rejects_same_length_raw_byte_change(self):
        scans = self.scans()
        with patch.object(module, "OUTPUT", self.root):
            cuts = [module.reader.write_cut(scans[symbol], 95_000_000, self.root / "raw" / symbol / f"{module.DAY}.bin.zst") for symbol in module.CARRIERS]
            module.validate_cuts(cuts, scans, 95_000_000)
            cut = cuts[0]
            original = Path(scans[cut["symbol"]]["spool"]["path"]).read_bytes()[: cut["uncompressed_prefix_bytes"]]
            changed = original[:-1] + bytes([original[-1] ^ 1])
            Path(cut["path"]).write_bytes(module.zstd.ZstdCompressor().compress(changed))
            cut.update(module.record(cut["path"]))
            with self.assertRaisesRegex(ValueError, "cut bytes differ"):
                module.validate_cuts(cuts, scans, 95_000_000)

    def test_omitted_input_source_drift_and_wrong_anchor_fail(self):
        path = self.root / "source.py"
        path.write_text("before")
        node = module.record(path)
        method_path = self.root / "method.yaml"
        module.freeze_payload({"schema": module.SELECTION_SCHEMA, "sources": [], "inputs": []}, method_path)
        method = module.canonical_method(method_path, module.SELECTION_SCHEMA)
        with self.assertRaisesRegex(ValueError, "mandatory"):
            module.require_closure(method, "inputs", [node])
        path.write_text("after")
        with self.assertRaisesRegex(ValueError, "identity drift"):
            module.check_record(node)
        method_path.with_suffix(".yaml.identity").write_text("0" * 64)
        with self.assertRaisesRegex(ValueError, "identity/anchor"):
            module.canonical_method(method_path, module.SELECTION_SCHEMA)

    def test_forged_execution_identity_command_or_exit_reject(self):
        work = self.root / "truncated"
        (work / "status").mkdir(parents=True)
        log = work / "native.log"
        log.write_text("synthetic complete")
        module.write_yaml(work / "status" / f"{module.DAY}.yaml", {"status": "completed", "fatal_error": False})
        command = ["immutable/coco", "exact-config"]
        receipt = {"command": command, "prefix_outputs": [str(work / "data" / module.DAY / "3481/values.parquet")]}
        execution = {"method_identity": "a" * 64, "job": "receive_prefix", "exit_code": 0, "command": command, "outputs": [], "log": module.record(log)}
        module.write_yaml(work / "execution-receipt.yaml", execution)
        with patch.object(module, "OUTPUT", self.root):
            module.validate_execution(receipt, "a" * 64)
            for key, value in (("method_identity", "b" * 64), ("exit_code", 1), ("command", ["other"]), ("job", "other")):
                changed = copy.deepcopy(execution)
                changed[key] = value
                module.write_yaml(work / "execution-receipt.yaml", changed)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.validate_execution(receipt, "a" * 64)


if __name__ == "__main__":
    unittest.main()
