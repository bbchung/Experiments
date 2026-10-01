"""Synthetic binary/prefix/state boundaries; no actual research data or replay."""

import copy
import ctypes as ct
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import zstandard as zstd

spec = importlib.util.spec_from_file_location("h15_prefix", Path(__file__).with_name("prepare_receive_prefix.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def binary_record(symbol, receive, exchange, *, seq=1, kind="B", status=0, quantity=1):
    header = module.Header(seq, receive, kind.encode())
    if kind == "B":
        payload = module.Book()
        payload.bid_depth = payload.ask_depth = 1
        payload.bid[0], payload.ask[0] = 100.0, 101.0
    else:
        payload = module.Trade()
        payload.quantity = quantity
    payload.symbol = symbol.encode()
    payload.exchange, payload.status = exchange, status
    return bytes(header) + bytes(payload)


class NativeReaderTest(unittest.TestCase):
    def test_abi_matches_declared_native_layout_and_rejects_other_endian(self):
        self.assertEqual(module.abi()["header_bytes"], 24)
        self.assertEqual(ct.sizeof(module.Book), 152)
        self.assertEqual(ct.sizeof(module.Trade), 56)
        with patch.object(sys, "byteorder", "big"), self.assertRaises(ValueError):
            module.abi()

    def test_receive_regression_clamps_without_rewriting_payload_or_zero_exchange(self):
        first = binary_record("2308", 200, 900, seq=17)
        second = binary_record("2308", 190, 0, seq=18, kind="T")
        stream = io.BytesIO(first + second)
        one = module.read_record(stream, "2308", 0, 1)
        two = module.read_record(stream, "2308", one["receive"], 2)
        self.assertEqual(two["receive"], 200)
        self.assertEqual(two["raw_receive"], 190)
        self.assertEqual(two["exchange"], 0)
        self.assertEqual(one["bytes"] + two["bytes"], first + second)
        self.assertIsNone(module.read_record(stream, "2308", 200, 3))

    def test_partial_reads_are_assembled_but_truncated_header_payload_reject(self):
        class Fragmented(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 3))

        raw = binary_record("2317", 30, 20)
        self.assertEqual(module.read_record(Fragmented(raw), "2317", 0, 1)["bytes"], raw)
        for fragment in (raw[:23], raw[:-1]):
            with self.subTest(size=len(fragment)), self.assertRaises(ValueError):
                module.read_record(io.BytesIO(fragment), "2317", 0, 1)

    def test_wrong_symbol_or_unsupported_message_never_silently_terminates(self):
        with self.assertRaises(ValueError):
            module.read_record(io.BytesIO(binary_record("2317", 30, 20)), "2308", 0, 1)
        with self.assertRaises(ValueError):
            module.read_record(io.BytesIO(bytes(module.Header(1, 30, b"X"))), "2308", 0, 1)

    def test_eof_does_not_supply_next_distinct_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = binary_record("2308", 10, 100) + binary_record("2308", 15, 100, seq=2)
            source = root / "source.bin.zst"
            source.write_bytes(zstd.ZstdCompressor().compress(raw))
            result = module.scan_stream(source, "2308", root / "spool.bin", [30], 4096)
            self.assertEqual(result["windows"], [])
            self.assertEqual((root / "spool.bin").read_bytes(), raw)
            self.assertEqual(result["records"], 2)

    def test_ignored_zero_print_does_not_close_pending_own_exchange(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = binary_record("2308", 10, 100) + binary_record("2308", 20, 101, seq=2, kind="T", quantity=0) + binary_record("2308", 40, 102, seq=3)
            source = root / "source.bin.zst"
            source.write_bytes(zstd.ZstdCompressor().compress(raw))
            result = module.scan_stream(source, "2308", root / "spool.bin", [30], 4096)
            self.assertEqual([(w["R"], w["S"], w["A"]) for w in result["windows"]], [(10, 30, 40)])
            self.assertEqual((root / "spool.bin").read_bytes(), raw)

    def test_pending_own_exchange_may_lead_receive_domain(self):
        row = module.read_record(io.BytesIO(binary_record("2308", 10, 10_000)), "2308", 0, 1)
        following = module.read_record(io.BytesIO(binary_record("2308", 40, 11_000)), "2308", 10, 2)
        group = {"last": row, "first": module.evidence(row), "has_book": True, "continuous": True}
        self.assertEqual(module.closed_window(group, following, [30])[0]["pending_exchange"], 10_000)
        self.assertFalse(module.positive_clock(2**53))
        self.assertFalse(module.positive_clock(0))
        self.assertFalse(module.positive_clock(10.0))

    def test_spool_limit_is_fail_closed_without_truncating_selection_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin.zst"
            source.write_bytes(zstd.ZstdCompressor().compress(binary_record("2308", 10, 100)))
            with self.assertRaises(ValueError):
                module.scan_stream(source, "2308", root / "spool.bin", [30], 32)

    def test_cut_preserves_native_padding_and_uses_effective_not_raw_receive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = bytearray(binary_record("2308", 10, 100))
            first[17:24] = b"padding"
            first[-2:] = b"XY"
            raw = bytes(first) + binary_record("2308", 40, 101, seq=2) + binary_record("2308", 20, 101, seq=3) + binary_record("2308", 50, 102, seq=4)
            source = root / "source.bin.zst"
            source.write_bytes(zstd.ZstdCompressor().compress(raw))
            scanned = module.scan_stream(source, "2308", root / "spool.bin", [30], 4096)
            scanned["source"] = module.record(source)
            cut = module.write_cut(scanned, 30, root / "cut.bin.zst")
            with zstd.ZstdDecompressor().stream_reader(io.BytesIO((root / "cut.bin.zst").read_bytes())) as stream:
                self.assertEqual(stream.read(), bytes(first))
            self.assertEqual(cut["records"], 1)
            self.assertEqual(cut["last_effective_receive"], 10)
            self.assertEqual(list(scanned["times"]), [10, 40, 40, 50])

    def test_exchange_regression_or_hard_status_is_not_native_closure_proof(self):
        row = module.read_record(io.BytesIO(binary_record("2308", 10, 100)), "2308", 0, 1)
        group = {"last": row, "first": module.evidence(row), "has_book": True, "continuous": True}
        for next_row in (binary_record("2308", 40, 90), binary_record("2308", 40, 110, status=1)):
            following = module.read_record(io.BytesIO(next_row), "2308", 10, 2)
            self.assertEqual(module.closed_window(group, following, [30]), [])


class NativeBitsTest(unittest.TestCase):
    def table(self, times, bits):
        values = np.asarray(bits, dtype=np.uint64).view(np.float64)
        return pa.table(
            {
                "SampleTime": pa.array(times, type=pa.int64()),
                "SampleBookTime": pa.array(times, type=pa.int64()),
                "SampleBookSeq": pa.array(list(range(1, len(times) + 1)), type=pa.int64()),
                "native": pa.array(values, type=pa.float64(), from_pandas=False),
                "phase": pa.array(["warmup", "unknown", "quote"][: len(times)]),
            }
        )

    def test_warmup_nan_payload_and_positive_zero_are_preserved_exactly(self):
        bits = [0xFFF0000000000000, 0x7FF8000000000011, 0x0000000000000000]
        full = self.table([10, 20, 30], bits)
        prefix = full.slice(0, 2)
        self.assertEqual(module.compare_tables(full, prefix, 20)["common_rows"], 2)
        for changed in ([bits[0], 0x7FF8000000000012, bits[2]], [bits[0], bits[1], 0x8000000000000000]):
            other = self.table([10, 20, 30], changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                module.compare_tables(full, other, 30)

    def test_lost_invented_reordered_or_future_origin_is_rejected(self):
        full = self.table([10, 20, 30], [0, 0, 0])
        for prefix in (full.slice(0, 1), full, full.take(pa.array([1, 0], type=pa.int64()))):
            with self.subTest(keys=prefix["SampleTime"].to_pylist()), self.assertRaises(ValueError):
                module.compare_tables(full, prefix, 20)

    def test_empty_original_event_population_through_cutoff_is_not_invented(self):
        full = self.table([10, 20, 30], [0, 0, 0])
        self.assertEqual(module.compare_tables(full, full.slice(0, 0), 5)["common_rows"], 0)
        with self.assertRaises(ValueError):
            module.compare_tables(full, full.slice(0, 0), 20)

    def test_category_recode_or_key_dtype_and_null_change_reject(self):
        full = self.table([10, 20, 30], [0, 0, 0])
        modified = full.set_column(full.schema.get_field_index("phase"), "phase", pa.array(["warmup", "unknown", "flow"]))
        with self.assertRaises(ValueError):
            module.compare_tables(full, modified, 30)
        modified = full.set_column(0, "SampleTime", pa.array([10.0, 20.0, 30.0]))
        with self.assertRaises(ValueError):
            module.native_keys(modified)
        modified = full.set_column(0, "SampleTime", pa.array([10, None, 30], type=pa.int64()))
        with self.assertRaises(ValueError):
            module.native_keys(modified)

    def test_only_raw_input_directory_can_change_in_prefix_yaml(self):
        original = {
            "Users": [],
            "Modules": [
                {"Gid": "", "Decl": [{"Desc": "TradeBookMd.0", "Spec": {"Dirs": ["old"], "Fixed": True}}]},
                {"Gid": "2308", "Decl": [{"Desc": "PeerTradeInformation.0", "Spec": {"TargetSymbol": "2308", "PeerSymbol": "2317"}}]},
            ],
        }
        untouched = copy.deepcopy(original)
        config = module.prefix_config(original, Path("/new/raw"))
        config["Modules"][0]["Decl"][0]["Spec"]["Dirs"] = ["old"]
        self.assertEqual(config, untouched)
        self.assertEqual(original, untouched)

    def test_prospective_profile_matches_immutable_code_recipe(self):
        self.assertEqual(module.read_yaml(module.PROFILE), module.CONTRACT)


class FrozenPrefixClosureTest(unittest.TestCase):
    def test_resigned_manifest_cannot_omit_raw_profile_support_or_compiled_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "selection-profile.yaml"
            module.write_yaml(profile, module.CONTRACT)
            source = root / "compiled.cpp"
            source.write_text("immutable source")
            nodes = []
            for name in ("original.bin.zst", "support.yaml", "old-method.yaml", "old-method.yaml.identity"):
                path = root / name
                path.write_text(name)
                nodes.append(module.record(path))
            nodes.append(module.record(profile))
            sources = [module.record(source)]
            body = {
                "profile": module.record(profile),
                "profile_identity": module.digest(module.CONTRACT),
                "sources": sources,
                "inputs": nodes,
                "raw_streams_decoded": 0,
                "native_replays_launched": 0,
                "labels_read": False,
                "model_fits": 0,
            }
            bound = root / "bound-plan.yaml"
            module.write_yaml(bound, body)
            path = root / "method.yaml"
            with patch.object(module, "OUTPUT", root), patch.object(module, "PROFILE", profile), patch.object(module, "parent_dependencies", return_value=({}, sources, nodes)):
                required = {"sources": sources, "inputs": [*nodes, module.record(bound)]}
                good = {
                    "schema": "h15-real-receive-prefix-selection-method-v1",
                    "status": "frozen",
                    "plan": module.record(bound),
                    "profile_identity": module.digest(module.CONTRACT),
                    **required,
                }
                good["identity"] = module.digest(good)
                module.write_yaml(path, good)
                path.with_suffix(".yaml.identity").write_text(good["identity"])
                module.verify_selection(path)
                for scope in ("sources", "inputs"):
                    for omitted in required[scope]:
                        candidate = {
                            "schema": "h15-real-receive-prefix-selection-method-v1",
                            "status": "frozen",
                            "plan": module.record(bound),
                            "profile_identity": module.digest(module.CONTRACT),
                            "sources": sources,
                            "inputs": required["inputs"],
                        }
                        candidate[scope] = [n for n in required[scope] if n != omitted]
                        candidate["identity"] = module.digest(candidate)
                        module.write_yaml(path, candidate)
                        path.with_suffix(".yaml.identity").write_text(candidate["identity"])
                        with self.subTest(scope=scope, omitted=omitted["path"]), self.assertRaises(ValueError):
                            module.verify_selection(path)


if __name__ == "__main__":
    unittest.main()
