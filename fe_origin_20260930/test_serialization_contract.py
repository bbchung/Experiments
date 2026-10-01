"""Candidate emitter parity; no active factory code is changed by these tests."""

from __future__ import annotations

import datetime as dt
import random
import unittest

from yaml_compat import compatible_bytes, needs_legacy

from AstraResearch.io import yaml_bytes


class SerializationContractTest(unittest.TestCase):
    def test_scalar_end_markers_and_unusual_string_bytes(self):
        for value in [None, True, False, 0, -1, 0.0, -0.0, 1e-17, float("nan"), float("inf"), "", "cpu", "中文", "on", "a:b", "a\nb\n", "\x85", "\u2028", "\u2029"]:
            with self.subTest(value=repr(value)):
                self.assertEqual(compatible_bytes(value), yaml_bytes(value))

    def test_nested_nel_nonprimitive_values_and_cycles(self):
        repeated = {"field": ["symbol", "20260101", 5.0]}
        cycle = []
        cycle.append(cycle)
        for value in [
            {"nested": [{"name": "a\x85b"}]},
            {"\x85": ["name"]},
            {"surrogate": "\ud800\udfff"},
            {"emoji": "\U0001f600", "extension_cjk": "\U00020000"},
            {"date": dt.date(2026, 1, 1)},
            {"payload": b"binary"},
            {"symbols": {"2330", "2481"}},
            [repeated, repeated],
            cycle,
        ]:
            with self.subTest(value=repr(value)):
                self.assertEqual(compatible_bytes(value), yaml_bytes(value))
        self.assertTrue(needs_legacy({"nested": ["\x85"]}))
        self.assertTrue(needs_legacy({"surrogate": "\ud800"}))
        self.assertTrue(needs_legacy({"emoji": "\U0001f600"}))
        self.assertFalse(needs_legacy({"name": ["中文", 5, None]}))

    def test_generated_native_like_nested_documents(self):
        rng = random.Random(20260930)
        strings = ["none", "False", "true", "0123", "1e-09", "0.05", "10s", "a: b", "x\ny", "中文", "\u2028", "\u2029", "\ufeff", "\x7f", "\x85", "x " * 200]
        scalars = [None, True, False, 0, -3, 5.0, -0.0, 1e-12, 1.623e32, float("nan"), float("inf"), *strings]

        def build(depth):
            if depth == 0 or rng.random() < 0.35:
                return rng.choice(scalars)
            if rng.random() < 0.5:
                return [build(depth - 1) for _ in range(rng.randrange(6))]
            return {f"field_{index}_{rng.choice(strings)}": build(depth - 1) for index in range(rng.randrange(6))}

        for index in range(600):
            value = build(4)
            with self.subTest(index=index):
                self.assertEqual(compatible_bytes(value), yaml_bytes(value))


if __name__ == "__main__":
    unittest.main()
