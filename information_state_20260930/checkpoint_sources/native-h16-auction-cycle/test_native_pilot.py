"""Synthetic H16 lifecycle/ABI/state failures; no real preparation or values."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

spec = importlib.util.spec_from_file_location("h16_native_pilot", Path(__file__).with_name("native_pilot.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def table_fixture(rows=4, *, day=None):
    start = module.session_bounds(day)[0] if day else 100_000_000
    sample = start + np.arange(rows, dtype=np.int64) * 10_000_000
    values = {name: np.zeros(rows, dtype=np.float64) for name in module.COLUMNS}
    values.update(SampleTime=sample, SampleBookTime=sample - 1, SampleBookSeq=np.arange(rows, dtype=np.int64))
    values["OriginMidPrice"][:] = 100
    for name in module.NUMERIC_COLUMNS:
        values[name][:] = -np.inf
    for name in module.PRICE_INFO:
        values[f"H16_{name}"][:] = np.nan
    values["H16_trial_matched_quantity"][:] = np.nan
    values[module.CAT_COLUMNS[0]] = np.full(rows, "warmup", dtype=object)
    values[module.CAT_COLUMNS[1]] = np.full(rows, "warmup", dtype=object)
    return build_table(values), values


def build_table(values):
    fields, arrays = [], []
    for name in module.COLUMNS:
        dtype = pa.int64() if name in module.KEYS else pa.string() if name in module.CAT_COLUMNS else pa.float64()
        role = b"time" if name == "SampleTime" else b"context" if name in module.KEYS else b"feature" if name in module.ALPHA_COLUMNS else b"metadata"
        fields.append(pa.field(name, dtype, metadata={b"coco.role": role}))
        arrays.append(pa.array(values[name], type=dtype))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def active_fixture():
    _, values = table_fixture()
    sample = values["SampleTime"]
    first = np.full(len(sample), sample[0] - 20_000_000, dtype=np.float64)
    values["H16_actual_anchor_exchange_time"] = first + 500_000_000  # independent E can exceed origin
    values["H16_actual_anchor_first_receive_time"] = first
    values["H16_actual_mass_available_time"] = first.copy()
    values["H16_actual_clearing_price"][:] = 100
    values["H16_observed_actual_quantity"][:] = 10
    values["H16_trial_exchange_time"] = sample + 500_000_000
    values["H16_trial_available_time"] = sample.astype(np.float64) - 1
    values["H16_trial_matched_quantity"][:] = 0
    values["H16_cycle_support"][:] = 1
    values[module.CAT_COLUMNS[0]][:] = "indicative"
    values[module.CAT_COLUMNS[1]][:] = "trial_zero"
    for name in module.NUMERIC_COLUMNS:
        values[name][:] = np.nan
    values[f"{module.MODULE}.0.proposal_match_mass_share.0"][:] = 0.0
    return build_table(values), values


class ConfigAndContractTest(unittest.TestCase):
    def original(self):
        return {
            "Users": [],
            "Modules": [{"Gid": "", "Decl": [{"Desc": "NativeMd.0", "Spec": {"source_order": "unchanged"}}]}]
            + [{"Gid": symbol, "Decl": [{"Desc": "CurrentBook.0", "Spec": {"Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}}]} for symbol in module.CARRIERS],
        }

    def test_raw_all16_route_keeps_zero_trial_without_any_twse_filter(self):
        original = self.original()
        config = module.daily_config(original, list(module.CARRIERS))
        module.validate_daily_config(config, original, list(module.CARRIERS))
        self.assertEqual(config["Modules"][0], original["Modules"][0])
        for group in config["Modules"][1:]:
            self.assertEqual([d["Desc"] for d in group["Decl"]], ["CurrentBook.0", "AuctionCycleInformation.0", "DatasetWriter.0"])
            source = group["Decl"][1]["Spec"]
            self.assertEqual(source["Symbol"], group["Gid"])
            self.assertEqual(source["StatusFilter"], "TRIAL || !TRIAL")
            self.assertEqual(source["Subscribe"], [{"Book": [group["Gid"]], "Trade": [group["Gid"]]}])
            self.assertNotIn("Dep", source)

    def test_no_silent_sampler_label_filter_or_schema_mutation(self):
        original = self.original()
        for field, value in (("PeriodicSampler", {"SampleInterval": "30s"}), ("Labelers", ["future"]), ("Exports", [])):
            config = module.daily_config(original, list(module.CARRIERS))
            config["Modules"][1]["Decl"][-1]["Spec"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.validate_daily_config(config, original, list(module.CARRIERS))
        config = module.daily_config(original, list(module.CARRIERS))
        config["Modules"][1]["Decl"][1]["Spec"]["StatusFilter"] = ""
        with self.assertRaises(ValueError):
            module.validate_daily_config(config, original, list(module.CARRIERS))

    def test_missing_writer_does_not_remove_original_md_carrier(self):
        present = [symbol for symbol in module.CARRIERS if symbol != "3481"]
        config = module.daily_config(self.original(), present)
        self.assertEqual([g["Gid"] for g in config["Modules"][1:]], list(module.CARRIERS))
        missing = next(g for g in config["Modules"] if g["Gid"] == "3481")
        self.assertEqual([d["Desc"] for d in missing["Decl"]], ["CurrentBook.0"])

    def test_actual_prospective_profile_is_fixed_and_no_field_degeneracy_gate(self):
        profile = module.read_yaml(module.PROFILE)
        module.validate_profile(profile)
        for key in ("minimum_session_cycles", "minimum_distinct_sampled_cycles", "minimum_support_origins", "indicative_max_age_micros"):
            changed = copy.deepcopy(profile)
            changed["contract"][key] -= 1
            with self.subTest(field=key), self.assertRaises(ValueError):
                module.validate_profile(changed)
        self.assertTrue(profile["contract"]["variation_is_fieldwise_descriptive"])


class NativeBoundaryTest(unittest.TestCase):
    def test_warmup_and_zero_trial_price_unknown_preserve_partial_states(self):
        table, _ = table_fixture()
        module.validate_table(table)
        active, _ = active_fixture()
        result = module.validate_table(active)
        self.assertTrue(np.isnan(result["numeric"]["proposal_displacement_5ticks"]).all())
        self.assertTrue(np.array_equal(result["numeric"]["proposal_match_mass_share"].view(np.uint64), np.zeros(4, dtype=np.uint64)))

    def test_exchange_ahead_is_valid_but_future_publication_not_valid(self):
        table, values = active_fixture()
        module.validate_table(table)
        values["H16_trial_available_time"][0] = values["SampleTime"][0] + 1
        with self.assertRaisesRegex(ValueError, "future receive"):
            module.validate_table(build_table(values))

    def test_semantic_zero_integer_unsafe_and_float_dtype_fail(self):
        for name, value in (
            ("H16_cycle_support", -0.0),
            ("H16_actual_anchor_exchange_time", 2**53),
            ("H16_trial_available_time", 100.5),
            (f"{module.MODULE}.0.proposal_match_mass_share.0", -0.0),
            (f"{module.MODULE}.0.received_cycle_age.0", np.inf),
        ):
            _, values = active_fixture()
            values[name][0] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                module.validate_table(build_table(values))
        table, _ = active_fixture()
        name = module.NUMERIC_COLUMNS[0]
        field = pa.field(name, pa.float32(), metadata={b"coco.role": b"feature"})
        table = table.set_column(table.schema.get_field_index(name), field, table[name].cast(pa.float32()))
        with self.assertRaisesRegex(ValueError, "dtype"):
            module.validate_table(table)

    def test_native_role_diagnostics_cannot_enter_predictors(self):
        table, _ = active_fixture()
        name = "H16_actual_clearing_price"
        field = pa.field(name, pa.float64(), metadata={b"coco.role": b"feature"})
        table = table.set_column(table.schema.get_field_index(name), field, table[name])
        with self.assertRaisesRegex(ValueError, "semantic role"):
            module.validate_table(table)

    def test_hard_unknown_cannot_publish_zero_and_support_requires_real_operand(self):
        _, values = table_fixture()
        values[module.CAT_COLUMNS[0]][:] = "suspended"
        values[module.CAT_COLUMNS[1]][:] = "unknown"
        for name in module.NUMERIC_COLUMNS:
            values[name][:] = np.nan
        module.validate_table(build_table(values))
        values[module.NUMERIC_COLUMNS[0]][0] = 0
        with self.assertRaisesRegex(ValueError, "hard boundary"):
            module.validate_table(build_table(values))
        _, values = active_fixture()
        values["H16_trial_matched_quantity"][:] = np.nan
        values[module.CAT_COLUMNS[1]][:] = "trial_na"
        with self.assertRaisesRegex(ValueError, "support violates"):
            module.validate_table(build_table(values))

    def test_continuous_is_known_zero_and_unmatched_book_basis_missing(self):
        _, values = table_fixture()
        values[module.CAT_COLUMNS[0]][:] = "continuous"
        values[module.CAT_COLUMNS[1]][:] = "inactive"
        for name in module.NUMERIC_COLUMNS:
            values[name][:] = 0.0
        module.validate_table(build_table(values))
        values[module.NUMERIC_COLUMNS[0]][0] = np.nan
        with self.assertRaisesRegex(ValueError, r"known \+0"):
            module.validate_table(build_table(values))
        _, values = active_fixture()
        values["H16_residual_book_available_time"] = values["H16_trial_available_time"].copy()
        values["H16_residual_book_exchange_time"] = values["H16_trial_exchange_time"] - 1
        values[f"{module.MODULE}.0.residual_mid_clearing_basis_5ticks.0"][:] = 0.1
        with self.assertRaisesRegex(ValueError, "unmatched"):
            module.validate_table(build_table(values))

    def test_repeat_source_identity_cannot_create_distinct_sampled_cycle_credit(self):
        _, values = table_fixture(1381, day=module.DAYS[0])
        sample = values["SampleTime"]
        first = sample[0] + 1
        active = np.arange(len(sample)) > 0
        for name in module.NUMERIC_COLUMNS:
            values[name][:] = np.nan
        values["H16_actual_anchor_first_receive_time"][active] = first
        values["H16_actual_mass_available_time"][active] = first
        values["H16_actual_anchor_exchange_time"][active] = first + 100_000_000
        values["H16_actual_clearing_price"][active] = 100
        values["H16_observed_actual_quantity"][active] = 10
        values["H16_session_actual_anchor_count"][active] = 8
        values["H16_cumulative_actual_anchor_count"][active] = 8
        values["H16_cumulative_unique_actual_print_count"][active] = 8
        values["H16_trial_exchange_time"][active] = sample[active] + 100_000_000
        values["H16_trial_available_time"][active] = sample[active] - 1
        values["H16_trial_matched_quantity"][active] = 0
        values["H16_cycle_support"][active] = 1
        values[module.CAT_COLUMNS[0]][active] = "indicative"
        values[module.CAT_COLUMNS[1]][active] = "trial_zero"
        for name in module.NUMERIC_COLUMNS:
            values[name][~active] = -np.inf
        values[f"{module.MODULE}.0.proposal_match_mass_share.0"][active] = 0
        table = build_table(values)
        original = table.select([*module.KEYS, "OriginMidPrice"])
        result = module.evaluate_cell(module.DAYS[0], "3481", table, original)
        self.assertGreater(result["fresh_post_anchor_support_origins"], 20)
        self.assertEqual(result["distinct_sampled_cycles"], 1)
        self.assertFalse(result["support_gate_passed"])
        result = module.evaluate_cell(module.DAYS[0], "2330", table, original)
        self.assertIsNone(result["support_gate_passed"])

    def test_original_book_sequence_and_mid_payload_bits_are_integrity(self):
        table, _ = table_fixture(1381, day=module.DAYS[0])
        original = table.select([*module.KEYS, "OriginMidPrice"])
        for name in ("SampleBookSeq", "OriginMidPrice"):
            changed = original[name].to_numpy().copy()
            changed[50] += 1
            original_changed = original.set_column(original.schema.get_field_index(name), original.schema.field(name), pa.array(changed))
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "bits changed"):
                module.evaluate_cell(module.DAYS[0], "2330", table, original_changed)
        nan_a = pa.array(np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64))
        nan_b = pa.array(np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64))
        self.assertFalse(module.same_bits(nan_a, nan_b))

    def test_continuous_zeros_remain_outside_mechanism_descriptive_mask(self):
        _, values = table_fixture(1381, day=module.DAYS[0])
        values[module.CAT_COLUMNS[0]][:] = "continuous"
        values[module.CAT_COLUMNS[1]][:] = "inactive"
        for name in module.NUMERIC_COLUMNS:
            values[name][:] = 0
        table = build_table(values)
        result = module.evaluate_cell(module.DAYS[0], "2330", table, table.select([*module.KEYS, "OriginMidPrice"]))
        summary = result["fieldwise_variability"][module.NUMERIC[0]]
        self.assertEqual(summary["all_original30s"]["known_positive_zero"], 461)
        self.assertEqual(summary["observed_auction_or_indicative_original30s"]["origins"], 0)
        self.assertEqual(summary["counted_session_cycle_support_original30s"]["origins"], 0)
        self.assertEqual(result["native_operand_support"]["all_original30s"]["native_cycle_support"], 0)

    def test_constant_error_and_partial_price_do_not_veto_native_cycle_support(self):
        _, values = table_fixture(1381, day=module.DAYS[0])
        sample = values["SampleTime"]
        active = np.arange(len(sample)) > 0
        count = np.minimum(5, 1 + (np.arange(len(sample)) - 1) // 6)
        first = sample[0] + 1 + (count - 1) * 60_000_000
        for name in module.NUMERIC_COLUMNS:
            values[name][active] = np.nan
        values[f"{module.MODULE}.0.proposal_match_mass_share.0"][active] = 0
        values[f"{module.MODULE}.0.previous_acceptance_error_5ticks.0"][active] = 0
        values["H16_previous_error_available_time"][active] = first[active]
        for name in ("actual_anchor_first_receive_time", "actual_mass_available_time"):
            values[f"H16_{name}"][active] = first[active]
        values["H16_actual_anchor_exchange_time"][active] = first[active] + 100_000_000
        values["H16_actual_clearing_price"][active] = 100
        values["H16_observed_actual_quantity"][active] = 10
        for name in ("session_actual_anchor_count", "cumulative_actual_anchor_count", "cumulative_unique_actual_print_count"):
            values[f"H16_{name}"][active] = count[active]
        values["H16_trial_exchange_time"][active] = sample[active] + 100_000_000
        values["H16_trial_available_time"][active] = sample[active] - 1
        values["H16_trial_matched_quantity"][active] = 0
        values["H16_cycle_support"][active] = 1
        values[module.CAT_COLUMNS[0]][active] = "indicative"
        values[module.CAT_COLUMNS[1]][active] = "trial_zero"
        table = build_table(values)
        result = module.evaluate_cell(module.DAYS[0], "3481", table, table.select([*module.KEYS, "OriginMidPrice"]))
        self.assertEqual(result["distinct_sampled_cycles"], 5)
        self.assertTrue(result["support_gate_passed"])
        summary = result["fieldwise_variability"]["previous_acceptance_error_5ticks"]["counted_session_cycle_support_original30s"]
        self.assertEqual(summary["distinct_finite_bits"], 1)
        self.assertEqual(summary["finite_nonzero"], 0)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def node(self, name, value="opaque"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return module.record(path)

    def test_required_absent_is_checked_before_large_closure_hashes(self):
        path = self.root / "absent.bin.zst"
        module.assert_absent([str(path)])
        path.write_bytes(b"new file")
        method_path = self.root / "method.yaml"
        prep_path = self.root / "prep.yaml"
        module.write_yaml(prep_path, {"required_absent_inputs": [str(path)]})
        method = {"preparation": module.record(prep_path)}
        with (
            patch.object(module, "canonical_method", return_value=method),
            patch.object(module, "check_record", side_effect=AssertionError("hash must not run")),
            self.assertRaisesRegex(ValueError, "RequiredAbsent"),
        ):
            module.preflight(method_path)

    def test_resigned_manifest_cannot_omit_mandatory_closure(self):
        mandatory = [self.node(name) for name in ("profile", "kernel", "compiled-source", "rawBIN", "runtime", "parentKeys")]
        for omitted in mandatory:
            method = {"schema": module.PRODUCER_SCHEMA, "status": "draft", "sources": [], "inputs": [n for n in mandatory if n != omitted]}
            path = self.root / f"{Path(omitted['path']).name}.yaml"
            module.freeze_payload(method, path)
            parsed = module.canonical_method(path, module.PRODUCER_SCHEMA)
            with self.subTest(omitted=omitted["path"]), self.assertRaisesRegex(ValueError, "mandatory"):
                module.require_closure(parsed, "inputs", mandatory)

    def test_forged_execution_receipt_or_changed_control_bytes_rejected(self):
        outputs = [self.node(f"control/out{i}", f"bytes{i}") for i in range(4)]
        expected = [self.node(f"old/out{i}", f"bytes{i}") for i in range(4)]
        job = {
            "key": "registration_control",
            "work": str(self.root / "control"),
            "command": ["immutable/coco", "exact-config"],
            "outputs": [n["path"] for n in outputs],
            "required_byte_equal_outputs": expected,
        }
        receipt = {"method_identity": "a" * 64, "job": job["key"], "exit_code": 0, "command": job["command"], "outputs": outputs, "log": self.node("control/native.log")}
        path = self.root / "control/execution-receipt.yaml"
        module.write_yaml(path, receipt)
        module.control_proof({"registration_control": job}, "a" * 64)
        for field, changed in (("method_identity", "b" * 64), ("exit_code", 1), ("job", "20260119"), ("command", ["other"]), ("outputs", outputs[:-1])):
            forged = {**receipt, field: changed}
            module.write_yaml(path, forged)
            with self.subTest(field=field), self.assertRaises(ValueError):
                module.control_proof({"registration_control": job}, "a" * 64)
        receipt["outputs"][0] = self.node("control/out0", "changed")
        module.write_yaml(path, receipt)
        with self.assertRaisesRegex(ValueError, "entire bytes changed"):
            module.control_proof({"registration_control": job}, "a" * 64)

    def test_missing_carrier_skips_day_without_replacement(self):
        basic = self.node("basic.csv", "symbol\n" + "\n".join(module.CARRIERS) + "\n")
        raw = {(module.DAYS[0], symbol): self.node(f"raw/{symbol}") for symbol in module.CARRIERS}
        Path(raw[(module.DAYS[0], "3481")]["path"]).unlink()
        present, skipped = module.select_day(module.DAYS[0], raw, basic, [])
        self.assertEqual(present, [])
        self.assertEqual(skipped[0]["symbol"], "3481")

    def test_code_drift_or_anchor_rewrite_without_matching_body_fails(self):
        source = self.node("source.py", "before")
        path = self.root / "frozen.yaml"
        module.freeze_payload({"schema": module.PRODUCER_SCHEMA, "sources": [source], "inputs": []}, path)
        method = module.canonical_method(path, module.PRODUCER_SCHEMA)
        Path(source["path"]).write_text("after")
        with self.assertRaisesRegex(ValueError, "identity drift"):
            module.check_record(method["sources"][0])
        path.with_suffix(".yaml.identity").write_text("0" * 64)
        with self.assertRaisesRegex(ValueError, "identity/anchor"):
            module.canonical_method(path, module.PRODUCER_SCHEMA)

    def test_ldd_names_and_actual_resolution_are_bound_to_shared_runtime(self):
        library = self.node("runtime/libexample.so")
        dependency = {**library, "name": "libexample.so"}
        good = f"libexample.so => {library['path']} (0xabc)\n"
        module.validate_ldd(good, [dependency])
        for text in ("libexample.so => not found\n", f"libother.so => {library['path']} (0xabc)\n", "libexample.so => /different/libexample.so (0xabc)\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.validate_ldd(text, [dependency])

    def test_recorded_system_library_alias_retains_exact_runtime_bytes(self):
        dependencies = []
        for index in range(51):
            value = f"native-library-{index}"
            library = self.node(f"snapshot/lib{index}.so", value)
            target = self.node(f"system/lib{index}.so.1", value)
            alias = self.root / f"system/lib{index}.so"
            alias.symlink_to(Path(target["path"]).name)
            dependencies.append({**library, "name": alias.name, "original_system_path": str(alias)})
        path = self.root / "runtime.yaml"
        module.write_yaml(path, {"dependencies": dependencies})
        preparation = {"runtime_snapshot": module.record(path), "runtime_dependencies": dependencies}
        module.runtime_check(preparation)
        changed = self.root / "system/lib0.so.1"
        changed.write_text("altered-library")
        with self.assertRaisesRegex(ValueError, "identity drift"):
            module.runtime_check(preparation)


class FullSyntheticLifecycleTest(unittest.TestCase):
    """Opaque temp files exercise lifecycle; no native/parent table is decoded."""

    setUp = LifecycleTest.setUp
    node = LifecycleTest.node

    def fixture(self):
        output = self.root / "study"
        output.mkdir()
        calendar = self.node("calendar.yaml")
        profile = self.root / "profile.yaml"
        module.write_yaml(profile, module.read_yaml(module.PROFILE))
        original = ConfigAndContractTest().original()
        old_work = self.root / "old" / module.DAYS[0]
        old_config_path = old_work / "config.yaml"
        module.write_yaml(old_config_path, original)
        old_outputs = [
            self.node(f"old/{module.DAYS[0]}/{folder}/{module.DAYS[0]}/{symbol}/values.parquet", f"old-{folder}-{symbol}")
            for folder in ("data", "events")
            for symbol in ("2308", "2317")
        ]
        raw = {(day, symbol): {**self.node(f"raw/{day}/{symbol}.bin.zst"), "day": day, "symbol": symbol} for day in module.DAYS for symbol in module.CARRIERS}
        basics = {day: self.node(f"basic/{day}.csv", "symbol\n" + "\n".join(module.CARRIERS) + "\n") for day in module.DAYS}
        parents = [
            {"day": day, "symbol": symbol, "parent_origins": self.node(f"parent/{day}/{symbol}.parquet"), "expected_native_rows": 1381, "expected_support_rows": 461}
            for day in module.DAYS
            for symbol in module.CARRIERS
        ]
        binary = self.node("study/coco", "new native")
        live_binary = self.node("build/Release/src/app/coco", "new native")
        library = {**self.node("runtime/libfixture.so"), "name": "libfixture.so", "original_system_path": str(self.root / "runtime/libfixture.so")}
        runtime = self.node("runtime.yaml")
        old_live = self.node("src/math.cpp", "unchanged math")
        old_snapshot = self.node("study/compiled-source/src/math.cpp", "unchanged math")
        live, sources = [old_live], [old_snapshot]
        for path in module.NEW_SOURCES:
            live.append(self.node(path, path))
            sources.append(self.node("study/compiled-source/" + path, path))
        capsule = {
            "schema": "h16-native-source-capsule-v1",
            "sources": sources,
            "live_sources_before_and_after": live,
            "binary": binary,
            "live_built_binary_before_and_after": live_binary,
            "existing_native_math_all_sha256_equal": True,
            "source_before_after_all_sha256_equal": True,
        }
        capsule_path = output / "capsule.yaml"
        module.write_yaml(capsule_path, capsule)
        registration = {
            "type": module.MODULE,
            "feature_families": [{"name": name, "attributes": ["alpha_factor", *(["categorical"] if name in module.CATEGORICAL else [])]} for name in module.ALPHA]
            + [{"name": name, "attributes": []} for name in module.INFO],
        }
        guide_path = output / "guide.yaml"
        module.write_yaml(guide_path, {"module_types": [registration]})
        ldd = self.node("study/ldd.txt", f"libfixture.so => {library['path']} (0xabc)\n")
        parent = {
            "jobs": [{"day": module.DAYS[0], "work": str(old_work), "config": module.record(old_config_path), "outputs": [n["path"] for n in old_outputs]}],
            "runtime_snapshot": runtime,
            "runtime_dependencies": [library],
            "runtime_directory": str(self.root / "runtime"),
        }
        inherited = {"sources": [old_snapshot], "inputs": [self.node("immutable-parent.yaml")]}
        links = [self.node("parent-method.yaml")]
        context = parent, {"live_sources_before_and_after": [old_live]}, parents, raw, basics, inherited, inherited, links
        patches = [
            patch.object(module, "ROOT", self.root),
            patch.object(module, "OUTPUT", output),
            patch.object(module, "CALENDAR", Path(calendar["path"])),
            patch.object(module, "PROFILE", profile),
            patch.object(module, "parent_metadata", return_value=context),
            patch.object(module, "science_sources", return_value=[module.record(profile), module.record(Path(module.__file__))]),
            patch.object(module, "runtime_check"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        environment = {"LD_LIBRARY_PATH": parent["runtime_directory"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        jobs = []
        for day in module.DAYS:
            work = output / "pilot" / day
            config_path = work / "config.yaml"
            module.write_yaml(config_path, module.daily_config(original, list(module.CARRIERS)))
            jobs.append(module.command_job(day, work, config_path, list(module.CARRIERS), binary, environment))
        control_work = output / "registration-control" / module.DAYS[0]
        control_config = control_work / "config.yaml"
        module.write_yaml(control_config, original)
        control_outputs = [str(control_work / Path(n["path"]).relative_to(old_work)) for n in old_outputs]
        control = module.command_job(module.DAYS[0], control_work, control_config, ["2308", "2317"], binary, environment, key="registration_control", outputs=control_outputs)
        control.update(old_config=module.record(old_config_path), required_byte_equal_outputs=old_outputs)
        inputs = module.unique_records(
            [
                *links,
                *inherited["inputs"],
                *raw.values(),
                *basics.values(),
                *[p["parent_origins"] for p in parents],
                *[module.record(path) for path in (profile, capsule_path, guide_path)],
                ldd,
                binary,
                runtime,
                library,
                calendar,
                *[j["config"] for j in jobs],
                control["config"],
                control["old_config"],
                *old_outputs,
            ]
        )
        body = {
            "profile": module.record(profile),
            "profile_identity": module.digest(module.read_yaml(profile)),
            "producer": module.record(capsule_path),
            "guide": module.record(guide_path),
            "ldd": ldd,
            "native_module_registration": registration,
            "binary": binary,
            "runtime_snapshot": runtime,
            "runtime_dependencies": [library],
            "runtime_directory": parent["runtime_directory"],
            "parents": parents,
            "raw_files": list(raw.values()),
            "basic_info": basics,
            "template": parent["jobs"][0]["config"],
            "jobs": jobs,
            "registration_control": control,
            "execution_order": ["registration_control", *module.DAYS],
            "required_absent_inputs": [],
            "missing": [],
            "sources": module.unique_records([*sources, *module.science_sources()]),
            "inputs": inputs,
        }
        prep_path = output / "preparation-receipt.yaml"
        module.write_yaml(prep_path, body)
        method_path = output / "producer.yaml"
        identity = module.freeze_payload(module.producer_payload(prep_path), method_path)
        return body, prep_path, method_path, identity

    def complete(self, job, identity):
        outputs = []
        for index, path in enumerate(job["outputs"]):
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(job["required_byte_equal_outputs"][index]["path"]).read_bytes() if job["key"] == "registration_control" else b"opaque native parquet")
            outputs.append(module.record(target))
        log = Path(job["work"]) / "native.log"
        log.write_text("actual synthetic exit0")
        module.write_yaml(
            Path(job["work"]) / "execution-receipt.yaml",
            {"method_identity": identity, "job": job["key"], "exit_code": 0, "command": job["command"], "outputs": outputs, "log": module.record(log)},
        )

    def test_full_preflight_control_first_and_hash_only_evaluation_binding(self):
        preparation, _, method, identity = self.fixture()
        module.preflight(method, "registration_control")
        with self.assertRaises(FileNotFoundError):
            module.preflight(method, module.DAYS[0])
        self.complete(preparation["registration_control"], identity)
        for job in preparation["jobs"]:
            module.preflight(method, job["key"])
            self.complete(job, identity)
        with self.assertRaisesRegex(ValueError, "fresh native destination"):
            module.preflight(method, "registration_control")
        with patch.object(module.pq, "read_table", side_effect=AssertionError("values cannot be decoded before eval freeze")):
            payload = module.bind_evaluation(method)
            self.assertEqual(len(module.read_yaml(Path(payload["profile"]["path"]))["cells"]), 32)
            evaluation = module.OUTPUT / "evaluation.yaml"
            module.freeze_payload(payload, evaluation)
            module.verify_evaluation(evaluation)

    def test_resigned_preparation_cannot_omit_raw_input_from_producer_closure(self):
        preparation, prep_path, method, _ = self.fixture()
        removed = preparation["raw_files"][0]["path"]
        preparation["inputs"] = [node for node in preparation["inputs"] if node["path"] != removed]
        module.write_yaml(prep_path, preparation)
        method.unlink()
        method.with_suffix(".yaml.identity").unlink()
        module.freeze_payload(module.producer_payload(prep_path), method)
        with self.assertRaisesRegex(ValueError, "mandatory inputs"):
            module.preflight(method)

    def test_evaluation_freeze_cannot_be_skipped_before_native_decode(self):
        _, _, method, _ = self.fixture()
        with patch.object(module.pq, "read_table", side_effect=AssertionError("unfrozen decode forbidden")), self.assertRaisesRegex(ValueError, "schema/status"):
            module.check(method)


if __name__ == "__main__":
    unittest.main()
