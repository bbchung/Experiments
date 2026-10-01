"""Prepare and preflight native H13 artifacts; never launch replay or training.

Only root executes --prepare after reviewing this script, then independently
freezes the generated profile/receipt/config/source/runtime/input closure.
--preflight requires that external canonical freeze before any replay command.
No raw stream, feature/label Parquet or model is decoded by this program.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
OLD = OUTPUT.parent / "native-h10-ordered-reset"
EXPERIMENT = ROOT / "AstraResearch/experiments/information_state_20260930"
PROFILE = EXPERIMENT / "h13-native-profile.yaml"
PLAN = EXPERIMENT / "H13_NATIVE_PLAN.md"
CALENDAR = ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"
sys.path.insert(0, str(ROOT / "AstraResearch"))

from astra.io import digest, file_hash, read_yaml, write_yaml

DAYS = ("20260119", "20260120")
TARGETS = ("2308", "2317")
CARRIERS = ("2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2615", "3481", "2409", "2344", "2337", "2481", "3037", "3711")
ALPHA = ("visit_phase", "completed_mark_direction", "accepted_anchor_offset_5ticks", "bilateral_test_strength", "completed_receive_age_300s")
INFO = (
    "processed_cluster_exchange_time",
    "processed_cluster_available_time",
    "completed_landmark_id",
    "completed_landmark_available_time",
    "completed_landmark_exchange_time",
    "cumulative_completed_landmark_count",
    "cumulative_up_completed_landmark_count",
    "cumulative_down_completed_landmark_count",
    "current_visit_id",
    "cumulative_quote_led_visit_count",
    "cumulative_ambiguous_cluster_count",
    "cumulative_hard_censor_count",
    "cumulative_cleared_visit_count",
    "cumulative_pair_ended_visit_count",
    "cumulative_cluster_overflow_count",
)
CONTRACT = {
    "periodic_seconds": 10,
    "expected_native_rows_per_present_cell": 1381,
    "formal_support_origin_seconds": 30,
    "expected_support_rows_per_present_cell": 461,
    "max_book_age_micros": 5_000_000,
    "maximum_buffered_observations": 256,
    "recent_support_age_micros": 300_000_000,
    "minimum_observed_fixed_cells": 1,
    "minimum_session_completions_per_present_cell": 5,
    "minimum_distinct_recent_sampled_landmarks_per_present_cell": 5,
    "minimum_session_completions_each_direction": 5,
    "minimum_provisional_or_unilateral_origins": 20,
}
RELOCATED = "src/oms/modules/feature/microstructure/liquidity/demand_repair_renewal"
NEW_H10 = "src/oms/modules/feature/experimental/demand_repair_renewal"
NEW_H13 = "src/oms/modules/feature/experimental/quote_visit_acceptance"
ALLOWED_OLD_SOURCE_CHANGES = {
    "src/oms/modules/feature/test/CMakeLists.txt",  # adds the focused H13 target
    "src/oms/modules/feature/test/demand_repair_renewal_test.cpp",  # independently validated dual-clock test
}


def record(path: Path):
    path = path.resolve()
    return {"path": str(path), "sha256": file_hash(path), "size": path.stat().st_size}


def check_record(node, *, allow_missing=False):
    if type(node.get("path")) is not str or type(node.get("sha256")) is not str or re.fullmatch("[0-9a-f]{64}", node["sha256"]) is None:
        raise ValueError("exact path and SHA256 artifact record required")
    path = Path(node["path"])
    if not path.is_absolute():
        raise ValueError("absolute artifact path required")
    if not path.is_file() and allow_missing:
        return None
    if not path.is_file() or file_hash(path) != node["sha256"]:
        raise ValueError(f"changed/missing bound artifact: {path}")
    if "size" in node and (type(node["size"]) is not int or node["size"] != path.stat().st_size):
        raise ValueError(f"bound artifact size changed: {path}")
    return path


def unique_records(records):
    result = {}
    for node in records:
        path = str(Path(node["path"]).resolve())
        if path in result and result[path]["sha256"] != node["sha256"]:
            raise ValueError(f"conflicting artifact identity: {path}")
        result[path] = node
    return [result[path] for path in sorted(result)]


def fresh_copy(source: Path, target: Path):
    if target.exists():
        raise ValueError(f"fresh capsule destination required: {target}")
    before = record(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target.chmod(0o555 if os.access(source, os.X_OK) else 0o444)
    frozen = record(target)
    if frozen["sha256"] != before["sha256"] or record(source) != before:
        raise ValueError(f"source changed while copying: {source}")
    return before, frozen


def source_capsule():
    prior = read_yaml(OLD / "producer-source-build.yaml")
    snapshot = read_yaml(OLD / "compiled-source-snapshot.yaml")
    if len(prior["sources"]) != 1637 or len(snapshot["sources"]) != 1637:
        raise ValueError("fixed 1637-source historical native capsule required")
    old_by_relative = {str(Path(node["path"]).relative_to(OLD / "compiled-source")): node for node in snapshot["sources"]}
    live = {}
    changed, relocations = [], []
    for node in prior["sources"]:
        relative = str(Path(node["path"]).relative_to(ROOT))
        old_node = old_by_relative[relative]
        if old_node["sha256"] != node["sha256"]:
            raise ValueError("historical build/snapshot source identity disagrees")
        check_record(old_node)
        relocated = relative.replace(RELOCATED + "/", NEW_H10 + "/", 1) if relative.startswith(RELOCATED + "/") else relative
        path = ROOT / relocated
        current = record(path)
        if current["sha256"] != node["sha256"]:
            if relative not in ALLOWED_OLD_SOURCE_CHANGES:
                raise ValueError(f"unaccounted prior native source drift: {relative}")
            changed.append({"relative_path": relocated, "old_sha256": node["sha256"], "new_sha256": current["sha256"]})
        if relocated != relative:
            relocations.append({"old_relative_path": relative, "new_relative_path": relocated, "math_source_sha256": current["sha256"]})
        live[relocated] = current
    if len(relocations) != 3:
        raise ValueError("exactly three H10 leaf files must move without byte changes")
    additions = [
        f"{NEW_H13}/quote_visit_acceptance.h",
        f"{NEW_H13}/quote_visit_acceptance.cpp",
        f"{NEW_H13}/CMakeLists.txt",
        "src/oms/modules/feature/test/quote_visit_acceptance_test.cpp",
        "src/oms/modules/feature/experimental/README.md",
        "build/Release/compile_commands.json",
        "build/Release/CMakeCache.txt",
    ]
    for relative in additions:
        live[relative] = record(ROOT / relative)
    commands = read_compile_commands(ROOT / "build/Release/compile_commands.json")
    if not any(str(ROOT / NEW_H13 / "quote_visit_acceptance.cpp") == str(Path(command["file"]).resolve()) for command in commands):
        raise ValueError("compiled H13 translation unit absent from current Release commands")
    frozen_sources = []
    for relative, node in sorted(live.items()):
        before, frozen = fresh_copy(Path(node["path"]), OUTPUT / "compiled-source" / relative)
        if before != node:
            raise ValueError("live source drifted after initial capsule inventory")
        frozen_sources.append(frozen)
    # Verify the ENTIRE live/source capsule after copying, not only last file.
    for node in live.values():
        check_record(node)
    for node in frozen_sources:
        check_record(node)
    return {
        "schema": "h13-native-source-capsule-v1",
        "prior_build": record(OLD / "producer-source-build.yaml"),
        "prior_capsule": record(OLD / "compiled-source-snapshot.yaml"),
        "prior_source_count": 1637,
        "live_sources_before_and_after": list(live.values()),
        "sources": frozen_sources,
        "relocations": relocations,
        "explicit_non_math_source_changes": changed,
        "source_before_after_all_sha256_equal": True,
    }


def read_compile_commands(path):
    import json

    return json.loads(path.read_text())


def runtime_preflight():
    runtime = read_yaml(OLD / "runtime-snapshot.yaml")
    if len(runtime["dependencies"]) != 51:
        raise ValueError("fixed 51-dependency native runtime snapshot required")
    for node in runtime["dependencies"]:
        check_record(node)
        if file_hash(Path(node["original_system_path"])) != node["sha256"]:
            raise ValueError(f"current native runtime differs from frozen snapshot: {node['name']}")
    return runtime["dependencies"]


def native_guide(binary, environment):
    target = OUTPUT / "feature-guide.yaml"
    if target.exists():
        raise ValueError("fresh feature-guide artifact required")
    result = subprocess.run([str(binary), "feature-guide"], env=environment, capture_output=True, check=True)
    target.write_bytes(result.stdout)
    guide = read_yaml(target)
    module = [item for item in guide["module_types"] if item["type"] == "QuoteVisitAcceptance"]
    if len(module) != 1:
        raise ValueError("exactly one native QuoteVisitAcceptance registration required")
    families = module[0]["feature_families"]
    if len(families) != 20 or {f["name"] for f in families if "alpha_factor" in f["attributes"]} != set(ALPHA):
        raise ValueError("native guide differs from exact five-Alpha contract")
    if {f["name"] for f in families if "alpha_factor" not in f["attributes"]} != set(INFO):
        raise ValueError("native guide differs from exact fifteen-Info contract")
    if {f["name"] for f in families if "categorical" in f["attributes"]} != {"visit_phase"}:
        raise ValueError("native categorical phase contract changed")
    return record(target), module[0]


def selected_targets(day, raw_nodes, basic_node):
    missing = []
    if check_record(basic_node, allow_missing=True) is None:
        return [], [{"day": day, "scope": "whole_day", "reason": "missing_basic_info_skip_no_replacement"}]
    with Path(basic_node["path"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))  # contract metadata, never market/label data
    present = []
    for symbol in TARGETS:
        selected = [row for row in rows if row["symbol"] == symbol]
        if not selected:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_contract_skip_no_replacement"})
        elif len(selected) != 1:
            raise ValueError("duplicate required symbol contract metadata")
        elif check_record(raw_nodes[(day, symbol)], allow_missing=True) is None:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_raw_skip_no_replacement"})
        else:
            present.append(symbol)
    return present, missing


def daily_config(original, present):
    groups = []
    for group in original["Modules"]:
        if not group["Gid"]:
            groups.append(copy.deepcopy(group))
            continue
        symbol = group["Gid"]
        current = next(item for item in group["Decl"] if item["Desc"] == "CurrentBook.0")
        declarations = [copy.deepcopy(current)]
        if symbol in present:
            metadata = [
                {"Feature": f"CurrentBook.0.{family}.0@{symbol}", "Name": name}
                for family, name in (
                    ("book_mid_price", "OriginMidPrice"),
                    ("book_mid_ticks", "OriginMidTicks"),
                    ("book_bid_ticks", "OriginBidTicks"),
                    ("book_ask_ticks", "OriginAskTicks"),
                )
            ]
            metadata += [{"Feature": f"QuoteVisitAcceptance.0.{family}.0@{symbol}", "Name": f"H13_{family}"} for family in INFO]
            declarations = [
                {"Desc": "TwseFilter.0", "Spec": {"RequireTradable": False, "Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}},
                copy.deepcopy(current),
                {"Desc": "QuoteVisitAcceptance.0", "Spec": {"MaxBookAge": "5s", "Dep": {"Book": [f"TwseFilter.0@{symbol}"], "Trade": [f"TwseFilter.0@{symbol}"]}}},
            ]
            for writer_id, folder, sampler in (
                (0, "data", {"PeriodicSampler": {"StartTime": "091000", "UntilTime": "130000", "SampleInterval": "10s"}}),
                (1, "events", {"BookFlipSampler": {"Mode": "all_book", "MinInterval": "0s", "StartTime": "091000", "UntilTime": "130000"}}),
            ):
                declarations.append(
                    {
                        "Desc": f"DatasetWriter.{writer_id}",
                        "Spec": {
                            "Subscribe": [{"Book": [symbol]}],
                            "Exports": [f"QuoteVisitAcceptance.0.{family}.0@{symbol}" for family in ALPHA],
                            "MetadataExports": copy.deepcopy(metadata),
                            **sampler,
                            "OutputPath": "${cwd}/" + folder + "/${trading_date}/" + symbol + "/values.parquet",
                            "Format": "parquet",
                            "UseTmp": True,
                            "EmitSampleContext": True,
                            "RequireAlphaFactorExports": True,
                        },
                    }
                )
        groups.append({"Gid": symbol, "Decl": declarations})
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def job(day, work, binary, config, symbols, kind):
    return {
        "kind": kind,
        "day": day,
        "symbols": list(symbols),
        "work": str(work),
        "config": record(config),
        "command": [str(binary), "-d", day, "-C", str(work), "--trading-calendar", str(CALENDAR), "--run-status-dir", str(work / "status"), str(config)],
        "environment": {"LD_LIBRARY_PATH": str(OLD / "runtime"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "outputs": [str(work / folder / day / symbol / "values.parquet") for folder in ("data", "events") for symbol in symbols],
    }


def prepare():
    receipt_path = OUTPUT / "preparation-receipt.yaml"
    if receipt_path.exists() or PROFILE.exists() or (OUTPUT / "coco").exists() or (OUTPUT / "compiled-source").exists():
        raise ValueError("fresh native H13 artifacts required; never overwrite/resume preparation")
    input_receipts = [
        OLD / "preparation-receipt.yaml",
        OLD / "full-preparation-receipt.yaml",
        OLD / "producer-source-build.yaml",
        OLD / "compiled-source-snapshot.yaml",
        OLD / "runtime-snapshot.yaml",
    ]
    initial_receipts = [record(path) for path in input_receipts]
    old = read_yaml(input_receipts[0])
    full = read_yaml(input_receipts[1])
    prospective = read_yaml(EXPERIMENT / "h13-support-profile-v2.yaml")
    if full["universe_order"] != list(CARRIERS):
        raise ValueError("fixed sixteen-carrier order changed")
    template = OLD / "pilot/20260119/2308/config.yaml"
    old_job = next(j for j in old["pilot_jobs"] if (j["day"], j["symbol"]) == (DAYS[0], TARGETS[0]))
    check_record({"path": str(template), "sha256": old_job["config_sha256"]})
    original = read_yaml(template)
    if [g["Gid"] for g in original["Modules"] if g["Gid"]] != list(CARRIERS):
        raise ValueError("H10 template carrier/source order changed")
    raw_nodes = {(n["day"], n["symbol"]): n for n in full["raw_inputs"] if n["day"] in DAYS}
    if set(raw_nodes) != {(day, symbol) for day in DAYS for symbol in CARRIERS}:
        raise ValueError("exact fixed32 BIN identities absent from historical receipt")
    if any(not n["path"].endswith(".bin.zst") for n in raw_nodes.values()):
        raise ValueError("source precedence requires bound BIN.zst; no fallback")
    basics = {Path(n["path"]).stem: n for n in full["basic_info_inputs"] if Path(n["path"]).stem in DAYS}
    if set(basics) != set(DAYS):
        raise ValueError("two fixed BasicInfo identities absent from historical receipt")
    parents = [copy.deepcopy(c) for c in prospective["cells"]]
    if [(c["day"], c["symbol"]) for c in parents] != [(d, s) for d in DAYS for s in TARGETS]:
        raise ValueError("four historical parent origin identities changed")
    for c in parents:
        c.pop("raw")
        c.pop("basic_info")
        c["expected_native_rows"] = 1381
        c["expected_support_rows"] = 461
    source = source_capsule()
    dependencies = runtime_preflight()
    original_binary, binary = fresh_copy(ROOT / "build/Release/src/app/coco", OUTPUT / "coco")
    environment = {**os.environ, "LD_LIBRARY_PATH": str(OLD / "runtime")}
    guide, module = native_guide(Path(binary["path"]), environment)
    dynamic = subprocess.run(["ldd", binary["path"]], env=environment, capture_output=True, text=True, check=True).stdout
    resolved_names = set(re.findall(r"^\s*(\S+)\s+=>", dynamic, re.MULTILINE))
    expected_names = {d["name"] for d in dependencies if not d["name"].startswith("ld-linux")}
    if resolved_names != expected_names or "not found" in dynamic:
        raise ValueError("new native binary dynamic dependencies differ from frozen51 closure")
    (OUTPUT / "ldd.txt").write_text(dynamic)
    jobs, missing = [], []
    for day in DAYS:
        present, skips = selected_targets(day, raw_nodes, basics[day])
        missing.extend(skips)
        for parent in (cell for cell in parents if cell["day"] == day and cell["symbol"] in present):
            if any(check_record(parent[key], allow_missing=True) is None for key in ("parent_origins", "expected_native_keys")):
                present.remove(parent["symbol"])
                missing.append({"day": day, "symbol": parent["symbol"], "reason": "missing_original_native_keys_skip_no_replacement"})
        work = OUTPUT / "pilot" / day
        config = work / "config.yaml"
        work.mkdir(parents=True)
        write_yaml(config, daily_config(original, present))
        jobs.append(job(day, work, Path(binary["path"]), config, present, "h13_two_target_day"))
    work = OUTPUT / "relocation-control/20260119/2308"
    control_config = work / "config.yaml"
    work.mkdir(parents=True)
    control_config.write_bytes(template.read_bytes())
    if file_hash(control_config) != old_job["config_sha256"]:
        raise ValueError("relocation control changed original YAML bytes")
    control = job(DAYS[0], work, Path(binary["path"]), control_config, (TARGETS[0],), "h10_relocation_byte_control")
    control["old_config"] = record(template)
    control["old_binary"] = record(OLD / "coco")
    control["required_byte_equal_outputs"] = [record(OLD / "pilot/20260119/2308" / folder / DAYS[0] / TARGETS[0] / "values.parquet") for folder in ("data", "events")]
    source["binary"] = binary
    source["live_built_binary_before_and_after"] = original_binary
    write_yaml(OUTPUT / "producer-source-build.yaml", source)
    inputs = [*initial_receipts, record(template), record(EXPERIMENT / "h13-support-profile-v2.yaml"), record(CALENDAR), *raw_nodes.values(), *basics.values()]
    inputs += [c[key] for c in parents for key in ("parent_origins", "expected_native_keys")]
    inputs += [*control["required_byte_equal_outputs"], control["old_binary"]]
    missing_paths = []
    for node in unique_records(inputs):
        if (
            check_record(
                node, allow_missing=node in list(raw_nodes.values()) + list(basics.values()) + [c[key] for c in parents for key in ("parent_origins", "expected_native_keys")]
            )
            is None
        ):
            missing_paths.append(node["path"])
    # Absence is bound as well as presence: later newly appearing fixed inputs
    # require a new study version, never an implicit replacement or fallback.
    for day, symbol in raw_nodes:
        path = Path(raw_nodes[(day, symbol)]["path"])
        if not path.exists():
            missing.append({"day": day, "symbol": symbol, "reason": "missing_carrier_raw_skip_no_replacement"})
    profile = {
        "schema": "h13-native-source-support-profile-v1",
        "status": "prospective_requires_root_freeze",
        "fixed_train_days": list(DAYS),
        "target_symbols": list(TARGETS),
        "carrier_order": list(CARRIERS),
        "contract": CONTRACT,
        "alpha_fields": list(ALPHA),
        "info_fields": list(INFO),
        "alpha_categorical_fields": ["visit_phase"],
        "labelers": [],
        "parents": parents,
        "missing": missing,
        "required_absent_inputs": sorted(set(missing_paths)),
        "native_support_qualification": "Native writer only; prior raw Python proxy cannot supply producer features or admit support",
        "routing": "Raw CurrentBook + raw Book writer; QuoteVisitAcceptance receives TwseFilter Book/Trade RequireTradable=false",
        "origin_contract": "Existing 1381 native10s keys/rawmid bit equality; choose actual S%30s==0 461 keys; no label read/grid invention",
        "event_writer_contract": "all_book receive snapshot with source sequence; no dedup by time; not every callback/cluster; availability from native Info",
        "counter_session_contract": "Use actual event and periodic native keys at09:10/13:00; subtract prior completed counters; bind landmark availability to session; do not credit pre09:10 facts",
        "support_contract": "Every present fixed cell5 completions+5 unique marks at actual30s keys with0<receive_age<=300s; >=1 observed; global5eachdirection+20 provisional/unilateral origins",
        "overflow_contract": "Above256 delivered book/positiveTrade callbacks poisons wholeopenE current attribution; historicalfact retained; no partialcompletion/EOFflush/timerclose",
        "source_roles": "Source support is not predictive admission; require causal native receive-prefix and fresh fixed FE judge before comparison",
    }
    write_yaml(PROFILE, profile)
    receipt = {
        "schema": "h13-native-preparation-v1",
        "script": record(Path(__file__)),
        "profile": record(PROFILE),
        "profile_identity": digest(profile),
        "native_plan": record(PLAN),
        "producer": record(OUTPUT / "producer-source-build.yaml"),
        "binary": binary,
        "guide": guide,
        "native_module_registration": module,
        "runtime_snapshot": record(OLD / "runtime-snapshot.yaml"),
        "runtime_dependencies": dependencies,
        "runtime_system_paths_match_frozen_sha256": True,
        "ldd": record(OUTPUT / "ldd.txt"),
        "sources": unique_records([*source["sources"], record(Path(__file__)), record(ROOT / "AstraResearch/astra/io.py"), record(PLAN)]),
        "inputs": unique_records(inputs),
        "jobs": jobs,
        "relocation_control": control,
        "execution_order": ["relocation_control", "20260119", "20260120"],
        "native_replays_launched": 0,
        "labels_read": False,
        "model_fits": 0,
        "requires_root_frozen_method_before_replay": True,
    }
    for node in initial_receipts:
        check_record(node)
    for node in source["live_sources_before_and_after"]:
        check_record(node)
    check_record(original_binary)
    runtime_preflight()
    write_yaml(receipt_path, receipt)
    return {"prepared_only": True, "daily_jobs": len(jobs), "relocation_controls": 1, "sources": len(receipt["sources"]), "inputs": len(receipt["inputs"])}


def preflight(method_path, job_key=None):
    frozen = read_yaml(method_path)
    anchor = method_path.with_suffix(method_path.suffix + ".identity").read_text().strip()
    if frozen.get("schema") != "h13-native-producer-method-v1" or frozen.get("status") != "frozen" or frozen.get("identity") != anchor:
        raise ValueError("external root frozen method and identity sidecar required")
    if digest({key: value for key, value in frozen.items() if key != "identity"}) != anchor:
        raise ValueError("canonical method identity mismatch")
    preparation_path = OUTPUT / "preparation-receipt.yaml"
    check_record(frozen["preparation"])
    if Path(frozen["preparation"]["path"]).resolve() != preparation_path:
        raise ValueError("wrong native preparation bound")
    preparation = read_yaml(preparation_path)
    profile = read_yaml(PROFILE)
    check_record(preparation["profile"])
    if frozen.get("preparation_identity") != digest(preparation) or frozen.get("profile_identity") != digest(profile) or preparation["profile_identity"] != digest(profile):
        raise ValueError("preparation/profile body differs from external root freeze")
    if profile["contract"] != CONTRACT or any(type(n) is not int for n in profile["contract"].values()):
        raise ValueError("fixed typed native sampling/support gates changed")
    if (
        profile["alpha_fields"] != list(ALPHA)
        or profile["info_fields"] != list(INFO)
        or profile["fixed_train_days"] != list(DAYS)
        or profile["target_symbols"] != list(TARGETS)
        or profile["carrier_order"] != list(CARRIERS)
    ):
        raise ValueError("fixed native fields/population/order changed")
    required_sources = unique_records(preparation["sources"])
    required_inputs = unique_records(
        [
            *preparation["inputs"],
            preparation["script"],
            preparation["profile"],
            preparation["native_plan"],
            preparation["producer"],
            preparation["binary"],
            preparation["guide"],
            preparation["runtime_snapshot"],
            preparation["ldd"],
            *preparation["runtime_dependencies"],
            *[j["config"] for j in preparation["jobs"]],
            preparation["relocation_control"]["config"],
            record(preparation_path),
        ]
    )
    for key, records in (("sources", required_sources), ("inputs", required_inputs)):
        bound = {str(Path(n["path"]).resolve()): n for n in frozen[key]}
        if any(bound.get(n["path"], {}).get("sha256") != n["sha256"] for n in records):
            raise ValueError(f"external frozen method omitted required {key} closure")
        for node in frozen[key]:
            path = check_record(node, allow_missing=node["path"] in profile["required_absent_inputs"])
            if path is not None and node["path"] in profile["required_absent_inputs"]:
                raise ValueError("previously absent fixed source appeared; new version required")
    runtime_preflight()
    jobs = {"relocation_control": preparation["relocation_control"], **{j["day"]: j for j in preparation["jobs"]}}
    selected = jobs.values() if job_key is None else [jobs[job_key]]
    for j in selected:
        if j["command"][0] != preparation["binary"]["path"] or j["environment"]["LD_LIBRARY_PATH"] != str(OLD / "runtime"):
            raise ValueError("job escaped immutable native/runtime capsule")
        if any(Path(path).exists() for path in j["outputs"]) or (Path(j["work"]) / "status").exists():
            raise ValueError("fresh job outputs/status required before first replay")
    return {"preflight_passed": True, "method_identity": anchor, "job": job_key, "native_replays_launched": 0, "model_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", type=Path, metavar="ROOT_FROZEN_METHOD_YAML")
    parser.add_argument("--job", choices=("relocation_control", *DAYS), help="Check fresh outputs only for this job; always verify the entire frozen closure")
    args = parser.parse_args()
    if args.prepare and args.job:
        parser.error("--job requires --preflight")
    print(prepare() if args.prepare else preflight(args.preflight.resolve(), args.job))


if __name__ == "__main__":
    main()
