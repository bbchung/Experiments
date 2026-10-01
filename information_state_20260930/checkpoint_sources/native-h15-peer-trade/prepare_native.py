"""Prepare and preflight H15 native producers without replay or data decoding.

Only root invokes --prepare after the final native fields, tests and Release
build are reviewed. Root freezes the generated method/profile/input closure
before using --preflight and running any recorded command. This script never
opens market streams, feature/label tables, model artifacts or support counts.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
PARENT = OUTPUT.parent / "native-h13-quote-visit"
EXPERIMENT = ROOT / "AstraResearch/experiments/information_state_20260930"
PROFILE = EXPERIMENT / "h15-native-profile.yaml"
PLAN = EXPERIMENT / "H15_TRADE_LED_NATIVE_PLAN.md"
CALENDAR = ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"
sys.path.insert(0, str(ROOT / "AstraResearch"))

from astra.io import digest, file_hash, read_yaml, write_yaml

DAYS = ("20260119", "20260120")
TARGETS = ("2308", "2317")
PEERS = {"2308": "2317", "2317": "2308"}
CARRIERS = ("2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2615", "3481", "2409", "2344", "2337", "2481", "3037", "3711")
PARENT_IDENTITY = "3591aa72a90bbdcbaa5c6b17cd426fa6415b0d686ed7d946fe4a0fe757ddb8da"
PARENT_BINARY_SHA256 = "cac55a04fa8ef0b5297fc4f70c43edb42cfe98ad3d728cc4750c263d5701d2a2"
MODULE = "PeerTradeInformation"
LEAF = "src/oms/modules/feature/experimental/peer_trade_information"
ALPHA_NUMERIC = (
    "external_impulse_target_unit",
    "peer_work_fraction",
    "target_response_5ticks",
    "target_known_work_imbalance",
    "target_known_work_strength",
    "source_persistence",
    "mark_receive_age_300s",
)
ALPHA_CATEGORICAL = ("j_receive_relation", "j_response_alignment", "k_peer_phase", "k_target_phase")
ALPHA = (*ALPHA_NUMERIC, *ALPHA_CATEGORICAL)
INFO = (
    "processed_peer_exchange_time",
    "processed_peer_available_time",
    "processed_target_exchange_time",
    "processed_target_available_time",
    "source_mark_id",
    "peer_source_available_time",
    "peer_source_exchange_time",
    "qualified_mark_available_time",
    "target_baseline_available_time",
    "target_baseline_receive_time",
    "target_baseline_exchange_time",
    "latest_target_evidence_receive_time",
    "latest_target_evidence_available_time",
    "cumulative_source_mark_count",
    "cumulative_qualified_mark_count",
    "session_qualified_mark_count",
    "session_up_qualified_mark_count",
    "session_down_qualified_mark_count",
    "cumulative_tied_mark_count",
    "cumulative_unknown_cluster_count",
    "cumulative_hard_censor_count",
    "cumulative_cluster_overflow_count",
    "ordered_response_support",
    "source_anchor_mid",
    "source_confirmed_mid",
    "target_baseline_mid",
    "mark_is_qualified",
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
    "minimum_session_marks_per_present_cell": 5,
    "minimum_distinct_recent_sampled_marks_per_present_cell": 5,
    "minimum_session_marks_each_direction": 5,
    "minimum_ordered_response_support_origins": 20,
}
ALLOWED_OLD_SOURCE_CHANGES = {
    "src/oms/modules/feature/test/CMakeLists.txt",
    "build/Release/compile_commands.json",
    "build/Release/CMakeCache.txt",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path: Path):
    path = path.resolve()
    return {"path": str(path), "sha256": file_hash(path), "size": path.stat().st_size}


def check_record(node, *, allow_missing=False):
    require(type(node.get("path")) is str and type(node.get("sha256")) is str and re.fullmatch("[0-9a-f]{64}", node["sha256"]) is not None, "exact path/SHA256 record required")
    path = Path(node["path"])
    require(path.is_absolute(), "absolute artifact path required")
    if not path.is_file() and allow_missing:
        return None
    require(path.is_file() and file_hash(path) == node["sha256"], f"changed/missing bound artifact: {path}")
    if "size" in node:
        require(type(node["size"]) is int and node["size"] == path.stat().st_size, f"bound artifact size changed: {path}")
    return path


def unique_records(nodes):
    result = {}
    for node in nodes:
        path = str(Path(node["path"]).resolve())
        require(path not in result or result[path]["sha256"] == node["sha256"], f"conflicting artifact identity: {path}")
        result[path] = node
    return [result[path] for path in sorted(result)]


def canonical_method(path, schema):
    body = read_yaml(path)
    anchor = path.with_suffix(path.suffix + ".identity").read_text().strip()
    require(body.get("schema") == schema and body.get("status") == "frozen" and body.get("identity") == anchor, "external canonical frozen method required")
    require(digest({k: v for k, v in body.items() if k != "identity"}) == anchor, "canonical method body identity mismatch")
    return body


def parent_context():
    path = PARENT / "frozen-method.yaml"
    method = canonical_method(path, "h13-native-producer-method-v1")
    require(method["identity"] == PARENT_IDENTITY, "different H13 parent producer; new study version required")
    check_record(method["preparation"])
    preparation = read_yaml(Path(method["preparation"]["path"]))
    check_record(preparation["profile"])
    profile = read_yaml(Path(preparation["profile"]["path"]))
    require(method["preparation_identity"] == digest(preparation) and method["profile_identity"] == digest(profile), "immutable H13 preparation/profile body changed")
    check_record(preparation["producer"])
    source = read_yaml(Path(preparation["producer"]["path"]))
    require(len(source["sources"]) == len(source["live_sources_before_and_after"]) == 1644, "fixed 1644-source H13 native parent required")
    require(source["binary"]["sha256"] == PARENT_BINARY_SHA256, "wrong H13 control binary")
    check_record(source["binary"])
    require(
        profile["carrier_order"] == list(CARRIERS) and profile["fixed_train_days"] == list(DAYS) and profile["target_symbols"] == list(TARGETS),
        "H13 parent population/order changed",
    )
    for node in method["sources"]:
        check_record(node)
    links = [record(path), record(path.with_suffix(".yaml.identity")), method["preparation"], preparation["profile"], preparation["producer"]]
    return preparation, profile, source, links


def fresh_copy(source, target):
    require(not target.exists(), f"fresh capsule destination required: {target}")
    before = record(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    target.chmod(0o555 if os.access(source, os.X_OK) else 0o444)
    frozen = record(target)
    require(frozen["sha256"] == before["sha256"] and record(source) == before, f"source changed while copying: {source}")
    return before, frozen


def source_capsule(parent_source):
    prior_live = {str(Path(node["path"]).relative_to(ROOT)): node for node in parent_source["live_sources_before_and_after"]}
    live, changes = {}, []
    for old in parent_source["sources"]:
        relative = str(Path(old["path"]).relative_to(PARENT / "compiled-source"))
        check_record(old)
        require(prior_live[relative]["sha256"] == old["sha256"], "H13 compiled/live source identity disagrees")
        current = record(ROOT / relative)
        if current["sha256"] != old["sha256"]:
            require(relative in ALLOWED_OLD_SOURCE_CHANGES, f"unaccounted existing native math/source drift: {relative}")
            if relative == "src/oms/modules/feature/test/CMakeLists.txt":
                require((ROOT / relative).read_bytes().startswith(Path(old["path"]).read_bytes()), "existing test CMake may only append the H15 target")
            changes.append({"relative_path": relative, "old_sha256": old["sha256"], "new_sha256": current["sha256"]})
        live[relative] = current
    additions = (f"{LEAF}/peer_trade_information.h", f"{LEAF}/peer_trade_information.cpp", f"{LEAF}/CMakeLists.txt", "src/oms/modules/feature/test/peer_trade_information_test.cpp")
    for relative in additions:
        require(relative not in live, "H15 source already existed in immutable H13 parent")
        live[relative] = record(ROOT / relative)
    commands = json.loads((ROOT / "build/Release/compile_commands.json").read_text())
    require(
        any(str(Path(command["file"]).resolve()) == str(ROOT / LEAF / "peer_trade_information.cpp") for command in commands),
        "H15 translation unit absent from Release compile commands",
    )
    frozen = []
    for relative, node in sorted(live.items()):
        before, after = fresh_copy(Path(node["path"]), OUTPUT / "compiled-source" / relative)
        require(before == node, "source drifted after initial capsule inventory")
        frozen.append(after)
    for node in [*live.values(), *frozen]:
        check_record(node)
    return {
        "schema": "h15-native-source-capsule-v1",
        "parent_producer_identity": PARENT_IDENTITY,
        "parent_binary_sha256": PARENT_BINARY_SHA256,
        "prior_build": record(PARENT / "producer-source-build.yaml"),
        "prior_source_count": 1644,
        "live_sources_before_and_after": list(live.values()),
        "sources": frozen,
        "explicit_non_math_source_changes": changes,
        "new_material_sources": list(additions),
        "existing_native_math_all_sha256_equal": True,
        "source_before_after_all_sha256_equal": True,
    }


def runtime_preflight(parent_preparation):
    check_record(parent_preparation["runtime_snapshot"])
    runtime = read_yaml(Path(parent_preparation["runtime_snapshot"]["path"]))
    dependencies = runtime["dependencies"]
    require(len(dependencies) == 51 and dependencies == parent_preparation["runtime_dependencies"], "fixed shared 51-dependency runtime changed")
    for node in dependencies:
        check_record(node)
        require(file_hash(Path(node["original_system_path"])) == node["sha256"], f"system runtime differs from immutable snapshot: {node['name']}")
    return dependencies, Path(parent_preparation["runtime_snapshot"]["path"]).parent / "runtime"


def native_guide(binary, environment):
    require(len(INFO) > 0 and len(set(INFO)) == len(INFO) and not set(ALPHA) & set(INFO), "final exact H15 Info declaration required before preparation")
    target = OUTPUT / "feature-guide.yaml"
    require(not target.exists(), "fresh feature-guide artifact required")
    result = subprocess.run([str(binary), "feature-guide"], env=environment, capture_output=True, check=True)
    target.write_bytes(result.stdout)
    guide = read_yaml(target)
    modules = [entry for entry in guide["module_types"] if entry["type"] == MODULE]
    require(len(modules) == 1, "exactly one native PeerTradeInformation registration required")
    families = modules[0]["feature_families"]
    require(len(families) == len(ALPHA) + len(INFO), "H15 native family count changed")
    require({f["name"] for f in families if "alpha_factor" in f["attributes"]} == set(ALPHA), "native guide differs from exact eleven-Alpha declaration")
    require({f["name"] for f in families if "alpha_factor" not in f["attributes"]} == set(INFO), "native guide differs from exact Info declaration")
    require({f["name"] for f in families if "categorical" in f["attributes"]} == set(ALPHA_CATEGORICAL), "native J/K categorical declaration changed")
    return record(target), modules[0]


def daily_selection(day, raw, basic, parents):
    missing = []
    for symbol in CARRIERS:
        if check_record(raw[(day, symbol)], allow_missing=True) is None:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_carrier_BIN_skip_whole_day_no_fallback"})
    if check_record(basic, allow_missing=True) is None:
        missing.append({"day": day, "reason": "missing_basic_info_skip_whole_day_no_replacement"})
        return [], missing
    with Path(basic["path"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))  # contract metadata only
    for symbol in TARGETS:
        selected = [row for row in rows if row["symbol"] == symbol]
        require(len(selected) <= 1, "duplicate required pair-leg contract metadata")
        if not selected:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_pair_leg_contract_skip_whole_day_no_replacement"})
    if missing:
        return [], missing
    present = []
    for parent in parents:
        if parent["day"] != day:
            continue
        if any(check_record(parent[field], allow_missing=True) is None for field in ("parent_origins", "expected_native_keys")):
            missing.append({"day": day, "symbol": parent["symbol"], "reason": "missing_original_native_keys_skip_target_writer_no_replacement"})
        else:
            present.append(parent["symbol"])
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
        # Both source filters remain, even if only one target writer is eligible.
        if symbol in TARGETS:
            declarations.insert(0, {"Desc": "TwseFilter.0", "Spec": {"RequireTradable": False, "StatusFilter": "", "Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}})
        if symbol in present:
            peer = PEERS[symbol]
            declarations.append(
                {
                    "Desc": f"{MODULE}.0",
                    "Spec": {
                        "TargetSymbol": symbol,
                        "PeerSymbol": peer,
                        "StatusFilter": "",
                        "Dep": {"Book": [f"TwseFilter.0@{symbol}", f"TwseFilter.0@{peer}"], "Trade": [f"TwseFilter.0@{symbol}", f"TwseFilter.0@{peer}"]},
                    },
                }
            )
            metadata = [
                {"Feature": f"CurrentBook.0.{family}.0@{symbol}", "Name": name}
                for family, name in (
                    ("book_mid_price", "OriginMidPrice"),
                    ("book_mid_ticks", "OriginMidTicks"),
                    ("book_bid_ticks", "OriginBidTicks"),
                    ("book_ask_ticks", "OriginAskTicks"),
                )
            ]
            metadata += [{"Feature": f"{MODULE}.0.{family}.0@{symbol}", "Name": f"H15_{family}"} for family in INFO]
            for writer_id, folder, sampler in (
                (0, "data", {"PeriodicSampler": {"StartTime": "091000", "UntilTime": "130000", "SampleInterval": "10s"}}),
                (1, "events", {"BookFlipSampler": {"Mode": "all_book", "MinInterval": "0s", "StartTime": "091000", "UntilTime": "130000"}}),
            ):
                declarations.append(
                    {
                        "Desc": f"DatasetWriter.{writer_id}",
                        "Spec": {
                            "Subscribe": [{"Book": [symbol]}],
                            "Exports": [f"{MODULE}.0.{family}.0@{symbol}" for family in ALPHA],
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
    require([g["Gid"] for g in groups if g["Gid"]] == list(CARRIERS), "fixed sixteen carrier order changed")
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def job(day, work, binary, config, symbols, kind, runtime):
    return {
        "kind": kind,
        "day": day,
        "symbols": list(symbols),
        "target_peer_pairs": [{"target": symbol, "peer": PEERS[symbol]} for symbol in symbols],
        "requires_replay": bool(symbols),
        "work": str(work),
        "config": record(config),
        "command": [str(binary), "-d", day, "-C", str(work), "--trading-calendar", str(CALENDAR), "--run-status-dir", str(work / "status"), str(config)] if symbols else [],
        "environment": {"LD_LIBRARY_PATH": str(runtime), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "outputs": [str(work / folder / day / symbol / "values.parquet") for folder in ("data", "events") for symbol in symbols],
    }


def parent_control_inputs(parent_preparation):
    path = PARENT / "frozen-support-evaluation-method.yaml"
    method = canonical_method(path, "h13-native-support-evaluation-method-v1")
    check_record(method["profile"])
    profile = read_yaml(Path(method["profile"]["path"]))
    require(
        method["profile_identity"] == digest(profile) and profile["producer_identity"] == PARENT_IDENTITY, "H13 control output identity differs from frozen evaluation metadata"
    )
    source_job = next(j for j in parent_preparation["jobs"] if j["day"] == DAYS[0])
    cells = {(c["day"], c["symbol"]): c for c in profile["cells"]}
    outputs = []
    for folder in ("data", "events"):
        for symbol in TARGETS:
            cell = cells[(DAYS[0], symbol)]
            require(cell["status"] == "observed_native", "both original H13 pair writer outputs required for byte control")
            node = cell[folder]
            require(node["path"] == str(Path(source_job["work"]) / folder / DAYS[0] / symbol / "values.parquet"), "H13 original control writer path changed")
            check_record(node)
            outputs.append(node)
    return source_job, outputs, [record(path), record(path.with_suffix(".yaml.identity")), method["profile"]]


def prepare():
    receipt_path = OUTPUT / "preparation-receipt.yaml"
    require(
        not any(path.exists() for path in (receipt_path, PROFILE, OUTPUT / "coco", OUTPUT / "compiled-source")), "fresh H15 artifacts required; never overwrite/resume preparation"
    )
    require(INFO, "final native Info fields must be reviewed before --prepare")
    parent_preparation, parent_profile, parent_source, parent_links = parent_context()
    source_job, controls, control_links = parent_control_inputs(parent_preparation)
    template = Path(source_job["config"]["path"])
    check_record(source_job["config"])
    original = read_yaml(template)
    require([g["Gid"] for g in original["Modules"] if g["Gid"]] == list(CARRIERS), "H13 template carrier/source order changed")
    raw = {(n["day"], n["symbol"]): n for n in parent_preparation["inputs"] if n["path"].endswith(".bin.zst") and n.get("day") in DAYS}
    require(set(raw) == {(day, symbol) for day in DAYS for symbol in CARRIERS}, "exact fixed32 original BIN identities required; no fallback")
    basics = {Path(n["path"]).stem: n for n in parent_preparation["inputs"] if Path(n["path"]).parent == Path("/mnt/data0/contract/tse/stock") and Path(n["path"]).stem in DAYS}
    require(set(basics) == set(DAYS), "two fixed BasicInfo identities required")
    parents = copy.deepcopy(parent_profile["parents"])
    require([(c["day"], c["symbol"]) for c in parents] == [(d, s) for d in DAYS for s in TARGETS], "four original native parent/key identities changed")
    require(
        all(
            type(c["expected_native_rows"]) is int and c["expected_native_rows"] == 1381 and type(c["expected_support_rows"]) is int and c["expected_support_rows"] == 461
            for c in parents
        ),
        "typed original native cohort changed",
    )
    source = source_capsule(parent_source)
    dependencies, runtime = runtime_preflight(parent_preparation)
    original_binary, binary = fresh_copy(ROOT / "build/Release/src/app/coco", OUTPUT / "coco")
    environment = {**os.environ, "LD_LIBRARY_PATH": str(runtime)}
    guide, module = native_guide(Path(binary["path"]), environment)
    dynamic = subprocess.run(["ldd", binary["path"]], env=environment, capture_output=True, text=True, check=True).stdout
    names = set(re.findall(r"^\s*(\S+)\s+=>", dynamic, re.MULTILINE))
    require(
        names == {d["name"] for d in dependencies if not d["name"].startswith("ld-linux")} and "not found" not in dynamic,
        "new binary dynamic dependencies differ from frozen51 closure",
    )
    (OUTPUT / "ldd.txt").write_text(dynamic)
    jobs, missing = [], []
    for day in DAYS:
        present, skips = daily_selection(day, raw, basics[day], parents)
        missing.extend(skips)
        work = OUTPUT / "pilot" / day
        work.mkdir(parents=True)
        config = work / "config.yaml"
        write_yaml(config, daily_config(original, present))
        jobs.append(job(day, work, Path(binary["path"]), config, present, "h15_fixed_two_leg_day", runtime))
    work = OUTPUT / "registration-control/20260119"
    work.mkdir(parents=True)
    control_config = work / "config.yaml"
    control_config.write_bytes(template.read_bytes())
    require(file_hash(control_config) == source_job["config"]["sha256"], "H13 registration-control YAML bytes changed")
    control = job(DAYS[0], work, Path(binary["path"]), control_config, TARGETS, "h13_registration_whole_byte_control", runtime)
    control.update(old_config=source_job["config"], old_binary=parent_source["binary"], required_byte_equal_outputs=controls)
    source.update(binary=binary, live_built_binary_before_and_after=original_binary)
    write_yaml(OUTPUT / "producer-source-build.yaml", source)
    inputs = unique_records(
        [
            *parent_links,
            *control_links,
            record(CALENDAR),
            record(template),
            *raw.values(),
            *basics.values(),
            *[c[field] for c in parents for field in ("parent_origins", "expected_native_keys")],
            *controls,
            parent_source["binary"],
            record(PLAN),
        ]
    )
    optional_paths = {n["path"] for n in [*raw.values(), *basics.values(), *[c[field] for c in parents for field in ("parent_origins", "expected_native_keys")]]}
    absent = []
    for node in inputs:
        if check_record(node, allow_missing=node["path"] in optional_paths) is None:
            absent.append(node["path"])
    profile = {
        "schema": "h15-native-source-support-profile-v1",
        "status": "prospective_requires_root_freeze",
        "fixed_train_days": list(DAYS),
        "target_symbols": list(TARGETS),
        "target_peer_pairs": [{"target": symbol, "peer": PEERS[symbol]} for symbol in TARGETS],
        "carrier_order": list(CARRIERS),
        "contract": CONTRACT,
        "alpha_fields": list(ALPHA),
        "alpha_numeric_fields": list(ALPHA_NUMERIC),
        "alpha_categorical_fields": list(ALPHA_CATEGORICAL),
        "j_alpha_fields": [*ALPHA_NUMERIC, *ALPHA_CATEGORICAL[:2]],
        "k_alpha_fields": [*ALPHA_NUMERIC, *ALPHA_CATEGORICAL[2:]],
        "info_fields": list(INFO),
        "labelers": [],
        "parents": parents,
        "missing": missing,
        "required_absent_inputs": sorted(absent),
        "routing": "Raw CurrentBook and raw Book writers; both explicit legs via TwseFilter Book/Trade RequireTradable:false empty scalar StatusFilter; module status/halt epoch guards",
        "origin_contract": "Original1381 native10s int64keys/rawmiduint64bits; actual existing461 S%30s origins; no labels/grid invention",
        "event_contract": "all_book receive snapshots with sequence and ties; not every callback/cluster; per-leg native closed availability only",
        "session_contract": "Native session counters require sourceA>=09:10 and qualificationA'<13:00, sampled at actual periodic boundaries; warm-in source facts excluded;13:00 remains support origin",
        "support_contract": "Every present fixed cell5 session qualified marks+5 unique fresh target-baselined recentIDs(0<=S-A<=300s); >=1 observed; global5eachdirection+20 exact native ordered_response_support origins",
        "absence_contract": "Missing fixed carrierBIN/dayBasicInfo/pairlegcontract skips whole daily replay; no loaderfallback/replacement; missing originaltargetkeys skips only its writer",
        "j_k_contract": "Seven numeric arrays emitted once, shared bit forbit; equal two-category capacity; K does not remove existing joined numeric history",
        "overflow_contract": "Fixed256 callback bound perleg openE; reject whole unknown/mixed/overflow attribution; no partialmark/EOFflush/timerclose",
        "source_roles": "Native source support only; causal native receive-prefix and fresh frozen FE judge required before predictive comparison",
        "native_plan": record(PLAN),
    }
    write_yaml(PROFILE, profile)
    receipt = {
        "schema": "h15-native-preparation-v1",
        "script": record(Path(__file__)),
        "profile": record(PROFILE),
        "profile_identity": digest(profile),
        "native_plan": record(PLAN),
        "parent_producer_identity": PARENT_IDENTITY,
        "producer": record(OUTPUT / "producer-source-build.yaml"),
        "binary": binary,
        "guide": guide,
        "native_module_registration": module,
        "runtime_snapshot": parent_preparation["runtime_snapshot"],
        "runtime_dependencies": dependencies,
        "runtime_system_paths_match_frozen_sha256": True,
        "runtime_directory": str(runtime),
        "ldd": record(OUTPUT / "ldd.txt"),
        "sources": unique_records([*source["sources"], record(Path(__file__)), record(ROOT / "AstraResearch/astra/io.py"), record(PLAN)]),
        "inputs": inputs,
        "jobs": jobs,
        "registration_control": control,
        "execution_order": ["registration_control", *DAYS],
        "native_replays_launched": 0,
        "labels_read": False,
        "model_fits": 0,
        "requires_root_frozen_method_before_replay": True,
    }
    for node in [*source["live_sources_before_and_after"], original_binary, *parent_links, *control_links]:
        check_record(node)
    runtime_preflight(parent_preparation)
    write_yaml(receipt_path, receipt)
    return {"prepared_only": True, "daily_jobs": len(jobs), "registration_controls": 1, "sources": len(receipt["sources"]), "inputs": len(inputs)}


def preflight(method_path, job_key=None):
    frozen = canonical_method(method_path, "h15-native-producer-method-v1")
    preparation_path = OUTPUT / "preparation-receipt.yaml"
    check_record(frozen["preparation"])
    require(Path(frozen["preparation"]["path"]).resolve() == preparation_path, "wrong H15 preparation bound")
    preparation = read_yaml(preparation_path)
    check_record(preparation["profile"])
    profile = read_yaml(PROFILE)
    require(
        frozen["preparation_identity"] == digest(preparation) and frozen["profile_identity"] == digest(profile) == preparation["profile_identity"],
        "frozen preparation/profile body changed",
    )
    require(profile["contract"] == CONTRACT and all(type(n) is int for n in profile["contract"].values()), "typed sampling/support gates changed")
    require(
        profile["alpha_fields"] == list(ALPHA) and profile["info_fields"] == list(INFO) and profile["alpha_categorical_fields"] == list(ALPHA_CATEGORICAL),
        "exact H15 Alpha/Info contract changed",
    )
    require(
        profile["fixed_train_days"] == list(DAYS) and profile["target_symbols"] == list(TARGETS) and profile["carrier_order"] == list(CARRIERS),
        "fixed population/source order changed",
    )
    require(profile["target_peer_pairs"] == [{"target": symbol, "peer": PEERS[symbol]} for symbol in TARGETS], "fixed pair roles changed")
    require(preparation["script"] == record(Path(__file__)), "running preparation script differs from frozen producer")
    check_record(preparation["producer"])
    capsule = read_yaml(Path(preparation["producer"]["path"]))
    require(
        capsule["schema"] == "h15-native-source-capsule-v1"
        and capsule["parent_producer_identity"] == PARENT_IDENTITY
        and capsule["prior_source_count"] == 1644
        and len(capsule["sources"]) == 1648
        and capsule["existing_native_math_all_sha256_equal"] is True
        and capsule["source_before_after_all_sha256_equal"] is True,
        "closed H15 source capsule required",
    )
    required_sources = unique_records(preparation["sources"])
    source_by_path = {node["path"]: node for node in required_sources}
    essential_sources = [record(Path(__file__)), record(ROOT / "AstraResearch/astra/io.py"), record(PLAN), *capsule["sources"]]
    require(all(source_by_path.get(n["path"], {}).get("sha256") == n["sha256"] for n in essential_sources), "essential running helper/plan/compiled source closure omitted")
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
            preparation["registration_control"]["config"],
            record(preparation_path),
        ]
    )
    for key, nodes in (("sources", required_sources), ("inputs", required_inputs)):
        bound = {str(Path(n["path"]).resolve()): n for n in frozen[key]}
        require(all(bound.get(n["path"], {}).get("sha256") == n["sha256"] for n in nodes), f"frozen method omitted mandatory {key} closure")
        for node in frozen[key]:
            absent = node["path"] in profile["required_absent_inputs"]
            path = check_record(node, allow_missing=absent)
            require(not absent or path is None, "previously absent fixed input appeared; new version required")
    parent_preparation, _, _, _ = parent_context()
    _, runtime = runtime_preflight(parent_preparation)
    jobs = {"registration_control": preparation["registration_control"], **{j["day"]: j for j in preparation["jobs"]}}
    require(job_key is None or job_key in jobs, "unknown frozen job key")
    selected = jobs.values() if job_key is None else [jobs[job_key]]
    for native_job in selected:
        require(type(native_job["requires_replay"]) is bool and native_job["requires_replay"] == bool(native_job["symbols"]), "native missing-day replay status changed")
        if not native_job["requires_replay"]:
            require(native_job["command"] == [] and native_job["outputs"] == [], "declared missing day cannot run fallback replay")
            continue
        require(native_job["command"][0] == preparation["binary"]["path"] and native_job["environment"]["LD_LIBRARY_PATH"] == str(runtime), "job escaped immutable binary/runtime")
        require(
            not any(Path(path).exists() for path in native_job["outputs"]) and not (Path(native_job["work"]) / "status").exists(), "fresh job outputs/status required before replay"
        )
    return {"preflight_passed": True, "method_identity": frozen["identity"], "job": job_key, "native_replays_launched": 0, "model_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", type=Path, metavar="ROOT_FROZEN_METHOD_YAML")
    parser.add_argument("--job", choices=("registration_control", *DAYS))
    args = parser.parse_args()
    if args.prepare and args.job:
        parser.error("--job requires --preflight")
    print(prepare() if args.prepare else preflight(args.preflight.resolve(), args.job))


if __name__ == "__main__":
    main()
