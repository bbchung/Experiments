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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
PARENT = OUTPUT.parent / "native-h13-quote-visit"
EXPERIMENT = ROOT / "AstraResearch/experiments/information_state_20260930"
PROFILE = EXPERIMENT / "h15-native-profile-v2.yaml"
PLAN = EXPERIMENT / "H15_CONFIG_METHOD_V2.md"
NATIVE_PLAN = EXPERIMENT / "H15_TRADE_LED_NATIVE_PLAN.md"
V1 = OUTPUT.parent / "native-h15-peer-trade"
V1_IDENTITY = "a2f2bbb1da368d755bf098a1e3e5d3b3e85c6cfd164e720cf7b1deab230f78b5"
BINARY_SHA256 = "613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b"
ACCEPT_ALL = "TRIAL || !TRIAL"
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
NUMERIC = ALPHA_NUMERIC
CATEGORICAL = ALPHA_CATEGORICAL
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


def require_closure(frozen, scope, nodes):
    require(scope in ("sources", "inputs"), "exact source/input closure scope required")
    bound = {str(Path(n["path"]).resolve()): n for n in unique_records(frozen[scope])}
    require(all(bound.get(str(Path(n["path"]).resolve()), {}).get("sha256") == n["sha256"] for n in nodes), f"frozen method omitted mandatory {scope} closure")


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


def v1_context():
    path = V1 / "frozen-method.yaml"
    method = canonical_method(path, "h15-native-producer-method-v1")
    require(method["identity"] == V1_IDENTITY, "fixed H15 V1 producer identity changed")
    check_record(method["preparation"])
    preparation = read_yaml(Path(method["preparation"]["path"]))
    for name in ("profile", "producer", "binary", "guide", "ldd", "script"):
        check_record(preparation[name])
    profile = read_yaml(Path(preparation["profile"]["path"]))
    require(
        digest(preparation) == method["preparation_identity"] and digest(profile) == method["profile_identity"] == preparation["profile_identity"], "H15 V1 immutable body changed"
    )
    require(preparation["binary"]["sha256"] == BINARY_SHA256, "different H15 native binary requires new scientific method")
    for node in method["sources"]:
        check_record(node)
    links = [
        record(path),
        record(path.with_suffix(".yaml.identity")),
        method["preparation"],
        *[preparation[k] for k in ("profile", "producer", "binary", "guide", "ldd", "script")],
    ]
    return preparation, profile, read_yaml(Path(preparation["producer"]["path"])), links


def validate_failed_startup(failure, preparation):
    require(
        failure.get("schema") == "h15-native-startup-failure-v1"
        and failure.get("producer_identity") == V1_IDENTITY
        and failure.get("binary_sha256") == BINARY_SHA256
        and failure.get("frozen_old_artifacts_modified") is False
        and failure.get("labels_read") is False
        and type(failure.get("model_fits")) is int
        and failure["model_fits"] == 0
        and type(failure.get("native_h15_output_artifacts")) is int
        and failure["native_h15_output_artifacts"] == 0
        and failure.get("native_h15_outputs_decoded") is False
        and failure.get("remaining_day_not_launched") == DAYS[1],
        "V1 config-only pre-output failure proof required",
    )
    failed = failure["startup_failure"]
    require(failed.get("day") == DAYS[0] and failed.get("error") == "invalid StatusFilter expression empty string", "different startup failure is outside V2 correction")
    for name in ("execution", "log", "status"):
        check_record(failed[name])
    execution = read_yaml(Path(failed["execution"]["path"]))
    original = next(j for j in preparation["jobs"] if j["day"] == DAYS[0])
    require(
        execution.get("method_identity") == V1_IDENTITY
        and execution.get("job") == DAYS[0]
        and type(execution.get("exit_code")) is int
        and execution["exit_code"] == 1
        and execution.get("outputs") == []
        and execution.get("command") == original["command"]
        and execution.get("log", {}).get("sha256") == failed["log"]["sha256"],
        "failed startup execution receipt differs from bound V1 job",
    )
    require("invalid StatusFilter expression ''" in Path(failed["log"]["path"]).read_text(), "native parser empty-expression failure absent from log")
    return [failed[name] for name in ("execution", "log", "status")]


def reused_control(preparation, failure):
    original = preparation["registration_control"]
    node = failure["registration_control"]["execution"]
    require(failure["registration_control"].get("whole_four_h13_parquet_bytes_equal") is True, "V1 registration whole-byte proof required")
    require(Path(node["path"]).resolve() == Path(original["work"]) / "execution-receipt.yaml", "registration execution path does not belong to bound V1 job")
    check_record(node)
    execution = read_yaml(Path(node["path"]))
    require(
        execution.get("method_identity") == V1_IDENTITY
        and execution.get("job") == "registration_control"
        and type(execution.get("exit_code")) is int
        and execution["exit_code"] == 0
        and execution.get("command") == original["command"]
        and [n["path"] for n in execution.get("outputs", [])] == original["outputs"]
        and len(execution["outputs"]) == len(original["required_byte_equal_outputs"]) == 4,
        "successful exact four-output V1 registration receipt required",
    )
    for output, expected in zip(execution["outputs"], original["required_byte_equal_outputs"], strict=True):
        check_record(output)
        check_record(expected)
        require(output["sha256"] == expected["sha256"], "registration output is not byte-equal to original H13 control")
    check_record(execution["log"])
    for name in ("config", "old_config", "old_binary"):
        check_record(original[name])
    control = copy.deepcopy(original)
    control.update(
        kind="reused_h13_registration_whole_byte_control",
        requires_replay=False,
        command=[],
        reused_command=original["command"],
        reused_execution_receipt=node,
        reuse_existing_outputs=True,
    )
    links = [node, execution["log"], *execution["outputs"], *original["required_byte_equal_outputs"], *[original[k] for k in ("config", "old_config", "old_binary")]]
    return control, links


def source_capsule(prior):
    require(
        prior.get("schema") == "h15-native-source-capsule-v1" and len(prior["sources"]) == len(prior["live_sources_before_and_after"]) == 1648,
        "fixed shared V1 source capsule required",
    )
    require(prior["existing_native_math_all_sha256_equal"] is True and prior["source_before_after_all_sha256_equal"] is True, "V1 native math/source proof required")
    by_relative = {str(Path(n["path"]).relative_to(ROOT)): n for n in prior["live_sources_before_and_after"]}
    for node in prior["sources"]:
        check_record(node)
        relative = str(Path(node["path"]).relative_to(V1 / "compiled-source"))
        require(by_relative[relative]["sha256"] == node["sha256"], "shared compiled/live source mismatch")
        check_record(by_relative[relative])
    result = copy.deepcopy(prior)
    result.update(
        schema="h15-native-source-capsule-v2",
        shared_v1_producer_identity=V1_IDENTITY,
        shared_v1_capsule=record(V1 / "producer-source-build.yaml"),
        native_source_copies_reused=True,
    )
    check_record(result["binary"])
    check_record(result["live_built_binary_before_and_after"])
    require(result["binary"]["sha256"] == result["live_built_binary_before_and_after"]["sha256"] == BINARY_SHA256, "live or shared native binary changed")
    return result


def runtime_preflight(parent_preparation):
    check_record(parent_preparation["runtime_snapshot"])
    runtime = read_yaml(Path(parent_preparation["runtime_snapshot"]["path"]))
    dependencies = runtime["dependencies"]
    require(len(dependencies) == 51 and dependencies == parent_preparation["runtime_dependencies"], "fixed shared 51-dependency runtime changed")
    for node in dependencies:
        check_record(node)
        require(file_hash(Path(node["original_system_path"])) == node["sha256"], f"system runtime differs from immutable snapshot: {node['name']}")
    return dependencies, Path(parent_preparation["runtime_snapshot"]["path"]).parent / "runtime"


def shared_native_guide(preparation):
    target = check_record(preparation["guide"])
    guide = read_yaml(target)
    modules = [entry for entry in guide["module_types"] if entry["type"] == MODULE]
    require(len(modules) == 1, "exactly one native PeerTradeInformation registration required")
    families = modules[0]["feature_families"]
    require(len(families) == len(ALPHA) + len(INFO), "H15 native family count changed")
    require({f["name"] for f in families if "alpha_factor" in f["attributes"]} == set(ALPHA), "native guide differs from exact eleven-Alpha declaration")
    require({f["name"] for f in families if "alpha_factor" not in f["attributes"]} == set(INFO), "native guide differs from exact Info declaration")
    require({f["name"] for f in families if "categorical" in f["attributes"]} == set(ALPHA_CATEGORICAL), "native J/K categorical declaration changed")
    require(modules[0] == preparation["native_module_registration"], "shared native registration metadata changed")
    return preparation["guide"], modules[0]


def status_filter_proof(capsule):
    paths = ("src/oms/model/quote.cpp", "src/oms/test/status_filter_test.cpp")
    by_path = {str(Path(n["path"]).relative_to(V1 / "compiled-source")): n for n in capsule["sources"]}
    nodes = [by_path[path] for path in paths]
    for node in nodes:
        check_record(node)
    parser, tests = (Path(n["path"]).read_text() for n in nodes)
    require(
        all(token in parser for token in ('while (match("||"))', 'if (match("!"))', "static_cast<uint16_t>(~parse_unary())", "index < 16", "expected status or"))
        and all(token in tests for token in ("EvaluatesBooleanStatusExpression", "DefaultAcceptsEveryStatus", "TRIAL || AUCTION || SUSPEND")),
        "native Boolean parser/test source proof incomplete",
    )
    return {
        "expression": ACCEPT_ALL,
        "status_classes": 16,
        "accept_all_mask": 65535,
        "sources": nodes,
        "proof": "Native uint16 truth-table OR with complement: T | uint16(~T) == 0xffff for all16 native status classes; empty scalar is invalid grammar.",
    }


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
            declarations.insert(0, {"Desc": "TwseFilter.0", "Spec": {"RequireTradable": False, "StatusFilter": ACCEPT_ALL, "Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}})
        if symbol in present:
            peer = PEERS[symbol]
            declarations.append(
                {
                    "Desc": f"{MODULE}.0",
                    "Spec": {
                        "TargetSymbol": symbol,
                        "PeerSymbol": peer,
                        "StatusFilter": ACCEPT_ALL,
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


def validate_daily_config(config, original, present):
    require(config == daily_config(original, present), "daily YAML escaped exact V2 routing/export/sampling contract")
    filtered = [d for g in config["Modules"] for d in g["Decl"] if d["Desc"] in ("TwseFilter.0", f"{MODULE}.0")]
    require(all(d["Spec"].get("StatusFilter") == ACCEPT_ALL for d in filtered), "native status grammar requires exact explicit accept-all tautology")
    require(len(filtered) == 2 + len(present), "both fixed peer-leg filters required")


def validate_profile(profile, original):
    fixed = set(original) - {"schema", "status", "routing", "missing", "required_absent_inputs", "native_plan"}
    require(all(profile.get(key) == original[key] for key in fixed), "V1 support, representation, cohort or causal rules changed")
    require(
        profile.get("schema") == "h15-native-source-support-profile-v2" and profile.get("status") == "prospective_requires_root_freeze", "exact V2 profile schema/status required"
    )
    require(profile.get("status_filter_expression") == ACCEPT_ALL and profile.get("shared_h15_v1_producer_identity") == V1_IDENTITY, "V2 config-only lineage changed")
    require(profile.get("contract") == CONTRACT and all(type(n) is int for n in profile["contract"].values()), "typed frozen support gate changed")
    require(profile.get("native_plan") == record(NATIVE_PLAN) and profile.get("config_method_plan") == record(PLAN), "prospective native/config method plan changed")


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
    v1_preparation, v1_profile, v1_source, v1_links = v1_context()
    failure_path = V1 / "startup-failure.yaml"
    failure = read_yaml(failure_path)
    failure_links = [record(failure_path), *validate_failed_startup(failure, v1_preparation)]
    control, registration_links = reused_control(v1_preparation, failure)
    source = source_capsule(v1_source)
    grammar = status_filter_proof(source)
    dependencies, runtime = runtime_preflight(parent_preparation)
    binary = v1_preparation["binary"]
    original_binary = source["live_built_binary_before_and_after"]
    guide, module = shared_native_guide(v1_preparation)
    dynamic = Path(v1_preparation["ldd"]["path"]).read_text()
    names = set(re.findall(r"^\s*(\S+)\s+=>", dynamic, re.MULTILINE))
    require(
        names == {d["name"] for d in dependencies if not d["name"].startswith("ld-linux")} and "not found" not in dynamic,
        "shared binary dynamic dependencies differ from frozen51 closure",
    )
    jobs, missing = [], []
    for day in DAYS:
        present, skips = daily_selection(day, raw, basics[day], parents)
        missing.extend(skips)
        work = OUTPUT / "pilot" / day
        work.mkdir(parents=True)
        config = work / "config.yaml"
        generated = daily_config(original, present)
        validate_daily_config(generated, original, present)
        write_yaml(config, generated)
        jobs.append(job(day, work, Path(binary["path"]), config, present, "h15_fixed_two_leg_day", runtime))
    write_yaml(OUTPUT / "producer-source-build.yaml", source)
    inputs = unique_records(
        [
            *parent_links,
            *control_links,
            *v1_links,
            *failure_links,
            *registration_links,
            record(CALENDAR),
            record(template),
            *raw.values(),
            *basics.values(),
            *[c[field] for c in parents for field in ("parent_origins", "expected_native_keys")],
            *controls,
            parent_source["binary"],
            record(PLAN),
            record(NATIVE_PLAN),
        ]
    )
    optional_paths = {n["path"] for n in [*raw.values(), *basics.values(), *[c[field] for c in parents for field in ("parent_origins", "expected_native_keys")]]}
    absent = []
    for node in inputs:
        if check_record(node, allow_missing=node["path"] in optional_paths) is None:
            absent.append(node["path"])
    profile = copy.deepcopy(v1_profile)
    profile.update(
        {
            "schema": "h15-native-source-support-profile-v2",
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
            "routing": "Raw CurrentBook and raw Book writers; both explicit legs via TwseFilter Book/Trade RequireTradable:false explicit StatusFilter 'TRIAL || !TRIAL'; module status/halt epoch guards",
            "origin_contract": "Original1381 native10s int64keys/rawmiduint64bits; actual existing461 S%30s origins; no labels/grid invention",
            "event_contract": "all_book receive snapshots with sequence and ties; not every callback/cluster; per-leg native closed availability only",
            "session_contract": "Native session counters require sourceA>=09:10 and qualificationA'<13:00, sampled at actual periodic boundaries; warm-in source facts excluded;13:00 remains support origin",
            "support_contract": "Every present fixed cell5 session qualified marks+5 unique fresh target-baselined recentIDs(0<=S-A<=300s); >=1 observed; global5eachdirection+20 exact native ordered_response_support origins",
            "absence_contract": "Missing fixed carrierBIN/dayBasicInfo/pairlegcontract skips whole daily replay; no loaderfallback/replacement; missing originaltargetkeys skips only its writer",
            "j_k_contract": "Seven numeric arrays emitted once, shared bit forbit; equal two-category capacity; K does not remove existing joined numeric history",
            "overflow_contract": "Fixed256 callback bound perleg openE; reject whole unknown/mixed/overflow attribution; no partialmark/EOFflush/timerclose",
            "source_roles": "Native source support only; causal native receive-prefix and fresh frozen FE judge required before predictive comparison",
            "native_plan": record(NATIVE_PLAN),
            "config_method_plan": record(PLAN),
            "status_filter_expression": ACCEPT_ALL,
            "status_filter_grammar_proof": grammar,
            "shared_h15_v1_producer_identity": V1_IDENTITY,
            "v1_startup_failure": record(failure_path),
            "registration_control_reused_without_replay": True,
        }
    )
    validate_profile(profile, v1_profile)
    write_yaml(PROFILE, profile)
    receipt = {
        "schema": "h15-native-preparation-v2",
        "script": record(Path(__file__)),
        "profile": record(PROFILE),
        "profile_identity": digest(profile),
        "native_plan": record(NATIVE_PLAN),
        "config_method_plan": record(PLAN),
        "status_filter_grammar_proof": grammar,
        "shared_h15_v1_producer_identity": V1_IDENTITY,
        "v1_startup_failure": record(failure_path),
        "parent_producer_identity": PARENT_IDENTITY,
        "producer": record(OUTPUT / "producer-source-build.yaml"),
        "binary": binary,
        "guide": guide,
        "native_module_registration": module,
        "runtime_snapshot": parent_preparation["runtime_snapshot"],
        "runtime_dependencies": dependencies,
        "runtime_system_paths_match_frozen_sha256": True,
        "runtime_directory": str(runtime),
        "ldd": v1_preparation["ldd"],
        "sources": unique_records(
            [*source["sources"], record(Path(__file__)), record(OUTPUT / "test_prepare_native.py"), record(ROOT / "AstraResearch/astra/io.py"), record(PLAN), record(NATIVE_PLAN)]
        ),
        "inputs": inputs,
        "jobs": jobs,
        "registration_control": control,
        "execution_order": list(DAYS),
        "native_replays_launched": 0,
        "labels_read": False,
        "model_fits": 0,
        "requires_root_frozen_method_before_replay": True,
    }
    for node in [*source["live_sources_before_and_after"], original_binary, *parent_links, *control_links, *v1_links, *failure_links, *registration_links]:
        check_record(node)
    runtime_preflight(parent_preparation)
    write_yaml(receipt_path, receipt)
    return {
        "prepared_only": True,
        "daily_jobs": len(jobs),
        "registration_controls_launched": 0,
        "registration_control_reused": True,
        "sources": len(receipt["sources"]),
        "inputs": len(inputs),
    }


def preflight(method_path, job_key=None):
    frozen = canonical_method(method_path, "h15-native-producer-method-v2")
    preparation_path = OUTPUT / "preparation-receipt.yaml"
    check_record(frozen["preparation"])
    require(Path(frozen["preparation"]["path"]).resolve() == preparation_path, "wrong H15 preparation bound")
    preparation = read_yaml(preparation_path)
    require(preparation.get("schema") == "h15-native-preparation-v2" and preparation.get("shared_h15_v1_producer_identity") == V1_IDENTITY, "exact V2 preparation lineage required")
    require(
        preparation.get("native_replays_launched") == 0
        and type(preparation.get("native_replays_launched")) is int
        and preparation.get("labels_read") is False
        and type(preparation.get("model_fits")) is int
        and preparation["model_fits"] == 0
        and preparation.get("requires_root_frozen_method_before_replay") is True,
        "preparation must precede native outputs, label access and model fits",
    )
    check_record(preparation["profile"])
    require(Path(preparation["profile"]["path"]).resolve() == PROFILE, "wrong V2 profile pointer")
    profile = read_yaml(PROFILE)
    require(
        frozen["preparation_identity"] == digest(preparation) and frozen["profile_identity"] == digest(profile) == preparation["profile_identity"],
        "frozen preparation/profile body changed",
    )
    v1_preparation, v1_profile, v1_source, v1_links = v1_context()
    validate_profile(profile, v1_profile)
    failure_path = V1 / "startup-failure.yaml"
    require(preparation["v1_startup_failure"] == profile["v1_startup_failure"] == record(failure_path), "V1 failure proof pointer/identity changed")
    failure = read_yaml(failure_path)
    failure_links = [record(failure_path), *validate_failed_startup(failure, v1_preparation)]
    control, registration_links = reused_control(v1_preparation, failure)
    require(preparation["registration_control"] == control and preparation["execution_order"] == list(DAYS), "reused registration control cannot be changed or launched")
    require(preparation["binary"] == v1_preparation["binary"] and preparation["binary"]["sha256"] == BINARY_SHA256, "V2 must share exact V1 binary")
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
        capsule["schema"] == "h15-native-source-capsule-v2"
        and capsule["parent_producer_identity"] == PARENT_IDENTITY
        and capsule["prior_source_count"] == 1644
        and len(capsule["sources"]) == 1648
        and capsule["existing_native_math_all_sha256_equal"] is True
        and capsule["source_before_after_all_sha256_equal"] is True,
        "closed H15 source capsule required",
    )
    require(capsule == source_capsule(v1_source), "shared V1 capsule or live compiled source changed")
    grammar = status_filter_proof(capsule)
    require(preparation["status_filter_grammar_proof"] == profile["status_filter_grammar_proof"] == grammar, "native parser grammar proof changed")
    required_sources = unique_records(preparation["sources"])
    source_by_path = {node["path"]: node for node in required_sources}
    essential_sources = [
        record(Path(__file__)),
        record(OUTPUT / "test_prepare_native.py"),
        record(ROOT / "AstraResearch/astra/io.py"),
        record(PLAN),
        record(NATIVE_PLAN),
        *capsule["sources"],
    ]
    require(all(source_by_path.get(n["path"], {}).get("sha256") == n["sha256"] for n in essential_sources), "essential running helper/plan/compiled source closure omitted")
    required_inputs = unique_records(
        [
            *preparation["inputs"],
            *v1_links,
            *failure_links,
            *registration_links,
            preparation["script"],
            preparation["profile"],
            preparation["native_plan"],
            preparation["config_method_plan"],
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
        require_closure(frozen, key, nodes)
        for node in frozen[key]:
            absent = node["path"] in profile["required_absent_inputs"]
            path = check_record(node, allow_missing=absent)
            require(not absent or path is None, "previously absent fixed input appeared; new version required")
    parent_preparation, _, _, parent_links = parent_context()
    parent_raw = [n for n in parent_preparation["inputs"] if n["path"].endswith(".bin.zst") and n.get("day") in DAYS]
    parent_basic = [n for n in parent_preparation["inputs"] if Path(n["path"]).parent == Path("/mnt/data0/contract/tse/stock") and Path(n["path"]).stem in DAYS]
    require(len(parent_raw) == 32 and len(parent_basic) == 2, "fixed32 BIN and two BasicInfo provenance required")
    parent_keys = [c[key] for c in profile["parents"] for key in ("parent_origins", "expected_native_keys")]
    require_closure(frozen, "inputs", [*parent_links, *parent_raw, *parent_basic, *parent_keys, record(CALENDAR)])
    _, runtime = runtime_preflight(parent_preparation)
    require([j["day"] for j in preparation["jobs"]] == list(DAYS), "exact two fresh daily jobs required")
    original = read_yaml(Path(control["old_config"]["path"]))
    jobs = {j["day"]: j for j in preparation["jobs"]}
    require(job_key is None or job_key in jobs, "unknown frozen job key")
    selected = jobs.values() if job_key is None else [jobs[job_key]]
    for native_job in selected:
        require(native_job["day"] in DAYS and native_job["symbols"] == [s for s in TARGETS if s in native_job["symbols"]], "fixed daily target order changed")
        require(Path(native_job["work"]) == OUTPUT / "pilot" / native_job["day"], "fresh V2 daily work directory required")
        expected = job(
            native_job["day"],
            Path(native_job["work"]),
            Path(preparation["binary"]["path"]),
            Path(native_job["config"]["path"]),
            native_job["symbols"],
            "h15_fixed_two_leg_day",
            runtime,
        )
        require(native_job == expected, "daily command/output/environment contract changed")
        validate_daily_config(read_yaml(Path(native_job["config"]["path"])), original, native_job["symbols"])
        require(type(native_job["requires_replay"]) is bool and native_job["requires_replay"] == bool(native_job["symbols"]), "native missing-day replay status changed")
        if not native_job["requires_replay"]:
            require(native_job["command"] == [] and native_job["outputs"] == [], "declared missing day cannot run fallback replay")
            continue
        require(native_job["command"][0] == preparation["binary"]["path"] and native_job["environment"]["LD_LIBRARY_PATH"] == str(runtime), "job escaped immutable binary/runtime")
        require(
            not any(Path(path).exists() for path in native_job["outputs"]) and not (Path(native_job["work"]) / "status").exists(), "fresh job outputs/status required before replay"
        )
    for node in capsule["live_sources_before_and_after"]:
        check_record(node)
    check_record(capsule["live_built_binary_before_and_after"])
    return {"preflight_passed": True, "method_identity": frozen["identity"], "job": job_key, "registration_control_reused": True, "native_replays_launched": 0, "model_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", type=Path, metavar="ROOT_FROZEN_METHOD_YAML")
    parser.add_argument("--job", choices=DAYS)
    args = parser.parse_args()
    if args.prepare and args.job:
        parser.error("--job requires --preflight")
    print(prepare() if args.prepare else preflight(args.preflight.resolve(), args.job))


if __name__ == "__main__":
    main()
