"""Draft H16 native pilot orchestration. Only root invokes the lifecycle APIs.

Preparation reads contract/config metadata and hashes identities. Producer
freeze precedes commands; evaluation freeze precedes native table decoding.
Python validates native outputs and joins keys; it never creates features.
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
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "AstraResearch"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astra.io import digest, file_hash, read_yaml, write_yaml

OUTPUT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "AstraResearch/experiments/information_state_20260930"
PROFILE = EXPERIMENT / "h16-native-profile.yaml"
PLAN = EXPERIMENT / "H16_NATIVE_PLAN.md"
HYPOTHESIS_PLAN = EXPERIMENT / "H16_PLAN.md"
PARENT = OUTPUT.parent / "native-h15-peer-trade-v2"
MAPPING = OUTPUT.parent / "native-h15-mapping-support"
CALENDAR = ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"
PARENT_ID = "00540736c9defcabf2eb9e577bee50840dd3eedd62e00b3fdd981572118a8b61"
PARENT_SUPPORT_ID = "0430f280dc5f18d71050507e878952c6e46cb11360006d59a5b0d8882d8614a5"
OLD_BINARY_SHA = "613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b"
DAYS = ("20260119", "20260120")
CARRIERS = ("2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2615", "3481", "2409", "2344", "2337", "2481", "3037", "3711")
BATCH_SYMBOLS = ("3481", "2344", "2337")
MODULE = "AuctionCycleInformation"
ACCEPT_ALL = "TRIAL || !TRIAL"
NUMERIC = (
    "proposal_displacement_5ticks",
    "residual_mid_clearing_basis_5ticks",
    "previous_acceptance_error_5ticks",
    "proposal_match_mass_share",
    "received_cycle_age",
)
CATEGORICAL = ("observation_regime", "cycle_state")
ALPHA = (*NUMERIC, *CATEGORICAL)
INFO = (
    "actual_anchor_exchange_time",
    "actual_anchor_first_receive_time",
    "actual_mass_available_time",
    "previous_anchor_first_receive_time",
    "observed_match_interval_us",
    "trial_exchange_time",
    "trial_available_time",
    "residual_book_exchange_time",
    "residual_book_available_time",
    "previous_error_available_time",
    "cumulative_actual_anchor_count",
    "session_actual_anchor_count",
    "cumulative_unique_actual_print_count",
    "cumulative_actual_repeat_count",
    "cumulative_trial_count",
    "cumulative_hard_censor_count",
    "actual_clearing_price",
    "trial_clearing_price",
    "observed_actual_quantity",
    "trial_matched_quantity",
    "cycle_support",
    "boundary_available_time",
)
KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")
RAW = ("OriginMidPrice", "OriginMidTicks", "OriginBidTicks", "OriginAskTicks")
ALPHA_COLUMNS = tuple(f"{MODULE}.0.{name}.0" for name in ALPHA)
NUMERIC_COLUMNS = tuple(f"{MODULE}.0.{name}.0" for name in NUMERIC)
CAT_COLUMNS = tuple(f"{MODULE}.0.{name}.0" for name in CATEGORICAL)
INFO_COLUMNS = tuple(f"H16_{name}" for name in INFO)
COLUMNS = (*KEYS, *ALPHA_COLUMNS, *RAW, *INFO_COLUMNS)
REGIMES = {"warmup", "continuous", "indicative", "auction", "unknown", "suspended"}
STATES = {"warmup", "inactive", "anchor_only", "trial_zero", "trial_up", "trial_down", "trial_flat", "trial_na", "stale", "unknown"}
PRICE_INFO = ("actual_clearing_price", "trial_clearing_price")
AVAILABLE_INFO = (
    "actual_anchor_first_receive_time",
    "actual_mass_available_time",
    "previous_anchor_first_receive_time",
    "trial_available_time",
    "residual_book_available_time",
    "previous_error_available_time",
    "boundary_available_time",
)
CONTRACT = {
    "periodic_seconds": 10,
    "expected_native_rows": 1381,
    "support_origin_seconds": 30,
    "expected_support_rows": 461,
    "indicative_max_age_micros": 10_000_000,
    "minimum_session_cycles": 5,
    "minimum_distinct_sampled_cycles": 5,
    "minimum_support_origins": 20,
    "clock_upper_exclusive": 2**53,
    "variation_is_fieldwise_descriptive": True,
    "repeat_credit": False,
    "labels_read": False,
    "model_fits": 0,
    "python_feature_calculation": False,
}
PRODUCER_SCHEMA = "h16-native-producer-method-v1"
EVALUATION_SCHEMA = "h16-native-source-support-method-v1"
ALLOWED_OLD_CHANGES = {"build/Release/CMakeCache.txt", "build/Release/compile_commands.json", "src/oms/modules/feature/test/CMakeLists.txt"}
NEW_SOURCES = (
    "src/oms/modules/feature/experimental/auction_cycle_information/auction_cycle_information.h",
    "src/oms/modules/feature/experimental/auction_cycle_information/auction_cycle_information.cpp",
    "src/oms/modules/feature/experimental/auction_cycle_information/CMakeLists.txt",
    "src/oms/modules/feature/test/auction_cycle_information_test.cpp",
)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path), "size": path.stat().st_size}


def check_record(node, *, allow_missing=False):
    path = Path(node["path"])
    require(path.is_absolute() and str(path.resolve()) == str(path) and re.fullmatch(r"[0-9a-f]{64}", node["sha256"]), "canonical file identity required")
    if allow_missing and not path.is_file():
        return None
    require(path.is_file() and file_hash(path) == node["sha256"], f"file identity drift: {path}")
    if "size" in node:
        require(type(node["size"]) is int and node["size"] == path.stat().st_size, f"file size drift: {path}")
    return path


def unique_records(nodes):
    index = {}
    for node in nodes:
        previous = index.get(node["path"])
        require(previous is None or previous["sha256"] == node["sha256"], "conflicting file identities")
        index[node["path"]] = node
    return [index[key] for key in sorted(index)]


def require_closure(method, scope, nodes):
    index = {node["path"]: node["sha256"] for node in method[scope]}
    require(len(index) == len(method[scope]), "duplicate closure paths")
    require(all(index.get(node["path"]) == node["sha256"] for node in nodes), f"mandatory {scope} closure omitted or changed")


def canonical_method(path, schema):
    path = Path(path).resolve()
    body = read_yaml(path)
    anchor = path.with_suffix(path.suffix + ".identity").read_text().strip()
    require(body.get("schema") == schema and body.get("status") == "frozen", "wrong frozen method schema/status")
    require(body.get("identity") == anchor == digest({k: v for k, v in body.items() if k != "identity"}), "method canonical identity/anchor mismatch")
    return body


def freeze_payload(payload, destination):
    """Root-only convenience: call after draft code/tests are suspended."""
    destination = Path(destination).resolve()
    require(not destination.exists() and not destination.with_suffix(destination.suffix + ".identity").exists(), "fresh frozen method path required")
    body = {**payload, "status": "frozen"}
    body["identity"] = digest(body)
    write_yaml(destination, body)
    destination.with_suffix(destination.suffix + ".identity").write_text(body["identity"] + "\n")
    return body["identity"]


def assert_absent(paths):
    require(len(paths) == len(set(paths)), "duplicate RequiredAbsent entries")
    require(all(Path(path).is_absolute() and not Path(path).exists() for path in paths), "RequiredAbsent input appeared; new version required")


def validate_profile(profile):
    require(
        profile.get("schema") == "h16-native-pilot-profile-v1"
        and profile.get("status") == "prospective_draft"
        and profile.get("module") == MODULE
        and profile.get("fixed_train_days") == list(DAYS)
        and profile.get("carrier_order") == list(CARRIERS)
        and profile.get("batch_support_symbols") == list(BATCH_SYMBOLS)
        and profile.get("alpha_numeric") == list(NUMERIC)
        and profile.get("alpha_categorical") == list(CATEGORICAL)
        and profile.get("info_fields") == list(INFO)
        and profile.get("contract") == CONTRACT
        and profile.get("status_filter") == ACCEPT_ALL
        and profile.get("raw_subscription") is True
        and profile.get("labelers") == [],
        "prospective H16 representation/support/cohort contract drift",
    )


def science_sources():
    return [
        record(path)
        for path in (Path(__file__), OUTPUT / "test_native_pilot.py", OUTPUT / "run_native_pipeline.py", PROFILE, PLAN, HYPOTHESIS_PLAN, ROOT / "AstraResearch/astra/io.py")
    ]


def parent_metadata():
    producer_path = PARENT / "frozen-method.yaml"
    support_path = PARENT / "frozen-support-evaluation-method.yaml"
    producer = canonical_method(producer_path, "h15-native-producer-method-v2")
    support = canonical_method(support_path, "h15-native-support-evaluation-method-v2")
    require(producer["identity"] == PARENT_ID and support["identity"] == PARENT_SUPPORT_ID, "wrong immutable H15 pair controls")
    preparation = read_yaml(check_record(producer["preparation"]))
    require(digest(preparation) == producer["preparation_identity"], "old native preparation payload drift")
    require(preparation["binary"]["sha256"] == OLD_BINARY_SHA, "registration baseline binary changed")
    capsule = read_yaml(check_record(preparation["producer"]))
    mapping_prep_path = MAPPING / "preparation-receipt.yaml"
    mapping_prep = read_yaml(mapping_prep_path)
    mapping_profile = read_yaml(check_record(mapping_prep["profile"]))
    require(digest(mapping_profile) == mapping_prep["profile_identity"], "mapping metadata profile drift")
    parents = copy.deepcopy(mapping_profile["parents"])
    require([(p["day"], p["symbol"]) for p in parents] == [(day, symbol) for day in DAYS for symbol in CARRIERS], "original32 cell order changed")
    original_manifests = [node for node in mapping_prep["inputs"] if node["path"].endswith("/artifact.yaml")]
    require(len(original_manifests) == 2, "two small original daily manifests required; no aggregate Store.resolve")
    for node in original_manifests:
        body = read_yaml(check_record(node))
        require(digest(body) == node["sha256"] and body["kind"] == "native_export_day", "original daily native artifact identity changed")
        day = str(body["metadata"]["days"][0])
        require(body["metadata"]["symbols"] == list(CARRIERS), "original carrier order changed")
        files = {entry["path"]: entry for entry in body["files"]}
        for cell in [p for p in parents if p["day"] == day]:
            relative = f"{day}/{cell['symbol']}/data/{day}/{cell['symbol']}/values.parquet"
            require(cell["parent_origins"] == {**files[relative], "path": str(Path(node["path"]).parent / relative)}, "parent row/mid file escaped original daily artifact")
    raw = {(n["day"], n["symbol"]): n for n in preparation["inputs"] if n["path"].endswith(".bin.zst") and n.get("day") in DAYS}
    basics = {Path(n["path"]).stem: n for n in preparation["inputs"] if Path(n["path"]).parent == Path("/mnt/data0/contract/tse/stock") and Path(n["path"]).stem in DAYS}
    require(set(raw) == {(day, symbol) for day in DAYS for symbol in CARRIERS} and set(basics) == set(DAYS), "fixed32 native BIN/two BasicInfo identities required")
    links = [record(path) for path in (producer_path, producer_path.with_suffix(".yaml.identity"), support_path, support_path.with_suffix(".yaml.identity"), mapping_prep_path)]
    links += [producer["preparation"], preparation["producer"], preparation["profile"], mapping_prep["profile"], *original_manifests]
    return preparation, capsule, parents, raw, basics, producer, support, links


def select_day(day, raw, basic, parents):
    missing = []
    for symbol in CARRIERS:
        if check_record(raw[(day, symbol)], allow_missing=True) is None:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_BIN_skip_day_no_replacement"})
    if check_record(basic, allow_missing=True) is None:
        return [], [*missing, {"day": day, "reason": "missing_BasicInfo_skip_day_no_replacement"}]
    with Path(basic["path"]).open(newline="") as stream:
        contracts = [row["symbol"] for row in csv.DictReader(stream)]
    require(len(contracts) == len(set(contracts)), "duplicate BasicInfo symbol")
    for symbol in CARRIERS:
        if symbol not in contracts:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_contract_skip_day"})
    if missing:
        return [], missing
    present = []
    for cell in [p for p in parents if p["day"] == day]:
        if check_record(cell["parent_origins"], allow_missing=True) is None:
            missing.append({"day": day, "symbol": cell["symbol"], "reason": "missing_original_keys_skip_writer"})
        else:
            present.append(cell["symbol"])
    return present, missing


def daily_config(original, present):
    groups = []
    require(present == [symbol for symbol in CARRIERS if symbol in present], "present writer order or uniqueness changed")
    for group in original["Modules"]:
        if not group["Gid"]:
            groups.append(copy.deepcopy(group))
            continue
        symbol = group["Gid"]
        current = next(decl for decl in group["Decl"] if decl["Desc"] == "CurrentBook.0")
        declarations = [copy.deepcopy(current)]
        if symbol in present:
            declarations.append(
                {"Desc": f"{MODULE}.0", "Spec": {"Symbol": symbol, "IndicativeMaxAge": "10s", "StatusFilter": ACCEPT_ALL, "Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}}
            )
            metadata = [
                {"Feature": f"CurrentBook.0.{family}.0@{symbol}", "Name": name}
                for family, name in zip(("book_mid_price", "book_mid_ticks", "book_bid_ticks", "book_ask_ticks"), RAW, strict=True)
            ]
            metadata += [{"Feature": f"{MODULE}.0.{name}.0@{symbol}", "Name": f"H16_{name}"} for name in INFO]
            declarations.append(
                {
                    "Desc": "DatasetWriter.0",
                    "Spec": {
                        "Subscribe": [{"Book": [symbol]}],
                        "Exports": [f"{MODULE}.0.{name}.0@{symbol}" for name in ALPHA],
                        "MetadataExports": metadata,
                        "PeriodicSampler": {"StartTime": "091000", "UntilTime": "130000", "SampleInterval": "10s"},
                        "OutputPath": "${cwd}/data/${trading_date}/" + symbol + "/values.parquet",
                        "Format": "parquet",
                        "UseTmp": True,
                        "EmitSampleContext": True,
                        "RequireAlphaFactorExports": True,
                    },
                }
            )
        groups.append({"Gid": symbol, "Decl": declarations})
    require([g["Gid"] for g in groups if g["Gid"]] == list(CARRIERS), "fixed16 global MD carrier order changed")
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def validate_daily_config(config, original, present):
    require(config == daily_config(original, present), "raw subscriptions, timing, labels or native exports escaped H16 contract")


def runtime_check(preparation):
    runtime = read_yaml(check_record(preparation["runtime_snapshot"]))
    require(runtime["dependencies"] == preparation["runtime_dependencies"] and len(runtime["dependencies"]) == 51, "shared runtime51 changed")
    for node in runtime["dependencies"]:
        check_record(node)
        # Historical runtime receipts retain loader aliases such as /lib/*.so.
        # Verify the actual current target's bytes without rewriting that receipt.
        system_target = Path(node["original_system_path"]).resolve()
        check_record({"path": str(system_target), "sha256": node["sha256"]})


def validate_ldd(text, dependencies):
    names = set(re.findall(r"^\s*(\S+)\s+=>", text, re.MULTILINE))
    require(
        "not found" not in text and names == {node["name"] for node in dependencies if not node["name"].startswith("ld-linux")},
        "new binary dynamic dependency names differ from frozen runtime51",
    )
    resolved = dict(re.findall(r"^\s*(\S+)\s+=>\s+(\S+)", text, re.MULTILINE))
    for node in dependencies:
        if node["name"] in resolved:
            require(Path(resolved[node["name"]]).resolve() == Path(node["path"]).resolve(), "new native binary resolves outside immutable runtime snapshot")


def capture_capsule(prior, binary_path):
    require(not (OUTPUT / "compiled-source").exists() and not (OUTPUT / "coco").exists(), "fresh H16 source/binary capsule required")
    live, changes = {}, []
    old = {str(Path(node["path"]).relative_to(ROOT)): node for node in prior["live_sources_before_and_after"]}
    require(len(old) == 1648, "immutable H15 compiled source scope changed")
    for relative, node in old.items():
        current = record(ROOT / relative)
        if current["sha256"] != node["sha256"]:
            require(relative in ALLOWED_OLD_CHANGES, f"unaccounted old native math/source change: {relative}")
            if relative.endswith("feature/test/CMakeLists.txt"):
                archived = next(n for n in prior["sources"] if n["path"].endswith("/" + relative))
                require((ROOT / relative).read_bytes().startswith(Path(archived["path"]).read_bytes()), "test CMake may only append H16 target")
            changes.append({"relative_path": relative, "old_sha256": node["sha256"], "new_sha256": current["sha256"]})
        live[relative] = current
    for relative in NEW_SOURCES:
        require(relative not in live, "H16 material source existed in old H15 scope")
        live[relative] = record(ROOT / relative)
    compile_commands = json.loads((ROOT / "build/Release/compile_commands.json").read_text())
    require(any(Path(command["file"]).resolve() == ROOT / NEW_SOURCES[1] for command in compile_commands), "H16 TU absent from Release build commands")
    snapshots = []
    for relative, node in sorted(live.items()):
        target = OUTPUT / "compiled-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(node["path"], target)
        target.chmod(0o444)
        snapshot = record(target)
        require(snapshot["sha256"] == node["sha256"] and record(node["path"]) == node, "source changed during capture")
        snapshots.append(snapshot)
    binary_live = record(binary_path)
    shutil.copy2(binary_path, OUTPUT / "coco")
    (OUTPUT / "coco").chmod(0o555)
    binary = record(OUTPUT / "coco")
    require(record(binary_path) == binary_live and binary["sha256"] == binary_live["sha256"], "binary changed during capture")
    return {
        "schema": "h16-native-source-capsule-v1",
        "prior_capsule": record(Path(prior["shared_v1_capsule"]["path"])),
        "sources": snapshots,
        "live_sources_before_and_after": list(live.values()),
        "binary": binary,
        "live_built_binary_before_and_after": binary_live,
        "explicit_non_math_source_changes": changes,
        "new_material_sources": list(NEW_SOURCES),
        "existing_native_math_all_sha256_equal": True,
        "source_before_after_all_sha256_equal": True,
    }


def registration(guide):
    modules = [node for node in guide["module_types"] if node["type"] == MODULE]
    require(len(modules) == 1, "one H16 native registration required")
    families = modules[0]["feature_families"]
    require(len(families) == len(ALPHA) + len(INFO), "H16 native family count changed")
    require({n["name"] for n in families if "alpha_factor" in n["attributes"]} == set(ALPHA), "native Alpha schema changed")
    require({n["name"] for n in families if "categorical" in n["attributes"]} == set(CATEGORICAL), "native categorical schema changed")
    require({n["name"] for n in families if "alpha_factor" not in n["attributes"]} == set(INFO), "Info entered feature representation")
    return modules[0]


def command_job(day, work, config, symbols, binary, environment, *, key=None, outputs=None):
    return {
        "day": day,
        "key": key or day,
        "work": str(work),
        "symbols": symbols,
        "config": record(config),
        "outputs": outputs or [str(work / "data" / day / symbol / "values.parquet") for symbol in symbols],
        "command": [binary["path"], "-d", day, "-C", str(work), "--trading-calendar", str(CALENDAR), "--run-status-dir", str(work / "status"), str(config)],
        "environment": environment,
        "requires_replay": True,
    }


def prepare_metadata(new_binary_path):
    """ROOT ONLY: after C++/test/build READY. No replay or native values read."""
    validate_profile(read_yaml(PROFILE))
    require(not (OUTPUT / "preparation-receipt.yaml").exists(), "fresh draft preparation required")
    parent, prior, parents, raw, basics, producer, support, parent_links = parent_metadata()
    runtime_check(parent)
    capsule = capture_capsule(prior, Path(new_binary_path).resolve())
    capsule_path = OUTPUT / "producer-source-build.yaml"
    write_yaml(capsule_path, capsule)
    env = {"LD_LIBRARY_PATH": parent["runtime_directory"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    guide_path = OUTPUT / "feature-guide.yaml"
    with guide_path.open("w") as stream:
        subprocess.run([capsule["binary"]["path"], "feature-guide"], stdout=stream, stderr=subprocess.PIPE, env={**os.environ, **env}, check=True)
    native_registration = registration(read_yaml(guide_path))
    dynamic = subprocess.run(["ldd", capsule["binary"]["path"]], capture_output=True, text=True, env={**os.environ, **env}, check=True).stdout
    validate_ldd(dynamic, parent["runtime_dependencies"])
    ldd_path = OUTPUT / "ldd.txt"
    ldd_path.write_text(dynamic)
    original_job = next(job for job in parent["jobs"] if job["day"] == DAYS[0])
    original_config = read_yaml(check_record(original_job["config"]))
    jobs, missing = [], []
    for day in DAYS:
        present, skips = select_day(day, raw, basics[day], parents)
        missing += skips
        if not present:
            continue
        work = OUTPUT / "pilot" / day
        work.mkdir(parents=True)
        config = work / "config.yaml"
        write_yaml(config, daily_config(original_config, present))
        jobs.append(command_job(day, work, config, present, capsule["binary"], env))
    required_absent = [node["path"] for node in [*raw.values(), *basics.values(), *[p["parent_origins"] for p in parents]] if not Path(node["path"]).is_file()]
    control = None
    if any(job["day"] == DAYS[0] for job in jobs) and all(Path(path).is_file() for path in original_job["outputs"]):
        work = OUTPUT / "registration-control" / DAYS[0]
        work.mkdir(parents=True)
        config = work / "config.yaml"
        shutil.copy2(original_job["config"]["path"], config)
        outputs = [str(work / Path(path).relative_to(original_job["work"])) for path in original_job["outputs"]]
        control = command_job(DAYS[0], work, config, original_job["symbols"], capsule["binary"], env, key="registration_control", outputs=outputs)
        control["required_byte_equal_outputs"] = [record(path) for path in original_job["outputs"]]
        control["old_config"] = original_job["config"]
        require(control["config"]["sha256"] == control["old_config"]["sha256"] and len(outputs) == 4, "registration control must retain exact old pair config/four outputs")
    else:
        missing.append({"day": DAYS[0], "reason": "missing_required_registration_day_or_original_outputs_no_substitution"})
        required_absent += [path for path in original_job["outputs"] if not Path(path).is_file()]
    inputs = unique_records(
        [
            *parent_links,
            *producer["inputs"],
            *support["inputs"],
            *raw.values(),
            *basics.values(),
            *[p["parent_origins"] for p in parents],
            *parent["runtime_dependencies"],
            parent["runtime_snapshot"],
            record(CALENDAR),
            capsule["binary"],
            record(capsule_path),
            record(guide_path),
            record(ldd_path),
            *[j["config"] for j in jobs],
            *([control["config"], control["old_config"], *control["required_byte_equal_outputs"]] if control else []),
        ]
    )
    sources = unique_records([*science_sources(), *capsule["sources"], *producer["sources"], *support["sources"]])
    body = {
        "schema": "h16-native-preparation-v1",
        "profile": record(PROFILE),
        "profile_identity": digest(read_yaml(PROFILE)),
        "producer": record(capsule_path),
        "guide": record(guide_path),
        "ldd": record(ldd_path),
        "native_module_registration": native_registration,
        "binary": capsule["binary"],
        "runtime_snapshot": parent["runtime_snapshot"],
        "runtime_dependencies": parent["runtime_dependencies"],
        "runtime_directory": parent["runtime_directory"],
        "parents": parents,
        "raw_files": list(raw.values()),
        "basic_info": basics,
        "template": original_job["config"],
        "jobs": jobs,
        "registration_control": control,
        "execution_order": (["registration_control"] + [j["day"] for j in jobs]) if control else [],
        "required_absent_inputs": sorted(set(required_absent)),
        "missing": missing,
        "sources": sources,
        "inputs": inputs,
        "labels_read": False,
        "raw_streams_decoded": 0,
        "native_replays_launched": 0,
        "model_fits": 0,
        "python_features_generated": False,
    }
    path = OUTPUT / "preparation-receipt.yaml"
    write_yaml(path, body)
    for node in [*capsule["live_sources_before_and_after"], capsule["live_built_binary_before_and_after"]]:
        check_record(node)
    return {"prepared_only": True, "daily_jobs": len(jobs), "registration_control": control is not None, "preparation": str(path), "producer_payload": producer_payload(path)}


def producer_payload(preparation_path):
    preparation_path = Path(preparation_path).resolve()
    preparation = read_yaml(preparation_path)
    return {
        "schema": PRODUCER_SCHEMA,
        "status": "draft",
        "preparation": record(preparation_path),
        "preparation_identity": digest(preparation),
        "profile_identity": preparation["profile_identity"],
        "sources": preparation["sources"],
        "inputs": unique_records([*preparation["inputs"], record(preparation_path)]),
    }


def execution(job, identity):
    path = Path(job["work"]) / "execution-receipt.yaml"
    receipt = read_yaml(path)
    require(
        receipt.get("method_identity") == identity
        and receipt.get("job") == job["key"]
        and type(receipt.get("exit_code")) is int
        and receipt["exit_code"] == 0
        and receipt.get("command") == job["command"],
        "execution receipt ownership/command/exit mismatch",
    )
    require([n["path"] for n in receipt.get("outputs", [])] == job["outputs"], "configured native output missing/reordered")
    nodes = [record(path), receipt["log"], *receipt["outputs"]]
    nodes += [record(path) for path in sorted((Path(job["work"]) / "status").rglob("*")) if path.is_file()]
    for node in nodes:
        check_record(node)
    return receipt, nodes


def control_proof(preparation, identity):
    control = preparation["registration_control"]
    require(control is not None, "fixed registration control unavailable")
    receipt, links = execution(control, identity)
    for observed, expected in zip(receipt["outputs"], control["required_byte_equal_outputs"], strict=True):
        check_record(expected)
        require(observed["sha256"] == expected["sha256"], "old H15 four Parquet entire bytes changed under new registration")
    return links


def preflight(method_path, job=None):
    method = canonical_method(method_path, PRODUCER_SCHEMA)
    preparation_path = Path(method["preparation"]["path"])
    preparation = read_yaml(preparation_path)
    assert_absent(preparation["required_absent_inputs"])  # highest priority, before any large input hashes
    check_record(method["preparation"])
    require(digest(preparation) == method["preparation_identity"], "preparation payload drift")
    profile = read_yaml(check_record(preparation["profile"]))
    validate_profile(profile)
    require(digest(profile) == method["profile_identity"] == preparation["profile_identity"], "profile payload drift")
    parent, prior, parents, raw, basics, parent_method, parent_support, parent_links = parent_metadata()
    require(
        preparation["parents"] == parents and preparation["raw_files"] == list(raw.values()) and preparation["basic_info"] == basics,
        "fixed original parent/raw/BasicInfo identities changed",
    )
    original_job = next(item for item in parent["jobs"] if item["day"] == DAYS[0])
    require(preparation["template"] == original_job["config"], "original H15 registration recipe changed")
    optional_paths = [node["path"] for node in [*raw.values(), *basics.values(), *[cell["parent_origins"] for cell in parents]]] + original_job["outputs"]
    require(
        preparation["required_absent_inputs"] == sorted({path for path in optional_paths if not Path(path).is_file()}),
        "RequiredAbsent escaped declared fixed raw/metadata/cohort inputs",
    )
    require(
        preparation["runtime_snapshot"] == parent["runtime_snapshot"]
        and preparation["runtime_dependencies"] == parent["runtime_dependencies"]
        and preparation["runtime_directory"] == parent["runtime_directory"],
        "shared runtime ownership changed",
    )
    expected = producer_payload(preparation_path)
    require_closure(method, "sources", unique_records([*expected["sources"], *science_sources()]))
    require_closure(method, "inputs", expected["inputs"])
    require_closure(method, "sources", unique_records([*parent_method["sources"], *parent_support["sources"]]))
    require_closure(
        method,
        "inputs",
        unique_records(
            [
                *parent_links,
                *parent_method["inputs"],
                *parent_support["inputs"],
                preparation["profile"],
                preparation["producer"],
                preparation["guide"],
                preparation["ldd"],
                preparation["binary"],
                preparation["runtime_snapshot"],
                *preparation["runtime_dependencies"],
                record(CALENDAR),
                *raw.values(),
                *basics.values(),
                *[p["parent_origins"] for p in parents],
                *[item["config"] for item in preparation["jobs"]],
            ]
        ),
    )
    for scope in ("sources", "inputs"):
        for node in method[scope]:
            check_record(node, allow_missing=node["path"] in preparation["required_absent_inputs"])
    capsule = read_yaml(check_record(preparation["producer"]))
    require(
        capsule["schema"] == "h16-native-source-capsule-v1"
        and capsule["existing_native_math_all_sha256_equal"] is True
        and capsule["source_before_after_all_sha256_equal"] is True,
        "compiled source lineage flags missing",
    )
    require(capsule["binary"] == preparation["binary"] and len(capsule["sources"]) == len(capsule["live_sources_before_and_after"]), "capsule source/binary ownership mismatch")
    source_map = {str(Path(node["path"]).relative_to(OUTPUT / "compiled-source")): node for node in capsule["sources"]}
    live_map = {str(Path(node["path"]).relative_to(ROOT)): node for node in capsule["live_sources_before_and_after"]}
    old_map = {str(Path(node["path"]).relative_to(ROOT)): node for node in prior["live_sources_before_and_after"]}
    require(set(source_map) == set(live_map) == set(old_map) | set(NEW_SOURCES), "compiled capsule omitted/added a source")
    require(all(source_map[key]["sha256"] == live_map[key]["sha256"] for key in source_map), "compiled/live source hash mismatch")
    require(all(live_map[key]["sha256"] == old_map[key]["sha256"] for key in old_map if key not in ALLOWED_OLD_CHANGES), "old native math changed")
    for node in [*capsule["live_sources_before_and_after"], capsule["live_built_binary_before_and_after"]]:
        check_record(node)
    require_closure(method, "sources", capsule["sources"])
    runtime_check(preparation)
    validate_ldd(check_record(preparation["ldd"]).read_text(), preparation["runtime_dependencies"])
    require(registration(read_yaml(check_record(preparation["guide"]))) == preparation["native_module_registration"], "registered family metadata drift")
    raw = {(node["day"], node["symbol"]): node for node in preparation["raw_files"]}
    original = read_yaml(check_record(preparation["template"]))
    expected_jobs = []
    expected_missing = []
    for day in DAYS:
        present, missing = select_day(day, raw, preparation["basic_info"][day], preparation["parents"])
        expected_missing += missing
        if present:
            matches = [item for item in preparation["jobs"] if item["day"] == day]
            require(len(matches) == 1 and matches[0]["symbols"] == present, "daily writer/missing cohort changed")
            native_job = matches[0]
            validate_daily_config(read_yaml(check_record(native_job["config"])), original, present)
            expected_jobs.append(day)
    require([item["day"] for item in preparation["jobs"]] == expected_jobs, "unexpected replacement day/job")
    require(all(missing in preparation["missing"] for missing in expected_missing), "fixed missing input not reported")
    control = preparation["registration_control"]
    require(
        control is not None and read_yaml(check_record(control["config"])) == original and len(control["outputs"]) == 4, "unchanged H15 control mandatory before any H16 replay"
    )
    require(
        control["old_config"] == original_job["config"] and [node["path"] for node in control["required_byte_equal_outputs"]] == original_job["outputs"],
        "registration bytes escaped immutable old pair outputs",
    )
    require_closure(method, "inputs", [control["config"], control["old_config"], *control["required_byte_equal_outputs"]])
    require(preparation["execution_order"] == ["registration_control", *expected_jobs], "native execution order changed")
    jobs = {"registration_control": control, **{item["day"]: item for item in preparation["jobs"]}}
    environment = {"LD_LIBRARY_PATH": preparation["runtime_directory"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for key, native_job in jobs.items():
        work = OUTPUT / ("registration-control" if key == "registration_control" else "pilot") / native_job["day"]
        require(
            native_job["key"] == key and native_job["work"] == str(work) and native_job["environment"] == environment and native_job["requires_replay"] is True,
            "job directory/ownership/runtime changed",
        )
        expected_command = command_job(
            native_job["day"], work, work / "config.yaml", native_job["symbols"], preparation["binary"], environment, key=key, outputs=native_job["outputs"]
        )
        require(native_job["command"] == expected_command["command"] and native_job["config"] == expected_command["config"], "native command escaped frozen config/binary")
        expected_outputs = (
            [str(work / Path(path).relative_to(original_job["work"])) for path in original_job["outputs"]]
            if key == "registration_control"
            else [str(work / "data" / native_job["day"] / symbol / "values.parquet") for symbol in native_job["symbols"]]
        )
        require(native_job["outputs"] == expected_outputs, "configured output path changed")
    if job is not None:
        require(job in jobs, "job absent; no fallback")
        if job != "registration_control":
            control_proof(preparation, method["identity"])
        for path in jobs[job]["outputs"]:
            require(not Path(path).exists(), "fresh native destination required; no overwrite/resume")
        require(not (Path(jobs[job]["work"]) / "execution-receipt.yaml").exists(), "completed native job cannot be relaunched")
    return {"preflight_passed": True, "method_identity": method["identity"], "job": job, "native_replays_launched": 0, "model_fits": 0}


def bind_evaluation(producer_path):
    """ROOT ONLY after all commands: hashes outputs; reads no native table values."""
    producer_path = Path(producer_path).resolve()
    preflight(producer_path)
    method = canonical_method(producer_path, PRODUCER_SCHEMA)
    preparation = read_yaml(Path(method["preparation"]["path"]))
    links = control_proof(preparation, method["identity"])
    cells = []
    parents = {(p["day"], p["symbol"]): p for p in preparation["parents"]}
    for job in preparation["jobs"]:
        receipt, execution_links = execution(job, method["identity"])
        links += execution_links
        for symbol, node in zip(job["symbols"], receipt["outputs"], strict=True):
            cells.append({"day": job["day"], "symbol": symbol, "native": node, "parent_origins": parents[(job["day"], symbol)]["parent_origins"]})
    profile = {
        "schema": "h16-native-source-support-profile-v1",
        "producer_identity": method["identity"],
        "contract": CONTRACT,
        "cells": cells,
        "missing": preparation["missing"],
        "labels_read": False,
        "model_fits": 0,
        "python_features_generated": False,
    }
    path = OUTPUT / "support-evaluation-profile.yaml"
    require(not path.exists(), "fresh prospective evaluator profile required")
    write_yaml(path, profile)
    return {
        "schema": EVALUATION_SCHEMA,
        "status": "draft",
        "producer_method": record(producer_path),
        "producer_identity": method["identity"],
        "profile": record(path),
        "profile_identity": digest(profile),
        "sources": unique_records([*method["sources"], *science_sources()]),
        "inputs": unique_records([*method["inputs"], *links, record(producer_path), record(producer_path.with_suffix(producer_path.suffix + ".identity")), record(path)]),
        "labels_read": False,
        "model_fits": 0,
    }


def verify_evaluation(method_path):
    method = canonical_method(method_path, EVALUATION_SCHEMA)
    check_record(method["producer_method"])
    producer_path = Path(method["producer_method"]["path"])
    preflight(producer_path)
    producer = canonical_method(producer_path, PRODUCER_SCHEMA)
    require(method["producer_identity"] == producer["identity"], "wrong evaluation producer identity")
    preparation = read_yaml(Path(producer["preparation"]["path"]))
    profile = read_yaml(check_record(method["profile"]))
    require(
        profile["schema"] == "h16-native-source-support-profile-v1"
        and profile["contract"] == CONTRACT
        and digest(profile) == method["profile_identity"]
        and profile["producer_identity"] == producer["identity"],
        "evaluation profile drift",
    )
    links = control_proof(preparation, producer["identity"])
    cells = []
    parents = {(p["day"], p["symbol"]): p for p in preparation["parents"]}
    for job in preparation["jobs"]:
        receipt, nodes = execution(job, producer["identity"])
        links += nodes
        cells += [
            {"day": job["day"], "symbol": symbol, "native": node, "parent_origins": parents[(job["day"], symbol)]["parent_origins"]}
            for symbol, node in zip(job["symbols"], receipt["outputs"], strict=True)
        ]
    require(profile["cells"] == cells and profile["missing"] == preparation["missing"], "evaluation source cells escaped frozen commands/cohort")
    require_closure(method, "sources", unique_records([*producer["sources"], *science_sources()]))
    require_closure(
        method,
        "inputs",
        unique_records([*producer["inputs"], *links, method["producer_method"], record(producer_path.with_suffix(producer_path.suffix + ".identity")), method["profile"]]),
    )
    for scope in ("sources", "inputs"):
        for node in method[scope]:
            check_record(node, allow_missing=node["path"] in preparation["required_absent_inputs"])
    return method, profile


def session_bounds(day):
    elapsed = (date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}") - date(1970, 1, 1)).days
    midnight = (elapsed * 86400 - 8 * 3600) * 1_000_000
    return midnight + (9 * 3600 + 10 * 60) * 1_000_000, midnight + 13 * 3600 * 1_000_000


def same_bits(left, right):
    if left.type != right.type or left.null_count or right.null_count or len(left) != len(right):
        return False
    if left.type == pa.float64():
        return np.array_equal(left.to_numpy().view(np.uint64), right.to_numpy().view(np.uint64))
    return left.equals(right)


def validate_table(table):
    require(len(table.column_names) == len(COLUMNS) and set(table.column_names) == set(COLUMNS), "exact native7 Alpha/22 Info/keys/raw schema required; no labels")
    for name in COLUMNS:
        dtype = pa.int64() if name in KEYS else pa.string() if name in CAT_COLUMNS else pa.float64()
        role = b"time" if name == KEYS[0] else b"context" if name in KEYS else b"feature" if name in ALPHA_COLUMNS else b"metadata"
        require(table[name].type == dtype and table[name].null_count == 0, f"native dtype/null changed: {name}")
        require((table.schema.field(name).metadata or {}).get(b"coco.role") == role, f"native semantic role changed: {name}")
    sample, book, sequence = (table[name].to_numpy() for name in KEYS)
    require(
        np.all((sample > 0) & (sample < 2**53) & (book >= 0) & (book <= sample) & (sequence >= 0) & (sequence < 2**53)) and np.all(np.diff(sample) > 0),
        "unsafe/future/unordered native keys",
    )
    categories = {name: np.asarray(table[f"{MODULE}.0.{name}.0"].to_pylist()) for name in CATEGORICAL}
    require(set(categories["observation_regime"]).issubset(REGIMES) and set(categories["cycle_state"]).issubset(STATES), "unknown native category")
    info = {name: table[f"H16_{name}"].to_numpy() for name in INFO}
    for name, values in info.items():
        if name in PRICE_INFO:
            require(np.all(np.isnan(values) | (np.isfinite(values) & (values > 0))), "unavailable native price must remain NaN")
        elif name == "trial_matched_quantity":
            known = np.isfinite(values)
            require(
                np.all(np.isnan(values) | (known & (values >= 0) & (values < 2**53) & (values == np.floor(values)))),
                "known trial quantity must be an exact integer; absent operand remains NaN",
            )
            require(np.all(values[values == 0].view(np.uint64) == 0), "negative-zero native quantity identity")
        else:
            require(np.all(np.isfinite(values) & (values >= 0) & (values < 2**53) & (values == np.floor(values))), f"native exact integer/clock invalid: {name}")
            require(np.all(values[values == 0].view(np.uint64) == 0), f"negative-zero native identity: {name}")
            info[name] = values.astype(np.int64)
    for name in AVAILABLE_INFO:
        require(np.all(info[name] <= sample), f"future receive publication: {name}")
    for e, r in (
        ("actual_anchor_exchange_time", "actual_anchor_first_receive_time"),
        ("trial_exchange_time", "trial_available_time"),
        ("residual_book_exchange_time", "residual_book_available_time"),
    ):
        require(np.all((info[e] == 0) == (info[r] == 0)), "own clock zero-pairing invalid")
    first = info["actual_anchor_first_receive_time"]
    require(
        np.array_equal(np.isfinite(info["actual_clearing_price"]), first > 0) and np.array_equal(info["observed_actual_quantity"] > 0, first > 0),
        "actual anchor clock/positive price/mass availability disagrees",
    )
    prior = info["previous_anchor_first_receive_time"]
    interval = info["observed_match_interval_us"]
    known_interval = interval > 0
    require(np.all((~known_interval) | ((first > prior) & (prior > 0) & (first - prior == interval))), "past receive interval identity invalid")
    require(np.all((info["actual_mass_available_time"] == 0) | ((first > 0) & (info["actual_mass_available_time"] >= first))), "actual mass publication ownership invalid")
    for name in INFO:
        if name.startswith(("cumulative_", "session_")):
            require(np.all(np.diff(info[name]) >= 0), "cumulative/session native counter regressed")
    require(np.all(info["session_actual_anchor_count"] <= info["cumulative_actual_anchor_count"]), "session cycles exceed day cycles")
    require(np.all(info["cumulative_actual_anchor_count"] <= info["cumulative_unique_actual_print_count"]), "actual cycles exceed unique observed prints")
    support = info["cycle_support"]
    require(np.isin(support, (0, 1)).all(), "native support predicate must be exact0/1")
    supported = support == 1
    trial_r = info["trial_available_time"]
    known_trial_operand = np.isfinite(info["trial_matched_quantity"]) | np.isfinite(info["trial_clearing_price"])
    require(
        np.all((~supported) | (known_trial_operand & (first > 0) & (trial_r > first) & (sample - trial_r <= CONTRACT["indicative_max_age_micros"]))),
        "support violates observed fresh strict-post-anchor ownership",
    )
    numeric = {name: table[f"{MODULE}.0.{name}.0"].to_numpy() for name in NUMERIC}
    for name, values in numeric.items():
        require(not np.isposinf(values).any(), f"H16 cannot publish positive infinity: {name}")
        require(np.all(values[values == 0].view(np.uint64) == 0), f"semantic-zero numeric identity changed: {name}")
    inactive = categories["observation_regime"] == "continuous"
    require(np.all(categories["cycle_state"][inactive] == "inactive"), "continuous cycle state must be typed inactive")
    require(all(np.all(values[inactive].view(np.uint64) == 0) for values in numeric.values()), "continuous numeric facts must be exact known +0")
    hard = np.isin(categories["observation_regime"], ("unknown", "suspended"))
    require(
        np.all(categories["cycle_state"][hard] == "unknown") and all(np.isnan(values[hard]).all() for values in numeric.values()),
        "hard boundary cannot publish known current feature facts",
    )
    warmup = categories["observation_regime"] == "warmup"
    require(np.all(categories["cycle_state"][warmup] == "warmup") and all(np.isneginf(values[warmup]).all() for values in numeric.values()), "pre-observation warmup state changed")
    stale = categories["cycle_state"] == "stale"
    require(all(np.isnan(numeric[name][stale]).all() for name in (NUMERIC[0], NUMERIC[1], NUMERIC[3])), "stale proposal cannot remain a current known fact")
    for name in ("proposal_match_mass_share", "received_cycle_age"):
        values = numeric[name]
        finite = np.isfinite(values)
        require(np.all(values[finite] >= 0), f"native nonnegative feature boundary invalid: {name}")
        if name == "proposal_match_mass_share":
            require(np.all(values[finite] <= 1), "mass share outside probability simplex")
    mass = numeric["proposal_match_mass_share"]
    zero_mass = categories["cycle_state"] == "trial_zero"
    require(np.all(info["trial_matched_quantity"][zero_mass] == 0), "trial_zero lacks native known-zero quantity")
    require(np.all(mass[zero_mass].view(np.uint64) == 0), "known-zero trial mass must be +0")
    basis = numeric["residual_mid_clearing_basis_5ticks"]
    finite_active_basis = np.isfinite(basis) & ~inactive
    require(
        np.all((~finite_active_basis) | ((info["trial_exchange_time"] > 0) & (info["trial_exchange_time"] == info["residual_book_exchange_time"]))),
        "residual basis used unmatched exchange disclosure",
    )
    return {"sample": sample, "info": info, "numeric": numeric, "categories": categories}


def evaluate_cell(day, symbol, table, original):
    data = validate_table(table)
    start, end = session_bounds(day)
    require(len(table) == len(original) == CONTRACT["expected_native_rows"], "original1381 rows missing; no availability row filtering")
    for name in (*KEYS, "OriginMidPrice"):
        require(same_bits(table[name], original[name]), f"original key/raw mid bits changed: {name}")
    sample = data["sample"]
    require(sample[0] == start and sample[-1] == end and np.all(np.diff(sample) == 10_000_000), "original10s session cohort changed")
    mask = sample % 30_000_000 == 0
    require(int(mask.sum()) == CONTRACT["expected_support_rows"], "original461 support origins changed")
    info = data["info"]
    count = int(info["session_actual_anchor_count"][-1] - info["session_actual_anchor_count"][0])
    admitted = mask & (info["cycle_support"] == 1) & (info["actual_anchor_first_receive_time"] >= start) & (info["actual_anchor_first_receive_time"] < end)
    identities = set(zip(info["actual_anchor_exchange_time"][admitted].tolist(), info["actual_anchor_first_receive_time"][admitted].tolist(), strict=True))
    support_origins = int(admitted.sum())
    required = symbol in BATCH_SYMBOLS
    passed = (
        count >= CONTRACT["minimum_session_cycles"] and len(identities) >= CONTRACT["minimum_distinct_sampled_cycles"] and support_origins >= CONTRACT["minimum_support_origins"]
    )
    variation = {}
    descriptive_masks = {
        "all_original30s": mask,
        "observed_auction_or_indicative_original30s": mask & np.isin(data["categories"]["observation_regime"], ("auction", "indicative")),
        "counted_session_cycle_support_original30s": admitted,
    }
    for name, values in data["numeric"].items():
        variation[name] = {}
        for scope, selected in descriptive_masks.items():
            v = values[selected]
            finite = np.isfinite(v)
            variation[name][scope] = {
                "origins": len(v),
                "finite": int(finite.sum()),
                "known_positive_zero": int((v.view(np.uint64) == 0).sum()),
                "finite_nonzero": int((finite & (v != 0)).sum()),
                "finite_positive": int((finite & (v > 0)).sum()),
                "finite_negative": int((finite & (v < 0)).sum()),
                "warmup": int(np.isneginf(v).sum()),
                "unknown": int(np.isnan(v).sum()),
                "distinct_finite_bits": len(set(v[finite].view(np.uint64).tolist())),
            }
    for name in (*PRICE_INFO, "observed_actual_quantity", "trial_matched_quantity", "observed_match_interval_us"):
        values = info[name][mask]
        variation[name] = {"distinct_observed": len(np.unique(values[np.isfinite(values)]))}
    operand_support = {}
    predicates = {
        "actual_anchor_present": info["actual_anchor_first_receive_time"] > 0,
        "actual_price_present": np.isfinite(info["actual_clearing_price"]),
        "trial_price_present": np.isfinite(info["trial_clearing_price"]),
        "trial_quantity_present": np.isfinite(info["trial_matched_quantity"]),
        "trial_book_exchange_matches": (info["trial_exchange_time"] > 0) & (info["trial_exchange_time"] == info["residual_book_exchange_time"]),
        "past_match_interval_present": info["observed_match_interval_us"] > 0,
        "historical_acceptance_error_present": info["previous_error_available_time"] > 0,
        "native_cycle_support": info["cycle_support"] == 1,
    }
    for scope, selected in descriptive_masks.items():
        operand_support[scope] = {name: int((predicate & selected).sum()) for name, predicate in predicates.items()}
    return {
        "day": day,
        "symbol": symbol,
        "rows": len(table),
        "support_origins_total": int(mask.sum()),
        "session_distinct_actual_cycles": count,
        "distinct_sampled_cycles": len(identities),
        "fresh_post_anchor_support_origins": support_origins,
        "support_gate_required": required,
        "support_gate_passed": passed if required else None,
        "fieldwise_variability": variation,
        "native_operand_support": operand_support,
        "category_counts": {name: dict(Counter(values[mask].tolist())) for name, values in data["categories"].items()},
    }


def check(evaluation_path, destination=None):
    """ROOT ONLY after evaluation freeze. Narrow native decode; no labels/models."""
    method, profile = verify_evaluation(evaluation_path)
    cells = []
    for cell in profile["cells"]:
        table = pq.read_table(cell["native"]["path"], columns=list(COLUMNS), use_threads=False)
        original = pq.read_table(cell["parent_origins"]["path"], columns=[*KEYS, "OriginMidPrice"], use_threads=False)
        cells.append(evaluate_cell(cell["day"], cell["symbol"], table, original))
    expected = [(day, symbol) for day in DAYS for symbol in CARRIERS]
    full = [(cell["day"], cell["symbol"]) for cell in cells] == expected
    batch = [cell for cell in cells if cell["support_gate_required"]]
    gate = full and len(batch) == len(DAYS) * len(BATCH_SYMBOLS) and all(cell["support_gate_passed"] for cell in batch)
    result = {
        "schema": "h16-native-source-support-validation-v1",
        "method_identity": method["identity"],
        "producer_identity": method["producer_identity"],
        "passed": bool(full and gate),
        "native_artifact_integrity_passed": True,
        "all_original_cells_present": full,
        "support_gate": {"passed": bool(gate), "fixed_batch_symbols": list(BATCH_SYMBOLS), "per_cell": batch},
        "cells": cells,
        "missing": profile["missing"],
        "sources": method["sources"],
        "inputs": unique_records([*method["inputs"], record(evaluation_path), record(Path(evaluation_path).with_suffix(Path(evaluation_path).suffix + ".identity"))]),
        "labels_read": False,
        "model_fits": 0,
        "raw_streams_decoded": 0,
        "native_replays_launched": 0,
        "python_features_generated": False,
        "predictive_admission": False,
        "native_receive_prefix_proof_passed": False,
    }
    destination = Path(destination) if destination else OUTPUT / "source-support-validation.yaml"
    require(not destination.exists(), "fresh validation receipt required")
    write_yaml(destination, result)
    return {"receipt": str(destination), "integrity_passed": True, "all32_present": full, "support_passed": bool(gate), "predictive_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-metadata", type=Path, metavar="NEW_ROOT_BUILT_BINARY")
    mode.add_argument("--preflight", type=Path, metavar="FROZEN_PRODUCER")
    mode.add_argument("--bind-evaluation", type=Path, metavar="FROZEN_PRODUCER")
    mode.add_argument("--check", type=Path, metavar="FROZEN_EVALUATION")
    parser.add_argument("--job", choices=("registration_control", *DAYS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare_metadata:
        result = prepare_metadata(args.prepare_metadata)
        payload = result.pop("producer_payload")
        path = args.output or OUTPUT / "draft-producer-method.yaml"
        write_yaml(path, payload)
        result["producer_payload_path"] = str(path)
    elif args.preflight:
        result = preflight(args.preflight, args.job)
    elif args.bind_evaluation:
        payload = bind_evaluation(args.bind_evaluation)
        path = args.output or OUTPUT / "draft-support-method.yaml"
        write_yaml(path, payload)
        result = {"evaluation_payload_path": str(path), "native_values_decoded": 0}
    else:
        result = check(args.check, args.output)
    print(result)


if __name__ == "__main__":
    main()
