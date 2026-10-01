"""Root-only H15 full native metadata preparation and exact label-free caches.

No helper launches coco. Producer and validation methods are externally frozen
before replay and before any native Parquet decoding respectively.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]
PARENT = OUTPUT.parent / "native-h15-peer-trade-v2"
MAPPING_ROOT = OUTPUT.parent / "native-h15-mapping-support"
PREFIX_ROOT = OUTPUT.parent / "native-h15-receive-prefix"
OLD_FULL = OUTPUT.parent / "native-h10-ordered-reset/full-preparation-receipt.yaml"
OLD_V1 = OUTPUT.parent / "v1/frozen-contract.yaml"
SHARED = OUTPUT.parent / "v4/shared"
MID_SOURCE = OUTPUT.parent / "v3/shared"
PARENT_STUDY = ROOT / "AstraResearch/runs/fe_origin_20260930/study-stock"
STORE = ROOT / "AstraResearch/runs/fe-origin-store/objects"
PLAN = OUTPUT / "FULL_PLAN.md"
PREPARATION = OUTPUT / "preparation-receipt.yaml"
BOUND_VALIDATION = OUTPUT / "bound-validation.yaml"
VALIDATION = OUTPUT / "validation.yaml"
FEATURE_RECEIPT = SHARED / "h15-preparation.yaml"
KEYS = ("day", "symbol", "SampleTime", "SampleBookTime", "SampleBookSeq")
NATIVE_KEYS = KEYS[2:]
ROLES = ("train", "tune", "forward")
SYMBOLS = ("2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2615", "3481", "2409", "2344", "2337", "2481", "3037", "3711")
MAPPING = {symbol: "2317" if symbol == "2330" else "2330" for symbol in SYMBOLS}
BINARY_SHA256 = "613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b"
PILOT_DAYS = ("20260119", "20260120")
sys.path.insert(0, str(PARENT))
import evaluate_native_support as native
from prepare_native import canonical_method, check_record, record, require, require_closure, unique_records
from astra.io import digest, read_yaml, write_yaml

require(Path(native.__file__).resolve() == PARENT / "evaluate_native_support.py", "fresh process with immutable V2 native validator required")
NUMERIC = tuple(f"H15.{name}.0" for name in native.ALPHA_NUMERIC)
CATEGORICAL = tuple(f"H15.{name}.0" for name in native.ALPHA_CATEGORICAL)
CANONICAL = (*NUMERIC, *CATEGORICAL)
CONTRACT = {
    "schema": "h15-full-material-contract-v1", "days": 161, "symbols": list(SYMBOLS), "mapping": MAPPING,
    "periodic_seconds": 10, "native_rows_per_original_cell": 1381, "numeric": list(NUMERIC), "categorical": list(CATEGORICAL),
    "missing": "fixed raw/BasicInfo omission skips without replacement; original cohort join failure means NOT PASS",
    "states": "immutable V2 native validate_table; no feature arithmetic, sentinel repair or feature availability row filter",
    "join": "exact original five keys and float64 raw-mid bits in original parent role order",
    "role_parent_keys_hash": "opaque file_hash(parent_study/role/rows.parquet); decode only fivekeys",
    "role_parent_mid_hash": "file_hash(v4/shared/mid-role.npy), exact original V3 float64 hardlink supplied by root",
    "labels_read": False, "models_permitted": 0,
}


def require_passed_gates(prefix, coverage):
    require(prefix.get("schema") == "h15-real-receive-prefix-validation-v1"
            and all(prefix.get(k) is True for k in ("passed", "genuine_straddle_found", "common_origin_present", "all_common_origin_bits_exact", "pending_cluster_not_published_at_origin"))
            and prefix.get("binary_sha256") == BINARY_SHA256 and prefix.get("labels_read") is False
            and type(prefix.get("model_fits")) is int and prefix["model_fits"] == 0, "passed exact native H15 real receive-prefix prerequisite required")
    require(coverage.get("schema") == "h15-fixed-mapping-source-coverage-validation-v1"
            and coverage.get("native_artifact_integrity_passed") is True and coverage.get("coverage_gate", {}).get("passed") is True
            and coverage.get("mapping") == MAPPING and coverage.get("labels_read") is False
            and type(coverage.get("model_fits")) is int and coverage["model_fits"] == 0, "passed prospective fixed16 mapping coverage prerequisite required")


def prerequisites():
    prefix_path, coverage_path = PREFIX_ROOT / "validation.yaml", MAPPING_ROOT / "coverage-validation.yaml"
    require(prefix_path.is_file() and coverage_path.is_file(), "root must complete prefix and mapping gates before full preparation")
    prefix, coverage = read_yaml(prefix_path), read_yaml(coverage_path)
    require_passed_gates(prefix, coverage)
    paths = ((PREFIX_ROOT / "frozen-selection-method.yaml", "h15-real-receive-prefix-selection-method-v1", prefix["selection_method_identity"]),
             (PREFIX_ROOT / "frozen-comparison-method.yaml", "h15-real-receive-prefix-comparison-method-v1", prefix["comparison_method_identity"]),
             (MAPPING_ROOT / "frozen-method.yaml", "h15-fixed-mapping-producer-method-v1", coverage["producer_identity"]),
             (MAPPING_ROOT / "frozen-evaluation-method.yaml", "h15-fixed-mapping-evaluation-method-v1", coverage["method_identity"]))
    methods, links = [], [record(prefix_path), record(coverage_path)]
    for path, schema, identity in paths:
        method = canonical_method(path, schema)
        require(method["identity"] == identity, "prerequisite receipt/method identity mismatch")
        methods.append(method)
        links.extend([record(path), record(path.with_suffix(".yaml.identity"))])
    production = methods[2]
    check_record(production["preparation"])
    mapping = read_yaml(Path(production["preparation"]["path"]))
    require(mapping["binary"]["sha256"] == BINARY_SHA256 and mapping["parent_pair_producer_identity"] == prefix["producer_identity"], "prefix/mapping native producer lineage differs")
    require(mapping["runtime_dependencies"] and len(mapping["runtime_dependencies"]) == 51, "same51 shared runtime required")
    check_record(mapping["shared_source_capsule"])
    capsule = read_yaml(Path(mapping["shared_source_capsule"]["path"]))
    require(len(capsule["sources"]) == 1648, "same1648 immutable compiled source copies required")
    sources = unique_records([*[n for method in methods for n in method["sources"]], *prefix["sources"], *coverage["sources"],
                              record(Path(__file__)), record(PLAN), record(OUTPUT / "test_full.py")])
    inputs = unique_records([*[n for method in methods for n in method["inputs"]], *prefix["inputs"], *coverage["inputs"], *links,
                             production["preparation"], mapping["shared_source_capsule"], mapping["binary"], mapping["runtime_snapshot"], *mapping["runtime_dependencies"]])
    return mapping, sources, inputs


def bounded_native_refs(path, limit=16 * 1024 * 1024):
    """Read only metadata.native_artifacts, stopping before giant partitions."""
    collected, size, active = [], 0, False
    with path.open() as stream:
        for line in stream:
            size += len(line.encode())
            require(size <= limit, "native reference prefix exceeded fixed16MiB bound; never parse aggregate manifest")
            if not active:
                if line.rstrip() == "  native_artifacts:":
                    active = True
                continue
            if line.strip() and len(line) - len(line.lstrip()) <= 2 and not line.startswith("  - "):
                break
            collected.append(line[2:])
    require(active and collected, "original projection has no bounded native reference sequence")
    refs = yaml.safe_load("native_artifacts:\n" + "".join("  " + line for line in collected))["native_artifacts"]
    require(type(refs) is list and all(type(n) is dict and set(n) == {"identity", "kind"} and re.fullmatch("[0-9a-f]{64}", n["identity"]) for n in refs), "invalid original native identity reference")
    return [n for n in refs if n["kind"] == "native_export_day"]


def original_parents(days):
    references, nodes = {}, []
    for role in ROLES:
        path = PARENT_STUDY / role / "projection.json"
        projection = json.loads(path.read_text())
        ref = projection["recipe"]["dataset"]
        require(ref["kind"] == "feature_dataset" and re.fullmatch("[0-9a-f]{64}", ref["identity"]), "original immutable role projection dataset identity required")
        aggregate = STORE / ref["identity"] / "artifact.yaml"
        references.update({n["identity"]: n for n in bounded_native_refs(aggregate)})
        nodes.extend([record(path), record(aggregate)])
    parents, found = [], set()
    for identity in sorted(references):
        path = STORE / identity / "artifact.yaml"
        body = read_yaml(path)  # small daily native manifest only
        require(body["kind"] == "native_export_day" and digest(body) == identity, "canonical original native day identity required")
        metadata = body["metadata"]
        require(len(metadata["days"]) == 1, "native parent must be a single original day")
        day = str(metadata["days"][0])
        if day not in days:
            continue
        require(day not in found and metadata["symbols"] == list(SYMBOLS), "duplicate original day or changed fixed16 universe")
        found.add(day)
        files = {entry["path"]: entry for entry in body["files"]}
        seen = set()
        for part in metadata["partitions"]:
            symbol = str(part["symbol"])
            require(str(part["day"]) == day and symbol in SYMBOLS and symbol not in seen, "invalid original native cell identity")
            seen.add(symbol)
            relative = part["path"]
            require(relative == f"{day}/{symbol}/data/{day}/{symbol}/values.parquet", "original native cell path changed")
            parents.append({"day": day, "symbol": symbol, "parent_origins": {**files[relative], "path": str(path.parent / relative)}})
        nodes.append(record(path))
    require(found == set(days), "a fixed original native day manifest is missing; no synthetic parent")
    return sorted(parents, key=lambda c: (c["day"], SYMBOLS.index(c["symbol"]))), unique_records(nodes)


def select_day(day, raw, basic, parents):
    missing = []
    for symbol in SYMBOLS:
        node = raw.get((day, symbol))
        if node is None or check_record(node, allow_missing=True) is None:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_rawBIN_skip_whole_day_no_replacement"})
    if basic is None or check_record(basic, allow_missing=True) is None:
        return [], [*missing, {"day": day, "reason": "missing_BasicInfo_skip_whole_day_no_replacement"}]
    with Path(basic["path"]).open(newline="") as stream:
        contracts = list(csv.DictReader(stream))
    for symbol in SYMBOLS:
        matching = [row for row in contracts if row["symbol"] == symbol]
        require(len(matching) <= 1, "duplicate fixed contract metadata")
        if not matching:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_contract_skip_whole_day"})
    if missing:
        return [], missing
    present = []
    for cell in parents:
        if cell["day"] == day:
            if check_record(cell["parent_origins"], allow_missing=True) is None:
                missing.append({"day": day, "symbol": cell["symbol"], "reason": "missing_original_parent_skip_target_writer_no_replacement"})
            else:
                present.append(cell["symbol"])
    return present, missing


def full_config(original, present):
    groups = []
    for group in original["Modules"]:
        if not group["Gid"]:
            groups.append(copy.deepcopy(group))
            continue
        symbol = group["Gid"]
        current = next(d for d in group["Decl"] if d["Desc"] == "CurrentBook.0")
        declarations = [{"Desc": "TwseFilter.0", "Spec": {"RequireTradable": False, "StatusFilter": "TRIAL || !TRIAL", "Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}}, copy.deepcopy(current)]
        if symbol in present:
            peer = MAPPING[symbol]
            declarations.append({"Desc": "PeerTradeInformation.0", "Spec": {"TargetSymbol": symbol, "PeerSymbol": peer, "StatusFilter": "TRIAL || !TRIAL",
                                 "Dep": {"Book": [f"TwseFilter.0@{symbol}", f"TwseFilter.0@{peer}"], "Trade": [f"TwseFilter.0@{symbol}", f"TwseFilter.0@{peer}"]}}})
            metadata = [{"Feature": f"CurrentBook.0.{field}.0@{symbol}", "Name": name} for field, name in
                        (("book_mid_price", "OriginMidPrice"), ("book_mid_ticks", "OriginMidTicks"), ("book_bid_ticks", "OriginBidTicks"), ("book_ask_ticks", "OriginAskTicks"))]
            metadata += [{"Feature": f"PeerTradeInformation.0.{name}.0@{symbol}", "Name": f"H15_{name}"} for name in native.INFO]
            declarations.append({"Desc": "DatasetWriter.0", "Spec": {"Subscribe": [{"Book": [symbol]}], "Exports": list(f"PeerTradeInformation.0.{name}.0@{symbol}" for name in native.ALPHA),
                                 "MetadataExports": metadata, "PeriodicSampler": {"StartTime": "091000", "UntilTime": "130000", "SampleInterval": "10s"},
                                 "OutputPath": "${cwd}/data/${trading_date}/" + symbol + "/values.parquet", "Format": "parquet", "UseTmp": True,
                                 "EmitSampleContext": True, "RequireAlphaFactorExports": True}})
        groups.append({"Gid": symbol, "Decl": declarations})
    require([g["Gid"] for g in groups if g["Gid"]] == list(SYMBOLS), "fixed16 source order changed")
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def reused_job(job, identity):
    path = Path(job["work"]) / "execution-receipt.yaml"
    execution = read_yaml(path)
    require(execution.get("method_identity") == identity and execution.get("job") == job["day"] and execution.get("exit_code") == 0
            and type(execution.get("exit_code")) is int and execution.get("command") == job["command"]
            and [n["path"] for n in execution.get("outputs", [])] == job["outputs"], "exact successful mapping pilot execution required for reuse")
    status_path = Path(job["work"]) / "status" / f"{job['day']}.yaml"
    status = read_yaml(status_path)
    require(status.get("status") == "completed" and status.get("fatal_error") is False, "mapping pilot not completed")
    links = [record(path), record(status_path), execution["log"], *execution["outputs"], job["config"]]
    for node in links:
        check_record(node)
    result = copy.deepcopy(job)
    result.update(requires_replay=False, command=[], reused_command=job["command"], reused_execution_receipt=record(path), reused_producer_identity=identity, reuse_existing_outputs=True)
    return result, links


def role_descriptors(profile):
    days = {"train": profile["splits"]["train"], "tune": profile["splits"]["es"] + profile["splits"]["calibration"], "forward": profile["splits"]["development"]}
    return [{"role": role, "requested_days": list(map(str, days[role])), "parent_keys": record(PARENT_STUDY / role / "rows.parquet"),
             "parent_mid": record(MID_SOURCE / f"mid-{role}.npy"), "target_mid_path": str(SHARED / f"mid-{role}.npy"), "cache_path": str(SHARED / f"{role}-h15.parquet")} for role in ROLES]


def prepare_full():
    require(not PREPARATION.exists() and not (OUTPUT / "full").exists(), "fresh full H15 preparation required")
    mapping, sources, inputs = prerequisites()
    full = read_yaml(OLD_FULL)
    days = list(map(str, full["fixed_role_days"]))
    require(days == sorted(set(days)) and len(days) == 161 and min(days) >= "20260101" and full["universe_order"] == list(SYMBOLS), "fixed161 native dates/full16 universe changed")
    check_record(full["reference_profile"])
    profile = read_yaml(Path(full["reference_profile"]["path"]))
    require(sorted({str(day) for role in ("train", "es", "calibration", "development") for day in profile["splits"][role]}) == days, "original frozen role day union changed")
    parents, original_nodes = original_parents(days)
    raw = {(n["day"], n["symbol"]): n for n in full["raw_inputs"]}
    require(all(n["path"].endswith(".bin.zst") for n in raw.values()), "same original actual BIN loader identities required; no fallback")
    basics = {Path(n["path"]).stem: n for n in full["basic_info_inputs"]}
    template = mapping["jobs"][0]["config"]
    check_record(template)
    original = read_yaml(Path(template["path"]))
    roles = role_descriptors(profile)
    jobs, missing, reuse_links = [], [], []
    for day in days:
        if day in PILOT_DAYS:
            old = next(j for j in mapping["jobs"] if j["day"] == day)
            reused, links = reused_job(old, read_yaml(MAPPING_ROOT / "coverage-validation.yaml")["producer_identity"])
            require(read_yaml(Path(old["config"]["path"])) == full_config(original, old["symbols"]), "mapping pilot full recipe differs from requested full producer")
            jobs.append(reused)
            reuse_links.extend(links)
            continue
        present, skipped = select_day(day, raw, basics.get(day), parents)
        missing.extend(skipped)
        work = OUTPUT / "full" / day
        work.mkdir(parents=True)
        config = work / "config.yaml"
        write_yaml(config, full_config(original, present))
        jobs.append({"day": day, "symbols": present, "work": str(work), "config": record(config), "requires_replay": bool(present),
                     "command": [mapping["binary"]["path"], "-d", day, "-C", str(work), "--trading-calendar", str(native.ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"),
                                 "--run-status-dir", str(work / "status"), str(config)] if present else [],
                     "environment": {"LD_LIBRARY_PATH": mapping["runtime_directory"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                     "outputs": [str(work / "data" / day / symbol / "values.parquet") for symbol in present]})
    inputs = unique_records([*inputs, record(OLD_FULL), record(OLD_V1), full["reference_profile"], template, *original_nodes, *raw.values(), *basics.values(),
                             *[c["parent_origins"] for c in parents], *reuse_links, *[r[k] for r in roles for k in ("parent_keys", "parent_mid")]])
    optional = {n["path"] for n in [*raw.values(), *basics.values(), *[c["parent_origins"] for c in parents]]}
    absent = [n["path"] for n in inputs if check_record(n, allow_missing=n["path"] in optional) is None]
    receipt = {"schema": "h15-full-native-preparation-v1", "contract": CONTRACT, "script": record(Path(__file__)), "plan": record(PLAN),
               "binary": mapping["binary"], "shared_source_capsule": mapping["shared_source_capsule"], "runtime_snapshot": mapping["runtime_snapshot"],
               "runtime_dependencies": mapping["runtime_dependencies"], "runtime_directory": mapping["runtime_directory"], "prerequisites": {
                   "prefix": record(PREFIX_ROOT / "validation.yaml"), "mapping": record(MAPPING_ROOT / "coverage-validation.yaml")},
               "days": days, "roles": roles, "parents": parents, "jobs": jobs, "missing": missing, "required_absent_inputs": sorted(absent),
               "sources": sources, "inputs": inputs, "execution_order": [j["day"] for j in jobs if j["requires_replay"]], "reused_days": list(PILOT_DAYS),
               "labels_read": False, "model_fits": 0, "native_replays_launched": 0}
    write_yaml(PREPARATION, receipt)
    return {"prepared_only": True, "jobs": len(jobs), "reused": 2, "fresh_replays": len(receipt["execution_order"]), "missing": len(missing)}


def verify_producer(path, job_key=None):
    method = canonical_method(path, "h15-full-native-producer-method-v1")
    require(method["preparation"] == record(PREPARATION), "wrong full preparation binding")
    preparation = read_yaml(PREPARATION)
    require(method["preparation_identity"] == digest(preparation) and preparation["contract"] == CONTRACT, "frozen full native recipe changed")
    mapping, sources, inputs = prerequisites()
    require(preparation["binary"] == mapping["binary"] and preparation["shared_source_capsule"] == mapping["shared_source_capsule"], "full producer escaped shared native binary/source")
    require_closure(method, "sources", sources)
    require_closure(method, "inputs", [*inputs, *preparation["inputs"], record(PREPARATION), *[j["config"] for j in preparation["jobs"]]])
    for scope in ("sources", "inputs"):
        for node in method[scope]:
            absent = node["path"] in preparation["required_absent_inputs"]
            require(check_record(node, allow_missing=absent) is None if absent else check_record(node) is not None, "frozen missing input appeared")
    template = read_yaml(Path(mapping["jobs"][0]["config"]["path"]))
    selected = preparation["jobs"] if job_key is None else [j for j in preparation["jobs"] if j["day"] == job_key]
    require(job_key is None or len(selected) == 1, "unknown fixed daily job")
    for job in selected:
        require(read_yaml(Path(job["config"]["path"])) == full_config(template, job["symbols"]), "daily mapping/native export/sampler recipe drift")
        if job.get("reuse_existing_outputs") is True:
            require(job["day"] in PILOT_DAYS and job["requires_replay"] is False and job["command"] == [], "mapping reuse cannot relaunch")
            continue
        require(type(job["requires_replay"]) is bool and job["requires_replay"] == bool(job["symbols"]), "missing day cannot replay fallback")
        if job_key and job["requires_replay"]:
            require(not any(Path(p).exists() for p in job["outputs"]) and not (Path(job["work"]) / "status").exists(), "fresh native outputs/status required")
    return method, preparation


def bind_validation(producer_path):
    method, preparation = verify_producer(producer_path)
    require(not BOUND_VALIDATION.exists(), "fresh native validation bind required")
    files, outputs, execution_nodes = [], [], []
    for job in preparation["jobs"]:
        if not job["symbols"]:
            continue
        execution_path = Path(job["work"]) / "execution-receipt.yaml"
        execution = read_yaml(execution_path)
        identity = job["reused_producer_identity"] if job.get("reuse_existing_outputs") else method["identity"]
        command = job["reused_command"] if job.get("reuse_existing_outputs") else job["command"]
        require(execution.get("method_identity") == identity and execution.get("job") == job["day"] and type(execution.get("exit_code")) is int
                and execution["exit_code"] == 0 and execution.get("command") == command and [n["path"] for n in execution.get("outputs", [])] == job["outputs"], "full native execution differs from frozen job")
        status_path = Path(job["work"]) / "status" / f"{job['day']}.yaml"
        status = read_yaml(status_path)
        require(status.get("status") == "completed" and status.get("fatal_error") is False, "full native day incomplete")
        execution_nodes.extend([record(execution_path), record(status_path), execution["log"]])
        for symbol, node in zip(job["symbols"], execution["outputs"], strict=True):
            check_record(node)
            files.append({"day": job["day"], "symbol": symbol, **node})
            outputs.append(node)
    mids = []
    for role in preparation["roles"]:
        path = Path(role["target_mid_path"])
        require(path.is_file() and record(path)["sha256"] == role["parent_mid"]["sha256"], "root must supply exact original mid-role hardlinks before validation freeze")
        mids.append(record(path))
    missing = sorted(set((c["day"], c["symbol"]) for c in preparation["parents"]) - set((c["day"], c["symbol"]) for c in files))
    body = {"schema": "h15-full-material-bound-validation-v1", "contract": CONTRACT, "producer_method": record(producer_path), "producer_identity": method["identity"],
            "preparation": record(PREPARATION), "native_files": files, "missing_original_cells": [list(c) for c in missing], "roles": preparation["roles"],
            "sources": method["sources"], "inputs": unique_records([*method["inputs"], record(producer_path), record(producer_path.with_suffix(".yaml.identity")), *execution_nodes, *outputs, *mids]),
            "native_outputs_decoded": False, "labels_read": False, "model_fits": 0}
    write_yaml(BOUND_VALIDATION, body)
    return {"bound_only": True, "native_files": len(files), "missing_original_cells": len(missing), "receipt": str(BOUND_VALIDATION)}


def verify_validation(path):
    method = canonical_method(path, "h15-full-material-validation-method-v1")
    require(method["plan"] == record(BOUND_VALIDATION), "wrong full validation bind")
    bound = read_yaml(BOUND_VALIDATION)
    production, preparation = verify_producer(Path(bound["producer_method"]["path"]))
    require(bound["producer_identity"] == production["identity"] and bound["contract"] == CONTRACT and bound["preparation"] == record(PREPARATION)
            and bound["native_outputs_decoded"] is False and bound["labels_read"] is False and type(bound["model_fits"]) is int and bound["model_fits"] == 0, "native validation must be frozen before decode")
    for scope in ("sources", "inputs"):
        require_closure(method, scope, [*bound[scope], *([record(BOUND_VALIDATION)] if scope == "inputs" else [])])
        for node in method[scope]:
            check_record(node, allow_missing=node["path"] in preparation["required_absent_inputs"])
    return method, bound, preparation


def exact_keys(table):
    require(all(table[name].type == pa.int64() and table[name].null_count == 0 for name in NATIVE_KEYS), "exact nonnullable int64 native keys required")
    keys = list(zip(*(table[name].to_pylist() for name in NATIVE_KEYS), strict=True))
    require(len(keys) == len(set(keys)) and all(0 < k[0] < 2**53 and 0 <= k[1] <= k[0] and 0 <= k[2] < 2**53 for k in keys), "duplicate/unsafe/future native keys")
    return keys


def validate_original_cell(original, exported):
    native.validate_table(exported, periodic=True)
    require(len(original) == len(exported) == 1381, "original native full10s cell population changed")
    require(exact_keys(original) == exact_keys(exported), "all original native context keys must match in order")
    require(original["OriginMidPrice"].type == pa.float64() and native.same_bits(original["OriginMidPrice"], exported["OriginMidPrice"]), "original native raw mid bits changed")


def exact_join(anchors, exported, mids):
    require(list(anchors.columns) == list(KEYS) and not anchors.isna().any().any() and not anchors.duplicated(list(KEYS)).any(), "unique exact five-key parent anchors required")
    for name in NATIVE_KEYS:
        require(pd.api.types.is_integer_dtype(anchors[name]), "parent native key cannot be rounded/coerced")
    require(mids.dtype == np.float64 and len(mids) == len(anchors), "exact original float64 mid alignment required")
    positions = {key: i for i, key in enumerate(exact_keys(exported))}
    parent_keys = list(zip(*(anchors[name].tolist() for name in NATIVE_KEYS), strict=True))
    require(all(key in positions for key in parent_keys), "missing exact parent native key; no asof/forwardfill/row drop")
    selected = exported.take(pa.array([positions[key] for key in parent_keys], type=pa.int64()))
    require(np.array_equal(selected["OriginMidPrice"].to_numpy().view(np.uint64), mids.view(np.uint64)), "original role mid bits changed")
    arrays = {name: pa.array(anchors[name], type=pa.string() if name in KEYS[:2] else pa.int64()) for name in KEYS}
    arrays["OriginMidPrice"] = selected["OriginMidPrice"]
    arrays.update({canonical: selected[source] for canonical, source in zip(CANONICAL, native.ALPHA_COLUMNS, strict=True)})
    return pa.table(arrays)


def prepare_cache(validation_path):
    method, bound, preparation = verify_validation(validation_path)
    require(not VALIDATION.exists() and not FEATURE_RECEIPT.exists(), "fresh full/cache validation receipts required")
    require(all(not Path(r["cache_path"]).exists() and not Path(r["cache_path"]).with_suffix(".staging.parquet").exists() for r in bound["roles"]), "fresh exact three cache outputs required")
    files = {(n["day"], n["symbol"]): n for n in bound["native_files"]}
    parents = {(n["day"], n["symbol"]): n for n in preparation["parents"]}
    base = {"schema": "h15-full-material-validation-v1", "passed": False, "all_original_cells_mid_clock_state_validated": False,
            "labels_read": False, "model_fits": 0, "method": record(validation_path), "method_identity": method["identity"], "producer_identity": bound["producer_identity"],
            "sources": method["sources"], "inputs": unique_records([*method["inputs"], record(validation_path), record(validation_path.with_suffix(".yaml.identity"))]),
            "native_files": bound["native_files"], "caches": []}
    validated, caches = set(), []
    try:
        require(not bound["missing_original_cells"], "declared missing input cannot supply the complete original cohort")
        require(set(files) == set(parents), "configured native/original cell populations disagree")
        for role in bound["roles"]:
            check_record(role["parent_keys"])
            mids_path = Path(role["target_mid_path"])
            require(record(mids_path)["sha256"] == role["parent_mid"]["sha256"], "original mid cache changed")
            anchors = pq.read_table(role["parent_keys"]["path"], columns=list(KEYS), use_threads=False).to_pandas()
            anchors["day"], anchors["symbol"] = anchors.day.astype(str), anchors.symbol.astype(str)
            mids = np.load(mids_path, mmap_mode="r")
            require(mids.dtype == np.float64 and len(mids) == len(anchors), "original parent role mid shape/storage changed")
            selected = anchors.day.isin(role["requested_days"]).to_numpy()
            positions_in_parent = np.flatnonzero(selected)
            anchors = anchors.loc[selected].reset_index(drop=True)
            require(len(anchors) > 0 and not anchors.duplicated(list(KEYS)).any(), "empty/duplicate fixed parent role cohort")
            pieces, positions = [], []
            for (day, symbol), indices in anchors.groupby(["day", "symbol"], sort=False).indices.items():
                cell = (day, symbol)
                require(cell in files and cell in parents, "original role cell missing; no replacement or silent drop")
                exported = pq.read_table(files[cell]["path"], columns=list(native.REQUIRED_COLUMNS), use_threads=False)
                original = pq.read_table(parents[cell]["parent_origins"]["path"], columns=[*NATIVE_KEYS, "OriginMidPrice"], use_threads=False)
                validate_original_cell(original, exported)
                validated.add(cell)
                pieces.append(exact_join(anchors.iloc[indices], exported, mids[positions_in_parent[indices]]))
                positions.extend(indices)
            combined = pa.concat_tables(pieces).take(pa.array(np.argsort(positions), type=pa.int64()))
            require(combined.select(list(KEYS)).to_pandas().equals(anchors[list(KEYS)]), "cache changed original parent role row order")
            cache_path = Path(role["cache_path"])
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            staging = cache_path.with_suffix(".staging.parquet")
            pq.write_table(combined, staging)
            caches.append({"role": role["role"], "path": str(cache_path), "sha256": record(staging)["sha256"], "rows": len(combined), "numeric": list(NUMERIC), "categorical": list(CATEGORICAL),
                           "parent_keys_sha256": role["parent_keys"]["sha256"], "parent_mid_sha256": record(mids_path)["sha256"]})
        # Original native cells outside selected anchor dates still require state/mid proof.
        for cell in sorted(set(parents) - validated):
            exported = pq.read_table(files[cell]["path"], columns=list(native.REQUIRED_COLUMNS), use_threads=False)
            original = pq.read_table(parents[cell]["parent_origins"]["path"], columns=[*NATIVE_KEYS, "OriginMidPrice"], use_threads=False)
            validate_original_cell(original, exported)
            validated.add(cell)
        require(validated == set(parents) and [c["role"] for c in caches] == list(ROLES), "all original cells and exact three role caches required")
        for cache in caches:
            Path(cache["path"]).with_suffix(".staging.parquet").rename(cache["path"])
        feature = {"schema": "h15-native-feature-preparation-v1", "passed": True, "labels_read": False, "model_fits": 0,
                   "all_original_parent_keys_mid_bits_validated": True, "native_files": bound["native_files"], "caches": caches,
                   "method_identity": method["identity"], "producer_identity": bound["producer_identity"], "sources": method["sources"], "inputs": base["inputs"]}
        write_yaml(FEATURE_RECEIPT, feature)
        base.update(passed=True, all_original_cells_mid_clock_state_validated=True, caches=caches, feature_preparation=record(FEATURE_RECEIPT))
        base["inputs"] = unique_records([*base["inputs"], record(FEATURE_RECEIPT), *[record(Path(c["path"])) for c in caches]])
    except (ValueError, OSError, pa.ArrowException) as error:
        base["failure"] = {"type": type(error).__name__, "message": str(error)}
        write_yaml(VALIDATION, base)
        raise
    write_yaml(VALIDATION, base)
    return {"passed": True, "receipt": str(VALIDATION), "cache_receipt": str(FEATURE_RECEIPT), "cells": len(validated), "caches": len(caches)}


def validate_cache(validation_path):
    method, bound, _ = verify_validation(validation_path)
    full, feature = read_yaml(VALIDATION), read_yaml(FEATURE_RECEIPT)
    require(full["schema"] == "h15-full-material-validation-v1" and full["passed"] is True and full["all_original_cells_mid_clock_state_validated"] is True
            and feature["schema"] == "h15-native-feature-preparation-v1" and feature["passed"] is True and feature["all_original_parent_keys_mid_bits_validated"] is True,
            "successful full original cohort and feature preparation proof required")
    require(feature["method_identity"] == full["method_identity"] == method["identity"] and feature["caches"] == full["caches"], "cache/full method identity or record disagreement")
    require([c["role"] for c in feature["caches"]] == list(ROLES), "exact three distinct role cache records required")
    for receipt in (full, feature):
        require(receipt["labels_read"] is False and type(receipt["model_fits"]) is int and receipt["model_fits"] == 0 and receipt["native_files"] == bound["native_files"], "labels/models/native file proof changed")
    for cache, role in zip(feature["caches"], bound["roles"], strict=True):
        require(cache["path"] == role["cache_path"] and cache["numeric"] == list(NUMERIC) and cache["categorical"] == list(CATEGORICAL)
                and cache["parent_keys_sha256"] == role["parent_keys"]["sha256"] and cache["parent_mid_sha256"] == role["parent_mid"]["sha256"], "cache escaped exact original role/field/hash contract")
        check_record(cache)
        require(pq.ParquetFile(cache["path"]).metadata.num_rows == cache["rows"], "cache row count changed")
    return {"passed": True, "caches": 3, "labels_read": False, "model_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--preflight", type=Path)
    modes.add_argument("--bind-validation", type=Path)
    modes.add_argument("--prepare-cache", type=Path)
    modes.add_argument("--validate-cache", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.job and not args.preflight:
        parser.error("--job requires --preflight")
    if args.prepare:
        result = prepare_full()
    elif args.preflight:
        method, _ = verify_producer(args.preflight.resolve(), args.job)
        result = {"preflight_passed": True, "method_identity": method["identity"], "job": args.job, "native_replays_launched": 0}
    elif args.bind_validation:
        result = bind_validation(args.bind_validation.resolve())
    elif args.prepare_cache:
        result = prepare_cache(args.prepare_cache.resolve())
    else:
        result = validate_cache(args.validate_cache.resolve())
    print(result)


if __name__ == "__main__":
    main()
