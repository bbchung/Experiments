"""Prospective native H15 fixed-mapping coverage orchestration and validation.

Only root prepares, freezes, replays, metadata-binds and separately freezes
evaluation before decoding any native output. No raw stream, material feature
calculation, labels, models or replay is implemented here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter
from datetime import date
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
PARENT = OUTPUT.parent / "native-h15-peer-trade-v2"
FULL_RECEIPT = OUTPUT.parent / "native-h10-ordered-reset/full-preparation-receipt.yaml"
PLAN = OUTPUT / "COVERAGE_PLAN.md"
TESTS = OUTPUT / "test_coverage.py"
PROFILE = OUTPUT / "coverage-profile.yaml"
PREPARATION = OUTPUT / "preparation-receipt.yaml"
EVALUATION_PROFILE = OUTPUT / "evaluation-profile.yaml"
CALENDAR = ROOT / "AstraResearch/experiments/fe_origin_20260930/calendar.yaml"
ORIGINAL_STORE = ROOT / "AstraResearch/runs/fe-origin-store/objects"
sys.path.insert(0, str(ROOT / "AstraResearch"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astra.io import digest, read_yaml, write_yaml

sys.path.insert(0, str(PARENT))
native = import_module("evaluate_native_support")
require_native_path = Path(native.__file__).resolve() == PARENT / "evaluate_native_support.py"
if not require_native_path:
    raise ValueError("immutable V2 native validator required; use a fresh process")
from prepare_native import canonical_method, check_record, record, unique_records

DAYS = ("20260119", "20260120")
SYMBOLS = ("2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2615", "3481", "2409", "2344", "2337", "2481", "3037", "3711")
MAPPING = {symbol: "2317" if symbol == "2330" else "2330" for symbol in SYMBOLS}
FIXED_CELLS = tuple((day, symbol) for day in DAYS for symbol in SYMBOLS)
PARENT_PRODUCER_ID = "00540736c9defcabf2eb9e577bee50840dd3eedd62e00b3fdd981572118a8b61"
PARENT_SUPPORT_ID = "0430f280dc5f18d71050507e878952c6e46cb11360006d59a5b0d8882d8614a5"
BINARY_SHA256 = "613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b"
ORIGINAL_IDS = {
    "20260119": "9c10e19edefa61c37bd7ba4c5c3d2d932eef9eb4a20815d4a1595d381b8bff1d",
    "20260120": "16036d233eb02c68622d5c9c54916e301c82161e1c0cd877f47e6eb729798d0b",
}
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
EVALUATION_CONTRACT = {
    "schema": "h15-mapping-coverage-evaluation-contract-v1",
    "typed_rules": CONTRACT,
    "mapping": MAPPING,
    "session": "Native session counter13:00-minus09:10; sourceA>=09:10 AND sealA'<13:00; initial carry-in excluded",
    "unique": "Native qualified sourceID with five finite native views;0<=S-peerA<=300s at actual461 original30s origins; count ID once; unknown targetwork allowed",
    "semantic_support": "Exact native ordered_response_support==1; no J/K vocabulary inequality or Python feature/predicate calculation",
    "states": "Immutable V2 validate_table exact schema/roles/partial states and independent clocks; only mapping/cohort validation is new",
    "missing": "Declared fixed missing input skip only; no fallback/backfill; configured writer missing/changed output rejects",
    "no_predictive_admission": True,
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def session_bounds(day):
    elapsed = (date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}") - date(1970, 1, 1)).days
    midnight = (elapsed * 86400 - 8 * 3600) * 1_000_000
    return midnight + (9 * 3600 + 10 * 60) * 1_000_000, midnight + 13 * 3600 * 1_000_000


def parent_context():
    producer_path = PARENT / "frozen-method.yaml"
    support_path = PARENT / "frozen-support-evaluation-method.yaml"
    producer = canonical_method(producer_path, "h15-native-producer-method-v2")
    support = canonical_method(support_path, "h15-native-support-evaluation-method-v2")
    require(producer["identity"] == PARENT_PRODUCER_ID and support["identity"] == PARENT_SUPPORT_ID, "wrong immutable H15 pair producer/support methods")
    check_record(producer["preparation"])
    preparation = read_yaml(Path(producer["preparation"]["path"]))
    check_record(preparation["profile"])
    profile = read_yaml(Path(preparation["profile"]["path"]))
    require(digest(preparation) == producer["preparation_identity"] and digest(profile) == producer["profile_identity"], "pair producer body changed")
    require(preparation["binary"]["sha256"] == BINARY_SHA256, "different native producer binary")
    require(digest(profile["contract"]) == digest(CONTRACT), "pair and fullmapping source/sampling/support numerics differ")
    require(profile["carrier_order"] == list(SYMBOLS), "fixed16 carrier order changed")
    for method in (producer, support):
        for nodes in (method["sources"], method["inputs"]):
            for node in nodes:
                absent = node["path"] in profile["required_absent_inputs"]
                path = check_record(node, allow_missing=absent)
                require(not absent or path is None, "previously absent fixed input appeared")
    links = [record(p) for p in (producer_path, producer_path.with_suffix(".yaml.identity"), support_path, support_path.with_suffix(".yaml.identity"))]
    links += [producer["preparation"], preparation["profile"], support["profile"]]
    return producer, support, preparation, profile, links


def original_parents():
    parents, manifests = [], []
    for day in DAYS:
        identity = ORIGINAL_IDS[day]
        path = ORIGINAL_STORE / identity / "artifact.yaml"
        body = read_yaml(path)  # small daily native manifest, never aggregate Store.resolve
        require(body["kind"] == "native_export_day" and digest(body) == identity, "canonical original native daily manifest required")
        metadata = body["metadata"]
        require(metadata["days"] == [day] and metadata["symbols"] == list(SYMBOLS), "original native day/full16 order changed")
        files = {n["path"]: n for n in body["files"]}
        parts = {(str(p["day"]), str(p["symbol"])): p for p in metadata["partitions"]}
        require(set(parts) == {(day, symbol) for symbol in SYMBOLS}, "original native16 partitions changed")
        for symbol in SYMBOLS:
            relative = parts[(day, symbol)]["path"]
            require(relative == f"{day}/{symbol}/data/{day}/{symbol}/values.parquet", "original native parent path changed")
            entry = files[relative]
            parents.append(
                {"day": day, "symbol": symbol, "parent_origins": {**entry, "path": str(path.parent / relative)}, "expected_native_rows": 1381, "expected_support_rows": 461}
            )
        manifests.append(record(path))
    return parents, manifests


def select_day(day, raw, basic, parents):
    missing = []
    for symbol in SYMBOLS:
        if check_record(raw[(day, symbol)], allow_missing=True) is None:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_carrierBIN_skip_daily_replay_no_fallback"})
    if check_record(basic, allow_missing=True) is None:
        missing.append({"day": day, "reason": "missing_BasicInfo_skip_daily_replay_no_replacement"})
        return [], missing
    with Path(basic["path"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))  # contract metadata only
    for symbol in SYMBOLS:
        selected = [row for row in rows if row["symbol"] == symbol]
        require(len(selected) <= 1, "duplicate fixed carrier contract metadata")
        if not selected:
            missing.append({"day": day, "symbol": symbol, "reason": "missing_fixed_carrier_contract_skip_daily_replay"})
    if missing:
        return [], missing
    present = []
    for cell in parents:
        if cell["day"] != day:
            continue
        if check_record(cell["parent_origins"], allow_missing=True) is None:
            missing.append({"day": day, "symbol": cell["symbol"], "reason": "missing_original_parent_skip_only_target_writer"})
        else:
            present.append(cell["symbol"])
    return present, missing


def daily_config(original, present):
    groups = []
    for group in original["Modules"]:
        if not group["Gid"]:
            groups.append(copy.deepcopy(group))
            continue
        symbol = group["Gid"]
        current = next(item for item in group["Decl"] if item["Desc"] == "CurrentBook.0")
        declarations = [
            {"Desc": "TwseFilter.0", "Spec": {"RequireTradable": False, "StatusFilter": "TRIAL || !TRIAL", "Subscribe": [{"Book": [symbol], "Trade": [symbol]}]}},
            copy.deepcopy(current),
        ]
        if symbol in present:
            peer = MAPPING[symbol]
            declarations.append(
                {
                    "Desc": "PeerTradeInformation.0",
                    "Spec": {
                        "TargetSymbol": symbol,
                        "PeerSymbol": peer,
                        "StatusFilter": "TRIAL || !TRIAL",
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
            metadata += [{"Feature": f"PeerTradeInformation.0.{family}.0@{symbol}", "Name": f"H15_{family}"} for family in native.INFO]
            declarations.append(
                {
                    "Desc": "DatasetWriter.0",
                    "Spec": {
                        "Subscribe": [{"Book": [symbol]}],
                        "Exports": [f"PeerTradeInformation.0.{family}.0@{symbol}" for family in native.ALPHA],
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
    require([g["Gid"] for g in groups if g["Gid"]] == list(SYMBOLS), "fixed16 native source/carrier order changed")
    return {"Users": copy.deepcopy(original["Users"]), "Modules": groups}


def validate_full_metadata(full, pair_profile):
    """The H10 receipt stores one chronological date list, not role lists."""
    days = full["fixed_role_days"]
    require(
        type(days) is list and all(type(day) is str for day in days) and days == sorted(set(days)) and days[:2] == list(DAYS),
        "fixed first2 chronological H10 dates changed",
    )
    require(pair_profile["fixed_train_days"] == list(DAYS), "immutable pair proof does not establish the same first2 TRAIN dates")
    require(full["universe_order"] == list(SYMBOLS), "fixed full16 order changed")


def prepare():
    require(not PREPARATION.exists() and not PROFILE.exists() and not (OUTPUT / "pilot").exists(), "fresh mapping producer preparation required")
    producer, support, parent, pair_profile, links = parent_context()
    check_record(parent["guide"])
    check_record(parent["runtime_snapshot"])
    runtime = read_yaml(Path(parent["runtime_snapshot"]["path"]))
    require(len(runtime["dependencies"]) == 51 and runtime["dependencies"] == parent["runtime_dependencies"], "shared51 runtime changed")
    for node in runtime["dependencies"]:
        check_record(node)
        check_record({"path": node["original_system_path"], "sha256": node["sha256"]})
    check_record(parent["binary"])
    check_record(parent["producer"])
    capsule = read_yaml(Path(parent["producer"]["path"]))
    require(len(capsule["sources"]) == 1648, "same1648 immutable native sourcecopies required")
    check_record(FULL_NODE := record(FULL_RECEIPT))
    full = read_yaml(FULL_RECEIPT)
    validate_full_metadata(full, pair_profile)
    raw = {(n["day"], n["symbol"]): n for n in full["raw_inputs"] if n["day"] in DAYS}
    require(set(raw) == set(FIXED_CELLS) and all(n["path"].endswith(".bin.zst") for n in raw.values()), "exact fixed32 BIN identities required")
    basics = {Path(n["path"]).stem: n for n in full["basic_info_inputs"] if Path(n["path"]).stem in DAYS}
    require(set(basics) == set(DAYS), "two fixed BasicInfo identities required")
    parents, original_manifests = original_parents()
    template = parent["jobs"][0]["config"]
    check_record(template)
    original = read_yaml(Path(template["path"]))
    jobs, missing = [], []
    for day in DAYS:
        present, skipped = select_day(day, raw, basics[day], parents)
        missing.extend(skipped)
        work = OUTPUT / "pilot" / day
        work.mkdir(parents=True)
        config = work / "config.yaml"
        write_yaml(config, daily_config(original, present))
        jobs.append(
            {
                "day": day,
                "symbols": present,
                "requires_replay": bool(present),
                "work": str(work),
                "config": record(config),
                "command": [parent["binary"]["path"], "-d", day, "-C", str(work), "--trading-calendar", str(CALENDAR), "--run-status-dir", str(work / "status"), str(config)]
                if present
                else [],
                "environment": {"LD_LIBRARY_PATH": parent["runtime_directory"], "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                "outputs": [str(work / "data" / day / symbol / "values.parquet") for symbol in present],
            }
        )
    inputs = unique_records(
        [
            *producer["inputs"],
            *support["inputs"],
            *links,
            FULL_NODE,
            record(CALENDAR),
            template,
            *original_manifests,
            *raw.values(),
            *basics.values(),
            *[c["parent_origins"] for c in parents],
        ]
    )
    optional = {n["path"] for n in [*raw.values(), *basics.values(), *[c["parent_origins"] for c in parents]]}
    absent = [n["path"] for n in inputs if check_record(n, allow_missing=n["path"] in optional) is None]
    profile = {
        "schema": "h15-fixed-mapping-coverage-profile-v1",
        "status": "prospective_requires_root_freeze",
        "fixed_train_days": list(DAYS),
        "symbols": list(SYMBOLS),
        "mapping": MAPPING,
        "contract": CONTRACT,
        "alpha_fields": list(native.ALPHA),
        "info_fields": list(native.INFO),
        "carrier_order": list(SYMBOLS),
        "parents": parents,
        "missing": missing,
        "required_absent_inputs": sorted(absent),
        "source_proof": [record(PLAN), record(TESTS)],
        "labelers": [],
        "periodic_only": True,
        "native_feature_calculation": "Immutable613260 native C++ only; same1648 sources/runtime51; J/K share seven native numeric fields",
        "qualification": "Source coverage only; no predictive/universe admission; no peer substitution after counts",
    }
    write_yaml(PROFILE, profile)
    sources = unique_records([*producer["sources"], *support["sources"], record(Path(__file__)), record(PLAN), record(TESTS)])
    receipt = {
        "schema": "h15-fixed-mapping-preparation-v1",
        "script": record(Path(__file__)),
        "profile": record(PROFILE),
        "profile_identity": digest(profile),
        "sources": sources,
        "inputs": inputs,
        "binary": parent["binary"],
        "guide": parent["guide"],
        "shared_source_capsule": parent["producer"],
        "runtime_snapshot": parent["runtime_snapshot"],
        "runtime_dependencies": parent["runtime_dependencies"],
        "runtime_directory": parent["runtime_directory"],
        "shared_pair_producer_identity": PARENT_PRODUCER_ID,
        "shared_pair_support_identity": PARENT_SUPPORT_ID,
        "registration_control_proof": "Four exact originalH13/V1 wholeParquet comparisons closed by immutable V2source/support methods; no new binary/source/build",
        "jobs": jobs,
        "execution_order": list(DAYS),
        "labels_read": False,
        "raw_streams_decoded": 0,
        "native_replays_launched": 0,
        "model_fits": 0,
    }
    for node in sources:
        check_record(node)
    write_yaml(PREPARATION, receipt)
    return {"prepared_only": True, "jobs": len(jobs), "outputs_configured": sum(len(j["symbols"]) for j in jobs), "native_replays_launched": 0, "model_fits": 0}


def producer_context(method_path):
    method = canonical_method(method_path, "h15-fixed-mapping-producer-method-v1")
    check_record(method["preparation"])
    require(Path(method["preparation"]["path"]).resolve() == PREPARATION, "wrong mapping preparation path")
    preparation = read_yaml(PREPARATION)
    check_record(preparation["profile"])
    require(Path(preparation["profile"]["path"]).resolve() == PROFILE, "wrong mapping profile path")
    profile = read_yaml(PROFILE)
    require(method["preparation_identity"] == digest(preparation) and method["profile_identity"] == digest(profile), "mapping producer body changed")
    require(
        profile["mapping"] == MAPPING and profile["symbols"] == list(SYMBOLS) and profile["fixed_train_days"] == list(DAYS) and digest(profile["contract"]) == digest(CONTRACT),
        "fixed mapping/universe/gates changed",
    )
    required_sources = unique_records([*preparation["sources"], record(Path(__file__)), record(PLAN), record(TESTS)])
    require(profile["source_proof"] == [record(PLAN), record(TESTS)], "mandatory coverage plan/tests omitted")
    required_inputs = unique_records(
        [
            *preparation["inputs"],
            preparation["script"],
            preparation["profile"],
            preparation["binary"],
            preparation["guide"],
            preparation["shared_source_capsule"],
            preparation["runtime_snapshot"],
            *preparation["runtime_dependencies"],
            *[j["config"] for j in preparation["jobs"]],
            record(PREPARATION),
        ]
    )
    for key, nodes in (("sources", required_sources), ("inputs", required_inputs)):
        bound = {n["path"]: n for n in method[key]}
        require(all(bound.get(n["path"], {}).get("sha256") == n["sha256"] for n in nodes), f"mandatory mapping producer {key} omitted")
        for node in method[key]:
            absent = node["path"] in profile["required_absent_inputs"]
            path = check_record(node, allow_missing=absent)
            require(not absent or path is None, "previously missing fixed input appeared")
    require(preparation["binary"]["sha256"] == BINARY_SHA256, "different mapping native binary")
    for node in preparation["runtime_dependencies"]:
        check_record({"path": node["original_system_path"], "sha256": node["sha256"]})
    return method, preparation, profile


def preflight(method_path, day=None):
    method, preparation, _ = producer_context(method_path)
    for job in preparation["jobs"]:
        if day is not None and job["day"] != day:
            continue
        require(type(job["requires_replay"]) is bool and job["requires_replay"] == bool(job["symbols"]), "mapping missing-job status changed")
        if job["requires_replay"]:
            require(
                job["command"][0] == preparation["binary"]["path"] and job["environment"]["LD_LIBRARY_PATH"] == preparation["runtime_directory"],
                "job escaped frozen native/runtime",
            )
            require(not any(Path(p).exists() for p in job["outputs"]) and not (Path(job["work"]) / "status").exists(), "fresh native job outputs/status required")
        else:
            require(job["command"] == [] and job["outputs"] == [], "missing daily job cannot replay fallback")
    return {"preflight_passed": True, "producer_identity": method["identity"], "day": day, "native_replays_launched": 0}


def evaluate_cell(day, symbol, table, parent):
    """Same frozen native state checks with the newly frozen full16 cohort only."""
    require((day, symbol) in FIXED_CELLS, "unregistered fixed mapping cell")
    data = native.validate_table(table, periodic=True)
    require(len(table) == len(parent) == 1381, "original native1381 cohort changed")
    for name in (*native.KEYS, "OriginMidPrice"):
        dtype = pa.float64() if name == "OriginMidPrice" else pa.int64()
        require(parent[name].type == dtype and native.same_bits(table[name], parent[name]), "original native keys/mid uint64bits changed")
    start, end = session_bounds(day)
    times = data["times"]
    require(times[0] == start and times[-1] == end and np.all(np.diff(times) == 10_000_000), "original native10s session cohort changed")
    mask = times % 30_000_000 == 0
    require(int(mask.sum()) == 461, "existing native461 support origins changed")
    info = data["info"]
    marks, highwater, cleared, last_hard = {}, 0, False, 0
    for i in range(len(table)):
        identity, hard = int(info["source_mark_id"][i]), int(info["cumulative_hard_censor_count"][i])
        if not identity and highwater:
            require(hard > last_hard or cleared, "native source ownership erased without hard boundary")
            cleared = True
        if identity:
            require(identity >= highwater and (not cleared or identity > highwater), "cleared native sourceID resurrected/regressed")
            highwater, cleared = identity, False
            fixed = tuple(
                int(info[name][i])
                for name in (
                    "peer_source_available_time",
                    "peer_source_exchange_time",
                    "target_baseline_available_time",
                    "target_baseline_receive_time",
                    "target_baseline_exchange_time",
                )
            )
            prices = tuple(np.float64(info[name][i]).view(np.uint64).item() for name in native.PRICE_INFO)
            seal, qualified = int(info["qualified_mark_available_time"][i]), int(info["mark_is_qualified"][i])
            previous = marks.setdefault(identity, {"fixed": fixed, "prices": prices, "seal": 0, "qualified": 0, "numeric": {}})
            require(previous["fixed"] == fixed and previous["prices"] == prices, "same sourceID changed frozen clocks/native mid facts")
            require(previous["seal"] == 0 or (seal == previous["seal"] and qualified == previous["qualified"]), "same sourceID changed qualification")
            if seal:
                previous["seal"], previous["qualified"] = seal, qualified
            for index in (0, 1):
                value = data["numeric"][i, index]
                if np.isfinite(value):
                    bits = np.float64(value).view(np.uint64).item()
                    require(index not in previous["numeric"] or previous["numeric"][index] == bits, "same sourceID changed frozen impulse/work")
                    previous["numeric"][index] = bits
        last_hard = hard
    names = ("session_qualified_mark_count", "session_up_qualified_mark_count", "session_down_qualified_mark_count")
    initial, terminal = ([int(info[name][i]) for name in names] for i in (0, -1))
    delta = [after - before for before, after in zip(initial, terminal, strict=True)]
    age = times - info["peer_source_available_time"]
    fresh = mask & data["fresh_qualified"] & (age >= 0) & (age <= 300_000_000)
    unique = {int(value) for value in info["source_mark_id"][fresh]}
    prior = {int(value) for value in info["source_mark_id"][fresh & (info["peer_source_available_time"] < start)]}
    return {
        "day": day,
        "symbol": symbol,
        "peer": MAPPING[symbol],
        "status": "observed_native",
        "integrity_passed": True,
        "periodic_rows": 1381,
        "support_rows": 461,
        "session_counter_at_start": initial,
        "session_counter_at_end": terminal,
        "session_marks": delta[0],
        "session_up_marks": delta[1],
        "session_down_marks": delta[2],
        "distinct_recent_sampled_marks": len(unique),
        "distinct_prior_session_recent_marks": len(prior),
        "distinct_within_session_recent_marks": len(unique - prior),
        "ordered_response_support_origins": int((mask & (info["ordered_response_support"] == 1)).sum()),
        "category_counts_on_original30s": {name: dict(Counter(values[mask].tolist())) for name, values in data["categories"].items()},
    }


def gates(cells):
    require([(c["day"], c["symbol"]) for c in cells] == list(FIXED_CELLS), "fixed chronological32-cell identities required")
    require(all(c["status"] in ("observed_native", "missing_fixed_input_skip") for c in cells), "unknown fixed cell disposition")
    observed = [c for c in cells if c["status"] == "observed_native"]
    for cell in observed:
        require(cell["peer"] == MAPPING[cell["symbol"]], "posthoc peer substitution rejected")
        require(
            type(cell["integrity_passed"]) is bool
            and all(
                type(cell[name]) is int and cell[name] >= 0
                for name in ("session_marks", "session_up_marks", "session_down_marks", "distinct_recent_sampled_marks", "ordered_response_support_origins")
            ),
            "native support types invalid",
        )
        require(cell["session_marks"] == cell["session_up_marks"] + cell["session_down_marks"], "native direction counts disagree")
    checks = {
        "at_least_one_present_cell": len(observed) >= 1,
        "all_present_integrity": all(c["integrity_passed"] for c in observed),
        "each_present5_session_marks": all(c["session_marks"] >= 5 for c in observed),
        "each_present5_distinct_recent_marks": all(c["distinct_recent_sampled_marks"] >= 5 for c in observed),
        "aggregate5_up": sum(c["session_up_marks"] for c in observed) >= 5,
        "aggregate5_down": sum(c["session_down_marks"] for c in observed) >= 5,
        "aggregate20_semantic_origins": sum(c["ordered_response_support_origins"] for c in observed) >= 20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_cells": len(observed),
        "missing_cells": len(cells) - len(observed),
        "all_fixed_cells_observed": len(observed) == len(cells),
    }


def bind_evaluation(producer_path):
    require(not EVALUATION_PROFILE.exists(), "fresh prospective coverage evaluation profile required")
    producer, preparation, profile = producer_context(producer_path)
    cells, statuses = [], []
    for job in preparation["jobs"]:
        day = job["day"]
        if job["requires_replay"]:
            status_path = Path(job["work"]) / "status" / f"{day}.yaml"
            status = read_yaml(status_path)
            require(status.get("status") == "completed" and status.get("fatal_error") is False, "mapping native producer not cleanly completed")
            statuses.append(record(status_path))
        for symbol in SYMBOLS:
            cell = {"day": day, "symbol": symbol, "peer": MAPPING[symbol], "status": "observed_native" if symbol in job["symbols"] else "missing_fixed_input_skip"}
            if cell["status"] == "observed_native":
                cell["data"] = record(Path(job["work"]) / "data" / day / symbol / "values.parquet")
            cells.append(cell)
    body = {
        "schema": "h15-fixed-mapping-evaluation-profile-v1",
        "status": "prospective_requires_root_freeze",
        "producer_identity": producer["identity"],
        "producer": record(producer_path),
        "producer_anchor": record(producer_path.with_suffix(producer_path.suffix + ".identity")),
        "evaluation_contract": EVALUATION_CONTRACT,
        "parents": profile["parents"],
        "status_receipts": statuses,
        "cells": cells,
        "source_proof": [record(PLAN), record(TESTS)],
        "labels_read": False,
        "model_fits": 0,
    }
    write_yaml(EVALUATION_PROFILE, body)
    return {"metadata_only_bound": True, "profile": str(EVALUATION_PROFILE), "profile_identity": digest(body), "native_outputs_decoded": 0}


def evaluation_context(method_path):
    method = canonical_method(method_path, "h15-fixed-mapping-evaluation-method-v1")
    check_record(method["profile"])
    require(Path(method["profile"]["path"]).resolve() == EVALUATION_PROFILE, "wrong coverage evaluation profile")
    profile = read_yaml(EVALUATION_PROFILE)
    require(method["profile_identity"] == digest(profile) and digest(profile["evaluation_contract"]) == digest(EVALUATION_CONTRACT), "coverage evaluation rules/profile changed")
    check_record(profile["producer"])
    check_record(profile["producer_anchor"])
    producer_path = Path(profile["producer"]["path"])
    producer, preparation, producer_profile = producer_context(producer_path)
    require(profile["producer_identity"] == producer["identity"] and profile["parents"] == producer_profile["parents"], "coverage producer/originalparents changed")
    require([(c["day"], c["symbol"]) for c in profile["cells"]] == list(FIXED_CELLS), "fixed32 evaluation cells changed")
    jobs = {j["day"]: j for j in preparation["jobs"]}
    for cell in profile["cells"]:
        job = jobs[cell["day"]]
        present = cell["symbol"] in job["symbols"]
        require(
            cell["peer"] == MAPPING[cell["symbol"]] and cell["status"] == ("observed_native" if present else "missing_fixed_input_skip"),
            "configured writer skipped or peer substituted",
        )
        if present:
            require(cell["data"]["path"] == str(Path(job["work"]) / "data" / cell["day"] / cell["symbol"] / "values.parquet"), "different configured native writer path")
    required_sources = unique_records([*producer["sources"], record(Path(__file__)), record(PLAN), record(TESTS), record(PARENT / "evaluate_native_support.py")])
    require(profile["source_proof"] == [record(PLAN), record(TESTS)], "essential coverage plan/tests omitted")
    required_inputs = unique_records(
        [
            *producer["inputs"],
            profile["producer"],
            profile["producer_anchor"],
            *profile["status_receipts"],
            *[c["data"] for c in profile["cells"] if c["status"] == "observed_native"],
            record(EVALUATION_PROFILE),
        ]
    )
    for key, nodes in (("sources", required_sources), ("inputs", required_inputs)):
        bound = {n["path"]: n for n in method[key]}
        require(all(bound.get(n["path"], {}).get("sha256") == n["sha256"] for n in nodes), f"mandatory coverage evaluation {key} closure omitted")
        for node in method[key]:
            absent = node["path"] in producer_profile["required_absent_inputs"]
            path = check_record(node, allow_missing=absent)
            require(not absent or path is None, "previously absent coverage input appeared")
    return method, profile


def evaluate(method_path, destination):
    require(not destination.exists(), "fresh coverage receipt required")
    method, profile = evaluation_context(method_path)
    parents = {(c["day"], c["symbol"]): c for c in profile["parents"]}
    cells = []
    for cell in profile["cells"]:
        if cell["status"] == "missing_fixed_input_skip":
            cells.append(cell)
            continue
        path = cell["data"]["path"]
        require(set(pq.read_schema(path).names) == set(native.REQUIRED_COLUMNS), "undeclared native output field/label")
        table = pq.read_table(path, columns=list(native.REQUIRED_COLUMNS), use_threads=False)
        original = pq.read_table(parents[(cell["day"], cell["symbol"])]["parent_origins"]["path"], columns=[*native.KEYS, "OriginMidPrice"], use_threads=False)
        cells.append(evaluate_cell(cell["day"], cell["symbol"], table, original))
    evaluation_context(method_path)
    result = {
        "schema": "h15-fixed-mapping-source-coverage-validation-v1",
        "method_identity": method["identity"],
        "profile_identity": digest(profile),
        "producer_identity": profile["producer_identity"],
        "method": record(method_path),
        "method_anchor": record(method_path.with_suffix(method_path.suffix + ".identity")),
        "sources": method["sources"],
        "inputs": method["inputs"],
        "mapping": MAPPING,
        "cells": cells,
        "coverage_gate": gates(cells),
        "native_artifact_integrity_passed": True,
        "native_receive_prefix_proof_passed": False,
        "native_prefix_required_before_fe": True,
        "predictive_admission": False,
        "labels_read": False,
        "raw_streams_decoded": 0,
        "native_replays_launched": 0,
        "model_fits": 0,
    }
    write_yaml(destination, result)
    return {"integrity_passed": True, "coverage_passed": result["coverage_gate"]["passed"], "receipt": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", type=Path)
    mode.add_argument("--bind-evaluation", type=Path, metavar="PRODUCER_METHOD")
    mode.add_argument("--evaluate", type=Path, metavar="EVALUATION_METHOD")
    parser.add_argument("--day", choices=DAYS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.evaluate and args.output is None:
        parser.error("--evaluate requires --output")
    if args.prepare:
        result = prepare()
    elif args.preflight:
        result = preflight(args.preflight.resolve(), args.day)
    elif args.bind_evaluation:
        result = bind_evaluation(args.bind_evaluation.resolve())
    else:
        result = evaluate(args.evaluate.resolve(), args.output.resolve())
    print(result)


if __name__ == "__main__":
    main()
