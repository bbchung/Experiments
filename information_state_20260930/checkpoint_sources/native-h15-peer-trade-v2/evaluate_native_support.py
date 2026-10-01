"""Validate frozen H15 native exports and label-free source support only.

Root binds file identities before separately freezing this evaluator/profile/
test/input closure. Only the frozen --evaluate step decodes native exports;
no raw stream, feature calculation, label, replay or learner is used.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "AstraResearch"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astra.io import digest, read_yaml, write_yaml
from prepare_native import (
    ALPHA,
    ALPHA_CATEGORICAL,
    ALPHA_NUMERIC,
    CARRIERS,
    CONTRACT,
    DAYS,
    INFO,
    MODULE,
    OUTPUT,
    PEERS,
    PROFILE,
    ROOT,
    TARGETS,
    canonical_method,
    check_record,
    record,
    unique_records,
)

KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")
RAW = ("OriginMidPrice", "OriginMidTicks", "OriginBidTicks", "OriginAskTicks")
ALPHA_COLUMNS = tuple(f"{MODULE}.0.{name}.0" for name in ALPHA)
NUMERIC_COLUMNS = tuple(f"{MODULE}.0.{name}.0" for name in ALPHA_NUMERIC)
CATEGORICAL_COLUMNS = tuple(f"{MODULE}.0.{name}.0" for name in ALPHA_CATEGORICAL)
INFO_COLUMNS = tuple(f"H15_{name}" for name in INFO)
PRICE_INFO = ("source_anchor_mid", "source_confirmed_mid", "target_baseline_mid")
INTEGER_INFO = tuple(name for name in INFO if name not in PRICE_INFO)
REQUIRED_COLUMNS = (*KEYS, *ALPHA_COLUMNS, *RAW, *INFO_COLUMNS)
FIXED_CELLS = tuple((day, symbol) for day in DAYS for symbol in TARGETS)
J_RELATIONS = {"waiting", "quote_later", "flow_later", "tied", "invalid"}
J_ALIGNMENTS = {"refuted", "unknown", "following", "absorbed", "opposing", "unresponsive"}
K_PHASES = {"invalid", "warmup", "stale", "unknown", "flow", "quote"}
FRESH_PHASES = {"unknown", "flow", "quote"}
EVALUATION_PROFILE = OUTPUT / "support-evaluation-profile.yaml"
EVALUATION_PLAN = OUTPUT / "SUPPORT_EVALUATION_PLAN.md"
TEST_SOURCE = OUTPUT / "test_evaluate_native_support.py"
EVALUATION_CONTRACT = {
    "schema": "h15-native-support-evaluation-contract-v2",
    "typed_support_rules": CONTRACT,
    "session": "Native session counter end-start at actual09:10/13:00 periodic snapshots; sourceA>=09:10 and sealA'<13:00; exclude warm-in source marks",
    "unique_support": "Existing461 inclusive13:00 origins; qualified nativeID with5 required finite views and0<=S-peerA<=300s; repeatedID once; work uncertainty allowed",
    "ordered_support": "Count exact native ordered_response_support==1 on existing30s origins; never compare J/K vocabulary or recompute native predicate",
    "clocks": "Every receive availability<=S; qualifiedA'>peerA; frozen target CLOSED baselineA<peerA with ownreceiveage<=5s; independent E has no E<=A/S/crossleg rule",
    "partial_states": "Native sealed historical peerwork/age may befinite during staleness/unqualifiedhistory; perleg fresh views and paired knownwork validated separately",
    "missing": "Only producer-declared fixed input/metadata omissions skip; configured writer missing/changed output rejects; no replacement",
    "control": "All4 original H13Jan19 data/events Parquet wholebytes equal under new registration",
    "no_predictive_admission": True,
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def session_bounds(day):
    elapsed = (date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}") - date(1970, 1, 1)).days
    midnight = (elapsed * 86400 - 8 * 3600) * 1_000_000
    return midnight + (9 * 3600 + 10 * 60) * 1_000_000, midnight + 13 * 3600 * 1_000_000


def column(table, name):
    return table[name].to_numpy(zero_copy_only=False)


def same_bits(left, right):
    if left.type != right.type or left.null_count or right.null_count or len(left) != len(right):
        return False
    if left.type == pa.float64():
        return np.array_equal(left.to_numpy().view(np.uint64), right.to_numpy().view(np.uint64))
    return left.equals(right)


def validate_table(table, *, periodic):
    require(
        len(table.column_names) == len(REQUIRED_COLUMNS) and set(table.column_names) == set(REQUIRED_COLUMNS),
        "exact native11Alpha/27Info schema required; no labels or extra feature fields",
    )
    for name in REQUIRED_COLUMNS:
        require(table[name].null_count == 0, f"native NULL unsupported: {name}")
        dtype = pa.int64() if name in KEYS else pa.string() if name in CATEGORICAL_COLUMNS else pa.float64()
        require(table[name].type == dtype, f"exact native dtype changed: {name}")
        role = (table.schema.field(name).metadata or {}).get(b"coco.role")
        expected_role = b"time" if name == "SampleTime" else b"context" if name in KEYS else b"feature" if name in ALPHA_COLUMNS else b"metadata"
        require(role == expected_role, f"native field role changed/diagnostic entered model: {name}")
    times, books, sequence = (column(table, name) for name in KEYS)
    require(np.all((times > 0) & (times < 2**53) & (books >= 0) & (books <= times) & (sequence >= 0) & (sequence < 2**53)), "native key boundaries invalid")
    require(np.all(np.diff(times) > 0) if periodic else np.all(np.diff(times) >= 0), "native origin order invalid")
    if not periodic:
        require(np.all(books == times) and np.all(np.diff(sequence) > 0), "all_book receive/source-sequence contract invalid; ties must remain")
    categories = {name: np.asarray(table[f"{MODULE}.0.{name}.0"].to_pylist()) for name in ALPHA_CATEGORICAL}
    for name, allowed in zip(ALPHA_CATEGORICAL, (J_RELATIONS, J_ALIGNMENTS, K_PHASES, K_PHASES), strict=True):
        require(set(categories[name]).issubset(allowed), f"unknown native categorical literal: {name}")
    info = {name: column(table, f"H15_{name}") for name in INFO}
    for name in INTEGER_INFO:
        values = info[name]
        require(np.all(np.isfinite(values) & (values >= 0) & (values < 2**53) & (values == np.floor(values))), f"native exact integer Info invalid: {name}")
        require(np.all(values[values == 0].view(np.uint64) == 0), f"native positive-zero Info identity invalid: {name}")
        info[name] = values.astype(np.int64)
    for name in PRICE_INFO:
        require(np.all(np.isfinite(info[name]) & (info[name] >= 0)), f"native fixed mid fact invalid: {name}")
        require(np.all(info[name][info[name] == 0].view(np.uint64) == 0), f"native fixed mid zero identity invalid: {name}")
    for leg in ("peer", "target"):
        available, exchange = info[f"processed_{leg}_available_time"], info[f"processed_{leg}_exchange_time"]
        require(np.all((available <= times) & ((available == 0) == (exchange == 0))), "processed receive availability or own-axis zero pairing invalid")
    counters = {name: info[name] for name in INTEGER_INFO if name.startswith(("cumulative_", "session_"))}
    require(all(np.all(np.diff(values) >= 0) for values in counters.values()), "native cumulative/session counter regressed")
    total = info["cumulative_qualified_mark_count"]
    source_count = info["cumulative_source_mark_count"]
    session = info["session_qualified_mark_count"]
    require(
        np.all((total <= source_count) & (session <= total)) and np.all(session == info["session_up_qualified_mark_count"] + info["session_down_qualified_mark_count"]),
        "native source/qualification/session counts disagree",
    )
    identity, peer_a, peer_e = info["source_mark_id"], info["peer_source_available_time"], info["peer_source_exchange_time"]
    qualified_a = info["qualified_mark_available_time"]
    baseline_a, baseline_r, baseline_e = (info[f"target_baseline_{axis}_time"] for axis in ("available", "receive", "exchange"))
    evidence_r, evidence_a = info["latest_target_evidence_receive_time"], info["latest_target_evidence_available_time"]
    qualified, support = info["mark_is_qualified"], info["ordered_response_support"]
    require(np.isin(qualified, (0, 1)).all() and np.isin(support, (0, 1)).all(), "native predicate Info must be exact0/1")
    require(np.all((identity == 0) | (identity == source_count)), "latest native source ID does not match cumulative source admission")
    require(np.all((identity == 0) == (peer_a == 0)) and np.all((identity == 0) == (peer_e == 0)), "sourceID/peer clocks zero pairing invalid")
    require(
        np.all(peer_a <= times) and np.all((qualified_a == 0) | ((qualified_a > peer_a) & (qualified_a <= times) & (identity > 0))),
        "qualification requires strictly later receive callback available at origin",
    )
    require(np.all((baseline_a == 0) == (baseline_r == 0)) and np.all((baseline_a == 0) == (baseline_e == 0)), "frozen target baseline own-axis zero pairing invalid")
    base = baseline_a > 0
    require(
        np.all(~base | ((identity > 0) & (baseline_a < peer_a) & (baseline_r <= baseline_a) & (peer_a - baseline_r <= 5_000_000))),
        "target CLOSED baseline is not strictly prior/fresh at source admission",
    )
    require(
        np.all((evidence_r == 0) == (evidence_a == 0)) and np.all((evidence_a <= times) & ((evidence_r == 0) | ((evidence_r > peer_a) & (evidence_r <= evidence_a)))),
        "target closed evidence is not strictly later/receive-available",
    )
    require(np.all((qualified == 0) | ((qualified_a > 0) & base & (identity > 0))), "qualified native mark lacks sealed target baseline")
    require(
        np.all((identity == 0) == (info["source_anchor_mid"] == 0)) and np.all((identity == 0) == (info["source_confirmed_mid"] == 0)), "native source mid/ID zero pairing invalid"
    )
    require(
        np.all((identity == 0) | (info["source_anchor_mid"] != info["source_confirmed_mid"])) and np.all(base == (info["target_baseline_mid"] > 0)),
        "native source displacement/baseline mid facts invalid",
    )
    require(np.all((identity > 0) | ((qualified_a == 0) & ~base & (evidence_a == 0) & (qualified == 0))), "hard-cleared native history retained ownership")
    numeric = np.column_stack([column(table, name) for name in NUMERIC_COLUMNS])
    require(np.all(np.isfinite(numeric) | np.isnan(numeric) | np.isneginf(numeric)), "native positive infinity numeric state invalid")
    sealed = (identity > 0) & (qualified_a > 0)
    require(~np.isfinite(numeric[~sealed]).any(), "unsealed/no native source mark must remain missing")
    require(np.all(np.isnan(numeric[~sealed]).all(axis=1) | np.isneginf(numeric[~sealed]).all(axis=1)), "unsealed native warmup/missing sentinels cannot be mixed")
    require(~np.isneginf(numeric[base]).any(), "a frozen CLOSED target baseline has completed native warmup")
    require(np.isfinite(numeric[sealed, 1]).all() and np.all((numeric[sealed, 1] > 0) & (numeric[sealed, 1] < 1)), "sealed historical peer work must be finite bounded fraction")
    require(np.all(numeric[sealed, 6] == (times[sealed] - peer_a[sealed]) / 300_000_000.0), "native receive age differs from source availability; no feature recalculation")
    peer_fresh = np.isin(categories["k_peer_phase"], tuple(FRESH_PHASES))
    target_fresh = np.isin(categories["k_target_phase"], tuple(FRESH_PHASES))
    require(
        np.isfinite(numeric[sealed & peer_fresh, 5]).all() and ~np.isfinite(numeric[sealed & ~peer_fresh, 5]).any(), "peer persistence freshness state disagrees with own-leg view"
    )
    target_view = sealed & base & target_fresh
    require(np.isfinite(numeric[target_view, 2]).all() and ~np.isfinite(numeric[sealed & ~target_view, 2]).any(), "target response freshness/baseline state disagrees")
    require(np.isfinite(numeric[qualified == 1, 0]).all() and ~np.isfinite(numeric[sealed & ~base, 0]).any(), "external impulse lacks frozen target unit")
    work_finite = np.isfinite(numeric[:, 3:5])
    require(np.all(work_finite[:, 0] == work_finite[:, 1]), "target known-work imbalance/strength must share availability")
    known_work = target_view & ~np.isin(categories["j_receive_relation"], ("invalid", "tied"))
    require(work_finite[known_work].all() and ~work_finite[sealed & ~known_work].any(), "target work freshness/tie/ambiguity semantics invalid")
    require(np.all(np.abs(numeric[work_finite[:, 0], 3]) <= 1) and np.all(numeric[work_finite[:, 1], 4] >= 0), "target work normalized bounds invalid")
    for index in (2, 3, 4, 5, 6):
        values = numeric[np.isfinite(numeric[:, index]), index]
        require(np.all(values[values == 0].view(np.uint64) == 0), "native numeric exact-zero identity invalid")
    fresh_qualified = (qualified == 1) & np.isfinite(numeric[:, (0, 1, 2, 5, 6)]).all(axis=1)
    require(
        np.all((support == 0) | (fresh_qualified & (evidence_r > peer_a) & (evidence_a <= times))),
        "native ordered response predicate lacks fresh qualified later CLOSED target evidence",
    )
    return {"times": times, "categories": categories, "info": info, "numeric": numeric, "fresh_qualified": fresh_qualified}


def evaluate_cell(day, symbol, periodic, parent, expected_keys, events, *, expected_rows=1381, expected_support_rows=461):
    require((day, symbol) in FIXED_CELLS, "unregistered fixed H15 cell")
    data, event = validate_table(periodic, periodic=True), validate_table(events, periodic=False)
    require(len(periodic) == len(parent) == len(expected_keys) == expected_rows, "original native1381 cohort changed")
    for name in KEYS:
        require(parent[name].type == pa.int64() and expected_keys[name].type == pa.int64(), "original native keys must remain int64")
        require(same_bits(periodic[name], parent[name]) and same_bits(periodic[name], expected_keys[name]), "original native keys changed; no invented origins")
    require(parent["OriginMidPrice"].type == pa.float64() and same_bits(periodic["OriginMidPrice"], parent["OriginMidPrice"]), "original raw mid uint64 bits changed")
    start, end = session_bounds(day)
    times = data["times"]
    require(times[0] == start and times[-1] == end and np.all(np.diff(times) == 10_000_000), "actual native10s boundary cohort changed")
    require(np.all((event["times"] >= start) & (event["times"] <= end)), "all_book escaped fixed native session")
    mask = times % 30_000_000 == 0
    require(int(mask.sum()) == expected_support_rows, "existing461 native30s origins changed")
    combined = sorted((int(time), priority, i, source) for priority, source in ((0, data), (1, event)) for i, time in enumerate(source["times"]))
    last_counts, marks, highwater = None, {}, 0
    last_hard, cleared = 0, False
    counter_names = tuple(name for name in INTEGER_INFO if name.startswith(("cumulative_", "session_")))
    for _, _, i, source in combined:
        info = source["info"]
        counts = tuple(int(info[name][i]) for name in counter_names)
        require(last_counts is None or all(now >= prior for now, prior in zip(counts, last_counts, strict=True)), "native counter/source-phase disagreement between writers")
        last_counts = counts
        identity, hard = int(info["source_mark_id"][i]), int(info["cumulative_hard_censor_count"][i])
        if not identity and highwater:
            require(hard > last_hard or cleared, "native source ownership erased without hard boundary")
            cleared = True
        if identity:
            require(identity >= highwater and (not cleared or identity > highwater), "native cleared source ID resurrected/regressed")
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
            prices = tuple(np.float64(info[name][i]).view(np.uint64).item() for name in PRICE_INFO)
            seal = int(info["qualified_mark_available_time"][i])
            qualify = int(info["mark_is_qualified"][i])
            previous = marks.setdefault(identity, {"fixed": fixed, "prices": prices, "seal": 0, "qualified": 0, "numeric": {}})
            require(previous["fixed"] == fixed and previous["prices"] == prices, "same sourceID changed frozen clocks/raw mid facts")
            require(previous["seal"] == 0 or (seal == previous["seal"] and qualify == previous["qualified"]), "same sourceID changed/undid closed qualification")
            if seal:
                previous["seal"], previous["qualified"] = seal, qualify
            for index in (0, 1):
                value = source["numeric"][i, index]
                if np.isfinite(value):
                    bits = np.float64(value).view(np.uint64).item()
                    require(index not in previous["numeric"] or previous["numeric"][index] == bits, "same sourceID changed frozen external impulse/peer work")
                    previous["numeric"][index] = bits
        last_hard = hard
    info = data["info"]
    names = ("session_qualified_mark_count", "session_up_qualified_mark_count", "session_down_qualified_mark_count")
    initial, terminal = ([int(info[name][index]) for name in names] for index in (0, -1))
    delta = [after - before for before, after in zip(initial, terminal, strict=True)]
    age = times - info["peer_source_available_time"]
    fresh = mask & data["fresh_qualified"] & (age >= 0) & (age <= 300_000_000)
    unique = {int(identity) for identity in info["source_mark_id"][fresh]}
    prior = {int(identity) for identity in info["source_mark_id"][fresh & (info["peer_source_available_time"] < start)]}
    return {
        "day": day,
        "symbol": symbol,
        "peer": PEERS[symbol],
        "status": "observed_native",
        "integrity_passed": True,
        "periodic_rows": len(periodic),
        "support_rows": int(mask.sum()),
        "event_book_rows": len(events),
        "session_counter_at_start": initial,
        "session_counter_at_end": terminal,
        "session_marks": delta[0],
        "session_up_marks": delta[1],
        "session_down_marks": delta[2],
        "distinct_recent_sampled_marks": len(unique),
        "distinct_prior_session_recent_marks": len(prior),
        "distinct_within_session_recent_marks": len(unique - prior),
        "ordered_response_support_origins": int((mask & (info["ordered_response_support"] == 1)).sum()),
        "qualified_fresh_pre_response_origins": int((mask & data["fresh_qualified"] & (info["latest_target_evidence_receive_time"] == 0)).sum()),
        "qualified_fresh_post_response_origins": int((mask & data["fresh_qualified"] & (info["latest_target_evidence_receive_time"] > info["peer_source_available_time"])).sum()),
        "category_counts_on_original30s": {name: dict(Counter(values[mask].tolist())) for name, values in data["categories"].items()},
        "diagnostic_first_last_counters": {name: [int(info[name][0]), int(info[name][-1])] for name in counter_names},
    }


def gates(cells):
    require([(c["day"], c["symbol"]) for c in cells] == list(FIXED_CELLS), "fixed chronological native cell identities required")
    require(all(c["status"] in ("observed_native", "missing_fixed_input_skip") for c in cells), "unknown fixed cell status")
    observed = [c for c in cells if c["status"] == "observed_native"]
    for cell in observed:
        require(type(cell["integrity_passed"]) is bool, "integrity must be boolean")
        require(
            all(
                type(cell[name]) is int and cell[name] >= 0
                for name in ("session_marks", "session_up_marks", "session_down_marks", "distinct_recent_sampled_marks", "ordered_response_support_origins")
            ),
            "support counts must be exact nonnegative integers",
        )
        require(cell["session_marks"] == cell["session_up_marks"] + cell["session_down_marks"], "session direction summary disagrees")
    checks = {
        "at_least_one_observed_fixed_cell": len(observed) >= 1,
        "every_present_cell_integrity": all(c["integrity_passed"] is True for c in observed),
        "every_present_cell_at_least5_session_marks": all(c["session_marks"] >= 5 for c in observed),
        "every_present_cell_at_least5_distinct_recent_sampled_marks": all(c["distinct_recent_sampled_marks"] >= 5 for c in observed),
        "aggregate_at_least5_up_marks": sum(c["session_up_marks"] for c in observed) >= 5,
        "aggregate_at_least5_down_marks": sum(c["session_down_marks"] for c in observed) >= 5,
        "aggregate_at_least20_ordered_response_support_origins": sum(c["ordered_response_support_origins"] for c in observed) >= 20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_cells": len(observed),
        "missing_cells": len(cells) - len(observed),
        "all_fixed_cells_observed": len(observed) == len(cells),
    }


def producer_context():
    path = OUTPUT / "frozen-method.yaml"
    producer = canonical_method(path, "h15-native-producer-method-v2")
    check_record(producer["preparation"])
    require(Path(producer["preparation"]["path"]).resolve() == OUTPUT / "preparation-receipt.yaml", "wrong V2 native preparation path")
    preparation = read_yaml(Path(producer["preparation"]["path"]))
    check_record(preparation["profile"])
    require(Path(preparation["profile"]["path"]).resolve() == PROFILE, "wrong V2 native profile path")
    profile = read_yaml(PROFILE)
    require(producer["preparation_identity"] == digest(preparation) and producer["profile_identity"] == digest(profile), "producer preparation/profile differs from freeze")
    require(profile["contract"] == CONTRACT and all(type(n) is int for n in profile["contract"].values()), "typed producer rules changed")
    require(
        profile["alpha_fields"] == list(ALPHA) and profile["info_fields"] == list(INFO) and profile["carrier_order"] == list(CARRIERS),
        "native producer fields/source order changed",
    )
    for nodes in (producer["sources"], producer["inputs"]):
        for node in nodes:
            absent = node["path"] in profile["required_absent_inputs"]
            checked = check_record(node, allow_missing=absent)
            require(not absent or checked is None, "previously absent producer input appeared")
    return producer, preparation, profile, [record(path), record(path.with_suffix(".yaml.identity")), producer["preparation"], preparation["profile"]]


def bind_profile():
    require(not EVALUATION_PROFILE.exists(), "fresh prospective support profile required")
    producer, preparation, profile, links = producer_context()
    jobs = {j["day"]: j for j in preparation["jobs"]}
    cells, statuses = [], []
    for day in DAYS:
        job = jobs[day]
        if job["requires_replay"]:
            status_path = Path(job["work"]) / "status" / f"{day}.yaml"
            status = read_yaml(status_path)
            require(status.get("status") == "completed" and status.get("fatal_error") is False, "native daily producer not cleanly completed")
            statuses.append(record(status_path))
        for symbol in TARGETS:
            cell = {"day": day, "symbol": symbol, "status": "observed_native" if symbol in job["symbols"] else "missing_fixed_input_skip"}
            if cell["status"] == "observed_native":
                cell.update({folder: record(Path(job["work"]) / folder / day / symbol / "values.parquet") for folder in ("data", "events")})
            cells.append(cell)
    control = preparation["registration_control"]
    status_path = Path(control["work"]) / "status/20260119.yaml"
    status = read_yaml(status_path)
    require(status.get("status") == "completed" and status.get("fatal_error") is False, "registration control not cleanly completed")
    statuses.append(record(status_path))
    body = {
        "schema": "h15-native-support-evaluation-profile-v2",
        "status": "prospective_requires_root_freeze",
        "evaluation_contract": EVALUATION_CONTRACT,
        "producer_identity": producer["identity"],
        "producer_links": links,
        "status_receipts": statuses,
        "cells": cells,
        "parents": profile["parents"],
        "registration_control": [
            {"original": original, "produced": record(Path(path))} for original, path in zip(control["required_byte_equal_outputs"], control["outputs"], strict=True)
        ],
        "source_proof": [record(EVALUATION_PLAN), record(TEST_SOURCE)],
        "labels_read": False,
        "model_fits": 0,
    }
    write_yaml(EVALUATION_PROFILE, body)
    return {"bound_metadata_only": True, "profile": str(EVALUATION_PROFILE), "profile_identity": digest(body), "native_outputs_decoded": 0}


def validate_evaluation_freeze(path):
    method = canonical_method(path, "h15-native-support-evaluation-method-v2")
    check_record(method["profile"])
    require(Path(method["profile"]["path"]).resolve() == EVALUATION_PROFILE, "wrong support evaluation profile path")
    profile = read_yaml(EVALUATION_PROFILE)
    require(
        profile.get("schema") == "h15-native-support-evaluation-profile-v2"
        and method["profile_identity"] == digest(profile)
        and digest(profile["evaluation_contract"]) == digest(EVALUATION_CONTRACT),
        "canonical support profile/rules changed",
    )
    require(all(type(n) is int for n in profile["evaluation_contract"]["typed_support_rules"].values()), "integer support gates cannot be boolean aliases")
    sources = {n["path"]: n for n in method["sources"]}
    essential = (Path(__file__), Path(__file__).with_name("prepare_native.py"), ROOT / "AstraResearch/astra/io.py", EVALUATION_PLAN, TEST_SOURCE)
    require(all(str(p.resolve()) in sources for p in essential), "essential evaluator/helper/test/plan source omitted")
    require(profile["source_proof"] == [sources[str(EVALUATION_PLAN)], sources[str(TEST_SOURCE)]], "evaluation source proof missing/changed")
    producer, preparation, producer_profile, links = producer_context()
    require(profile["producer_identity"] == producer["identity"] and profile["parents"] == producer_profile["parents"], "producer/original origin identities changed")
    require(all(sources.get(n["path"], {}).get("sha256") == n["sha256"] for n in producer["sources"]), "immutable producer source closure omitted")
    require([(c["day"], c["symbol"]) for c in profile["cells"]] == list(FIXED_CELLS), "fixed4 native cell identities changed")
    jobs = {j["day"]: j for j in preparation["jobs"]}
    for cell in profile["cells"]:
        job = jobs[cell["day"]]
        present = cell["symbol"] in job["symbols"]
        require(cell["status"] == ("observed_native" if present else "missing_fixed_input_skip"), "configured native writer cannot be silently skipped")
        if present:
            for folder in ("data", "events"):
                require(cell[folder]["path"] == str(Path(job["work"]) / folder / cell["day"] / cell["symbol"] / "values.parquet"), "wrong configured native artifact path")
    control = preparation["registration_control"]
    require(len(profile["registration_control"]) == 4, "all4 registration-control Parquets required")
    for pair, original, produced in zip(profile["registration_control"], control["required_byte_equal_outputs"], control["outputs"], strict=True):
        require(pair["original"] == original and pair["produced"]["path"] == produced, "whole-byte control source/output identity changed")
    inputs = unique_records(
        [
            *producer["inputs"],
            *links,
            *profile["status_receipts"],
            *[c[folder] for c in profile["cells"] if c["status"] == "observed_native" for folder in ("data", "events")],
            *[p[role] for p in profile["registration_control"] for role in ("original", "produced")],
            record(EVALUATION_PROFILE),
        ]
    )
    bound = {n["path"]: n for n in method["inputs"]}
    require(all(bound.get(n["path"], {}).get("sha256") == n["sha256"] for n in inputs), "required producer/output/input identities omitted")
    for nodes in (method["sources"], method["inputs"]):
        for node in nodes:
            absent = node["path"] in producer_profile["required_absent_inputs"]
            checked = check_record(node, allow_missing=absent)
            require(not absent or checked is None, "previously absent fixed input appeared")
    return method, profile


def evaluate(method_path, destination):
    require(not destination.exists(), "fresh support receipt required; never overwrite")
    method, profile = validate_evaluation_freeze(method_path)
    for pair in profile["registration_control"]:
        require(pair["original"]["sha256"] == pair["produced"]["sha256"], "H13 registration control entire Parquet bytes differ")
    parents = {(c["day"], c["symbol"]): c for c in profile["parents"]}
    cells = []
    for cell in profile["cells"]:
        if cell["status"] == "missing_fixed_input_skip":
            cells.append(cell)
            continue
        for folder in ("data", "events"):
            require(set(pq.read_schema(cell[folder]["path"]).names) == set(REQUIRED_COLUMNS), "undeclared native output columns/labels")
        data, events = (pq.read_table(cell[folder]["path"], columns=list(REQUIRED_COLUMNS), use_threads=False) for folder in ("data", "events"))
        parent = parents[(cell["day"], cell["symbol"])]
        original = pq.read_table(parent["parent_origins"]["path"], columns=[*KEYS, "OriginMidPrice"], use_threads=False)
        keys = pq.read_table(parent["expected_native_keys"]["path"], columns=list(KEYS), use_threads=False)
        cells.append(evaluate_cell(cell["day"], cell["symbol"], data, original, keys, events))
    validate_evaluation_freeze(method_path)
    result = {
        "schema": "h15-native-source-support-validation-v2",
        "method_identity": method["identity"],
        "profile_identity": digest(profile),
        "method": record(method_path),
        "method_anchor": record(method_path.with_suffix(method_path.suffix + ".identity")),
        "sources": method["sources"],
        "inputs": method["inputs"],
        "producer_identity": profile["producer_identity"],
        "native_output_artifacts": [c[folder] for c in profile["cells"] if c["status"] == "observed_native" for folder in ("data", "events")],
        "registration_control": [{**pair, "entire_parquet_bytes_equal": True} for pair in profile["registration_control"]],
        "cells": cells,
        "support_gate": gates(cells),
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
    return {"integrity_passed": True, "support_passed": result["support_gate"]["passed"], "receipt": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind-profile", action="store_true")
    mode.add_argument("--evaluate", type=Path, metavar="FROZEN_EVALUATION_METHOD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.evaluate and args.output is None:
        parser.error("--evaluate requires --output")
    print(bind_profile() if args.bind_profile else evaluate(args.evaluate.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
