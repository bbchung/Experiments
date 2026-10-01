"""Validate native H13 artifacts and its frozen source-support rule only.

No feature calculation, raw stream, label, replay or learner is used. Root
first binds artifact bytes with --bind-profile, independently freezes this
evaluation method/profile/source/input closure, then uses --evaluate.
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
from prepare_native import ALPHA, CONTRACT, DAYS, INFO, OUTPUT, PROFILE, ROOT, TARGETS, check_record, record, unique_records

KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")
RAW = ("OriginMidPrice", "OriginMidTicks", "OriginBidTicks", "OriginAskTicks")
ALPHA_COLUMNS = tuple(f"QuoteVisitAcceptance.0.{name}.0" for name in ALPHA)
INFO_COLUMNS = tuple(f"H13_{name}" for name in INFO)
REQUIRED_COLUMNS = (*KEYS, *ALPHA_COLUMNS, *RAW, *INFO_COLUMNS)
PHASES = {"warmup", "inactive", "provisional", "bid_tested", "ask_tested", "bilateral", "untestable", "censored", "stale"}
CONTRAST_PHASES = {"provisional", "bid_tested", "ask_tested"}
FIXED_CELLS = tuple((day, symbol) for day in DAYS for symbol in TARGETS)
PRODUCER_IDENTITY = "3591aa72a90bbdcbaa5c6b17cd426fa6415b0d686ed7d946fe4a0fe757ddb8da"
EVALUATION_PROFILE = OUTPUT / "support-evaluation-profile.yaml"
EVALUATION_PLAN = OUTPUT / "SUPPORT_EVALUATION_PLAN.md"
TEST_SOURCE = OUTPUT / "test_evaluate_native_support.py"
EVALUATION_CONTRACT = {
    "schema": "h13-native-support-evaluation-contract-v1",
    "typed_support_rules": CONTRACT,
    "completion_session": "[09:10,13:00); actual native periodic start/end counter delta before R>=S callbacks",
    "origin_support": "Existing461 S%30s==0 rows including13:00; finite native landmark with0<S-A<=300s; pre09:10 landmarks may count unique",
    "clock_domains": "A<=S; positive E independently ordered within native source epochs; no E<=A/S requirement",
    "event_cadence": "all_book may skip trade-closed counter jumps; retain duplicate receives with exact source sequence; never require everyID",
    "missing": "Producer-declared missing fixed input/metadata skips only; configured present writer missing/changed output rejects integrity",
    "no_predictive_admission": True,
}


def session_bounds(day):
    elapsed_days = (date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}") - date(1970, 1, 1)).days
    midnight = (elapsed_days * 86400 - 8 * 3600) * 1_000_000
    return midnight + (9 * 3600 + 10 * 60) * 1_000_000, midnight + 13 * 3600 * 1_000_000


def column(table, name):
    return table[name].to_numpy(zero_copy_only=False)


def same_bits(left, right):
    if left.type != right.type or left.null_count or right.null_count or len(left) != len(right):
        return False
    if left.type == pa.float64():
        return np.array_equal(left.to_numpy().view(np.uint64), right.to_numpy().view(np.uint64))
    return left.equals(right)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def validate_table(table, *, periodic):
    require(
        len(table.column_names) == len(REQUIRED_COLUMNS) and set(table.column_names) == set(REQUIRED_COLUMNS),
        "exact five-Alpha/fifteen-Info native schema required; no labels/extra model fields",
    )
    for name in REQUIRED_COLUMNS:
        require(table[name].null_count == 0, f"native null unsupported: {name}")
        expected_type = pa.int64() if name in KEYS else pa.string() if name == ALPHA_COLUMNS[0] else pa.float64()
        require(table[name].type == expected_type, f"native exact dtype changed: {name}")
        role = (table.schema.field(name).metadata or {}).get(b"coco.role")
        expected_role = b"time" if name == "SampleTime" else b"context" if name in KEYS else b"feature" if name in ALPHA_COLUMNS else b"metadata"
        require(role == expected_role, f"native field role changed/Info entered model exports: {name}")
    times = column(table, "SampleTime")
    books = column(table, "SampleBookTime")
    sequences = column(table, "SampleBookSeq")
    require(np.all((times > 0) & (times < 2**53) & (books >= 0) & (books <= times) & (sequences >= 0) & (sequences < 2**53)), "native source-key integer boundaries invalid")
    require(np.all(np.diff(times) > 0) if periodic else np.all(np.diff(times) >= 0), "native source-key sample order invalid")
    if not periodic:
        require(np.all(books == times) and np.all(np.diff(sequences) > 0), "all_book source sequence/time contract invalid; never deduplicate ties")
    phase = np.asarray(table[ALPHA_COLUMNS[0]].to_pylist())
    require(set(phase).issubset(PHASES), "unknown native phase literal")
    info = {name: column(table, f"H13_{name}") for name in INFO}
    for name, values in info.items():
        require(np.all(np.isfinite(values) & (values >= 0) & (values < 2**53) & (values == np.floor(values))), f"native Info exact integer boundary invalid: {name}")
        require(np.all(values[values == 0].view(np.uint64) == 0), f"native Info zero identity invalid: {name}")
        info[name] = values.astype(np.int64)
    available = info["processed_cluster_available_time"]
    exchange = info["processed_cluster_exchange_time"]
    require(np.all(available <= times) and np.all((available == 0) == (exchange == 0)), "processed receive availability/zero pairing invalid")
    require(np.all(np.diff(available) >= 0), "processed receive availability regressed")
    counters = np.column_stack([info[name] for name in INFO if name.startswith("cumulative_")])
    require(np.all(np.diff(counters, axis=0) >= 0), "native cumulative counter regressed")
    total = info["cumulative_completed_landmark_count"]
    up = info["cumulative_up_completed_landmark_count"]
    down = info["cumulative_down_completed_landmark_count"]
    quotes = info["cumulative_quote_led_visit_count"]
    landmark = info["completed_landmark_id"]
    current = info["current_visit_id"]
    mark_available = info["completed_landmark_available_time"]
    mark_exchange = info["completed_landmark_exchange_time"]
    require(np.all(total == up + down) and np.all(total <= quotes), "native completion direction/visit counts disagree")
    require(
        np.all((landmark <= quotes) & (current <= quotes)) and np.all((landmark == 0) | ((landmark >= total) & (total > 0))),
        "native landmark/visit IDs inconsistent with completion/formation counters",
    )
    require(np.all((landmark == 0) == (mark_available == 0)) and np.all((landmark == 0) == (mark_exchange == 0)), "native landmark clock/ID zero pairing invalid")
    require(np.all(mark_available <= available) and np.all(mark_available <= times), "native completed landmark not receive-available at origin")
    numeric = np.column_stack([column(table, name) for name in ALPHA_COLUMNS[1:]])
    warmup = phase == "warmup"
    hidden = np.isin(phase, ("stale", "censored"))
    exposed = ~warmup & ~hidden & (landmark > 0)
    require(np.isneginf(numeric[warmup]).all() and np.all(landmark[warmup] == 0), "warmup must retain negative infinity and no completed landmark")
    require(np.isnan(numeric[hidden]).all(), "stale/censored Alphas must be NaN; native counters remain readable")
    require(np.isnan(numeric[~warmup & ~hidden & (landmark == 0)]).all(), "no native historical landmark must remain missing")
    require(np.isfinite(numeric[exposed]).all(), "exposed native historical landmark Alphas must be finite")
    require(np.isin(numeric[exposed, 0], (-1.0, 1.0)).all(), "native completed direction must be +/-1")
    require(np.all((numeric[exposed, 2] > 0) & (numeric[exposed, 2] <= 1)), "native frozen bilateral strength outside physical bounded fraction")
    require(np.all(numeric[exposed, 3] == (times[exposed] - mark_available[exposed]) / 300_000_000.0), "native age does not match receive availability; no feature replacement")
    require(np.all(numeric[exposed, 1][numeric[exposed, 1] == 0].view(np.uint64) == 0), "native accepted-anchor exact zero must be positive zero")
    require(np.all(numeric[exposed, 3][numeric[exposed, 3] == 0].view(np.uint64) == 0), "native receive-age exact zero must be positive zero")
    require(np.all(current[np.isin(phase, tuple(CONTRAST_PHASES))] > 0), "provisional/unilateral phase lacks native visit identity")
    bilateral = phase == "bilateral"
    require(np.all((landmark[bilateral] > 0) & (current[bilateral] == landmark[bilateral])), "bilateral phase does not own its completed native mark")
    require(np.all((landmark[phase == "censored"] == 0) & (current[phase == "censored"] == 0)), "hard-censored epoch retained native ownership")
    return {"times": times, "phase": phase, "info": info, "numeric": numeric, "exposed": exposed}


def evaluate_cell(day, symbol, periodic, parent, expected_keys, events, *, expected_rows=1381, expected_support_rows=461):
    require((day, symbol) in FIXED_CELLS, "unregistered fixed native cell")
    data = validate_table(periodic, periodic=True)
    event = validate_table(events, periodic=False)
    require(len(periodic) == len(parent) == len(expected_keys) == expected_rows, "original native1381 cohort changed")
    for name in KEYS:
        require(parent[name].type == pa.int64() and expected_keys[name].type == pa.int64(), "original native keys must be int64")
        require(same_bits(periodic[name], parent[name]) and same_bits(periodic[name], expected_keys[name]), "original native keys changed; no invented grid")
    require(parent["OriginMidPrice"].type == pa.float64() and same_bits(periodic["OriginMidPrice"], parent["OriginMidPrice"]), "original native mid uint64 bits changed")
    start, end = session_bounds(day)
    times = data["times"]
    require(times[0] == start and times[-1] == end and np.all(np.diff(times) == 10_000_000), "native periodic start/end10s contract changed")
    require(np.all((event["times"] >= start) & (event["times"] <= end)), "all_book writer escaped fixed native session window")
    mask = times % 30_000_000 == 0
    require(int(mask.sum()) == expected_support_rows, "original native461 support origins changed")
    # Sort native observations in real source phase: periodic(S) precedes ALL
    # callbacks at receive==S, then all_book rows retain their source sequence.
    combined = []
    for priority, table_data in ((0, data), (1, event)):
        for index, time in enumerate(table_data["times"]):
            combined.append((int(time), priority, index, table_data))
    combined.sort(key=lambda row: row[:3])
    last_counts = None
    landmarks = {}
    greatest_landmark = 0
    history_cleared = False
    for _, _, index, item in combined:
        info = item["info"]
        counts = tuple(int(info[name][index]) for name in INFO if name.startswith("cumulative_"))
        require(
            last_counts is None or all(now >= then for now, then in zip(counts, last_counts, strict=True)),
            "native counter/source-phase disagreement between event and periodic writers",
        )
        last_counts = counts
        identity = int(info["completed_landmark_id"][index])
        if identity == 0 and greatest_landmark > 0:
            history_cleared = True
        if identity:
            require(
                identity >= greatest_landmark and (not history_cleared or identity > greatest_landmark), "native hard-cleared/completed landmark identity resurrected or regressed"
            )
            greatest_landmark = identity
            history_cleared = False
            fixed = tuple(
                int(info[name][index])
                for name in (
                    "completed_landmark_available_time",
                    "completed_landmark_exchange_time",
                    "cumulative_completed_landmark_count",
                    "cumulative_up_completed_landmark_count",
                    "cumulative_down_completed_landmark_count",
                )
            )
            if identity in landmarks:
                require(landmarks[identity][0] == fixed, "same native completed landmark changed clocks/counters")
            else:
                landmarks[identity] = (fixed, None)
            if item["exposed"][index]:
                meaning = tuple(item["numeric"][index, (0, 2)].view(np.uint64).tolist())
                if landmarks[identity][1] is not None:
                    require(landmarks[identity][1] == meaning, "native completed direction/strength changed after first completion")
                landmarks[identity] = (fixed, meaning)
    info = data["info"]
    completion_names = ("cumulative_completed_landmark_count", "cumulative_up_completed_landmark_count", "cumulative_down_completed_landmark_count")
    initial = [int(info[name][0]) for name in completion_names]
    terminal = [int(info[name][-1]) for name in completion_names]
    delta = [right - left for left, right in zip(initial, terminal, strict=True)]
    ages = times - info["completed_landmark_available_time"]
    fresh = mask & data["exposed"] & (ages > 0) & (ages <= 300_000_000)
    unique = {int(identity) for identity in info["completed_landmark_id"][fresh]}
    prior = {int(identity) for identity in info["completed_landmark_id"][fresh & (info["completed_landmark_available_time"] < start)]}
    return {
        "day": day,
        "symbol": symbol,
        "status": "observed_native",
        "integrity_passed": True,
        "periodic_rows": len(periodic),
        "support_rows": int(mask.sum()),
        "event_book_rows": len(events),
        "session_counter_at_start": initial,
        "session_counter_at_end": terminal,
        "session_completions": delta[0],
        "session_up_completions": delta[1],
        "session_down_completions": delta[2],
        "distinct_recent_sampled_landmarks": len(unique),
        "distinct_prior_session_recent_landmarks": len(prior),
        "distinct_within_session_recent_landmarks": len(unique - prior),
        "provisional_or_unilateral_origins": int((mask & np.isin(data["phase"], tuple(CONTRAST_PHASES))).sum()),
        "phase_counts_on_original30s": dict(Counter(data["phase"][mask].tolist())),
        "diagnostic_first_last_counters": {name: [int(info[name][0]), int(info[name][-1])] for name in INFO if name.startswith("cumulative_")},
    }


def gates(cells):
    require([(cell["day"], cell["symbol"]) for cell in cells] == list(FIXED_CELLS), "fixed chronological cell identities required")
    require(all(cell["status"] in ("observed_native", "missing_fixed_input_skip") for cell in cells), "unknown native cell status")
    observed = [cell for cell in cells if cell["status"] == "observed_native"]
    count_names = ("session_completions", "session_up_completions", "session_down_completions", "distinct_recent_sampled_landmarks", "provisional_or_unilateral_origins")
    for cell in observed:
        require(type(cell["integrity_passed"]) is bool, "native integrity result must be boolean")
        require(all(type(cell[name]) is int and cell[name] >= 0 for name in count_names), "native support counts must be exact nonnegative integers")
        require(cell["session_completions"] == cell["session_up_completions"] + cell["session_down_completions"], "native summary direction counts disagree")
    checks = {
        "at_least_one_observed_fixed_cell": len(observed) >= 1,
        "every_present_cell_integrity": all(cell["integrity_passed"] is True for cell in observed),
        "every_present_cell_at_least5_session_completions": all(cell["session_completions"] >= 5 for cell in observed),
        "every_present_cell_at_least5_distinct_recent_sampled_landmarks": all(cell["distinct_recent_sampled_landmarks"] >= 5 for cell in observed),
        "aggregate_at_least5_up_completions": sum(cell["session_up_completions"] for cell in observed) >= 5,
        "aggregate_at_least5_down_completions": sum(cell["session_down_completions"] for cell in observed) >= 5,
        "aggregate_at_least20_provisional_or_unilateral_origins": sum(cell["provisional_or_unilateral_origins"] for cell in observed) >= 20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed_cells": len(observed),
        "missing_cells": len(cells) - len(observed),
        "all_fixed_cells_observed": len(observed) == len(FIXED_CELLS),
    }


def canonical_method(path, schema):
    body = read_yaml(path)
    anchor_path = path.with_suffix(path.suffix + ".identity")
    anchor = anchor_path.read_text().strip()
    require(body.get("schema") == schema and body.get("status") == "frozen" and body.get("identity") == anchor, "canonical frozen method/identity sidecar required")
    require(digest({k: v for k, v in body.items() if k != "identity"}) == anchor, "canonical frozen method body identity mismatch")
    return body


def producer_context():
    method_path = OUTPUT / "frozen-method.yaml"
    producer = canonical_method(method_path, "h13-native-producer-method-v1")
    require(producer["identity"] == PRODUCER_IDENTITY, "different native producer study; new version required")
    check_record(producer["preparation"])
    preparation = read_yaml(Path(producer["preparation"]["path"]))
    check_record(preparation["profile"])
    profile = read_yaml(PROFILE)
    require(producer["preparation_identity"] == digest(preparation) and producer["profile_identity"] == digest(profile), "native preparation/profile differs from producer freeze")
    require(profile["contract"] == CONTRACT and all(type(value) is int for value in profile["contract"].values()), "typed producer sampling/support contract changed")
    for nodes in (producer["sources"], producer["inputs"]):
        for node in nodes:
            absent = node["path"] in profile["required_absent_inputs"]
            require(check_record(node, allow_missing=absent) is None if absent else check_record(node) is not None, "previously absent producer input appeared")
    return producer, preparation, profile, [record(method_path), record(method_path.with_suffix(".yaml.identity")), record(Path(producer["preparation"]["path"])), record(PROFILE)]


def bind_profile():
    require(not EVALUATION_PROFILE.exists(), "fresh prospective native evaluation profile required")
    _, preparation, profile, links = producer_context()
    cells, statuses = [], []
    jobs = {job["day"]: job for job in preparation["jobs"]}
    for day in DAYS:
        status_path = Path(jobs[day]["work"]) / "status" / (day + ".yaml")
        status = read_yaml(status_path)
        require(status.get("status") == "completed" and status.get("fatal_error") is False, "native day producer must finish cleanly before evaluation binding")
        statuses.append(record(status_path))
        for symbol in TARGETS:
            if symbol not in jobs[day]["symbols"]:
                cells.append({"day": day, "symbol": symbol, "status": "missing_fixed_input_skip"})
            else:
                cells.append(
                    {
                        "day": day,
                        "symbol": symbol,
                        "status": "observed_native",
                        **{folder: record(Path(jobs[day]["work"]) / folder / day / symbol / "values.parquet") for folder in ("data", "events")},
                    }
                )
    control = preparation["relocation_control"]
    control_status = Path(control["work"]) / "status/20260119.yaml"
    status = read_yaml(control_status)
    require(status.get("status") == "completed" and status.get("fatal_error") is False, "native relocation control must finish cleanly")
    statuses.append(record(control_status))
    body = {
        "schema": "h13-native-support-evaluation-profile-v1",
        "status": "prospective_requires_root_freeze",
        "evaluation_contract": EVALUATION_CONTRACT,
        "producer_identity": PRODUCER_IDENTITY,
        "producer_links": links,
        "status_receipts": statuses,
        "cells": cells,
        "parents": profile["parents"],
        "relocation_control": [
            {"original": original, "produced": record(Path(path))} for original, path in zip(control["required_byte_equal_outputs"], control["outputs"], strict=True)
        ],
        "source_proof": [record(EVALUATION_PLAN), record(TEST_SOURCE)],
        "labels_read": False,
        "model_fits": 0,
    }
    write_yaml(EVALUATION_PROFILE, body)
    return {"bound_metadata_only": True, "profile": str(EVALUATION_PROFILE), "profile_identity": digest(body), "native_outputs_decoded": 0}


def validate_evaluation_freeze(method_path):
    method = canonical_method(method_path, "h13-native-support-evaluation-method-v1")
    check_record(method["profile"])
    require(Path(method["profile"]["path"]).resolve() == EVALUATION_PROFILE, "different evaluation profile path")
    profile = read_yaml(EVALUATION_PROFILE)
    require(method["profile_identity"] == digest(profile) and profile["evaluation_contract"] == EVALUATION_CONTRACT, "fixed evaluation profile/rules changed")
    require(all(type(n) is int for n in profile["evaluation_contract"]["typed_support_rules"].values()), "evaluation integer gates cannot use boolean aliases")
    required_sources = [Path(__file__), Path(__file__).with_name("prepare_native.py"), ROOT / "AstraResearch/astra/io.py", EVALUATION_PLAN, TEST_SOURCE]
    sources = {node["path"]: node for node in method["sources"]}
    require(all(str(path.resolve()) in sources for path in required_sources), "required running evaluator/test/plan source closure omitted")
    require(profile["source_proof"] == [sources[str(EVALUATION_PLAN)], sources[str(TEST_SOURCE)]], "essential evaluation source proof omitted/changed")
    producer, preparation, producer_profile, producer_links = producer_context()
    require(all(sources.get(node["path"], {}).get("sha256") == node["sha256"] for node in producer["sources"]), "frozen evaluation omitted immutable producer source closure")
    require(profile["parents"] == producer_profile["parents"], "original native parent/key identities changed")
    declared = {job["day"]: job for job in preparation["jobs"]}
    for cell in profile["cells"]:
        day, symbol = cell["day"], cell["symbol"]
        present = symbol in declared[day]["symbols"]
        require(cell["status"] == ("observed_native" if present else "missing_fixed_input_skip"), "configured native cell cannot be silently skipped")
        if present:
            for folder in ("data", "events"):
                require(cell[folder]["path"] == str(Path(declared[day]["work"]) / folder / day / symbol / "values.parquet"), "different native writer artifact bound")
    control = preparation["relocation_control"]
    require(len(profile["relocation_control"]) == 2, "both complete relocation-control native Parquets required")
    for pair, original, produced in zip(profile["relocation_control"], control["required_byte_equal_outputs"], control["outputs"], strict=True):
        require(pair["original"] == original and pair["produced"]["path"] == produced, "relocation-control source/output identity changed")
    inputs = unique_records(
        [
            *producer["inputs"],
            *producer_links,
            *profile["status_receipts"],
            *[cell[role] for cell in profile["cells"] if cell["status"] == "observed_native" for role in ("data", "events")],
            *[pair[role] for pair in profile["relocation_control"] for role in ("original", "produced")],
            record(EVALUATION_PROFILE),
        ]
    )
    bound = {node["path"]: node for node in method["inputs"]}
    require(all(bound.get(node["path"], {}).get("sha256") == node["sha256"] for node in inputs), "frozen evaluation omitted producer/native output/input identities")
    for nodes in (method["sources"], method["inputs"]):
        for node in nodes:
            absent = node["path"] in producer_profile["required_absent_inputs"]
            path = check_record(node, allow_missing=absent)
            require(not absent or path is None, "previously absent fixed native input appeared")
    require([(cell["day"], cell["symbol"]) for cell in profile["cells"]] == list(FIXED_CELLS), "fixed four native cells changed")
    require(profile["producer_identity"] == PRODUCER_IDENTITY, "wrong native producer identity")
    return method, profile, preparation


def evaluate(method_path, destination):
    require(not destination.exists(), "fresh evaluation receipt required; never overwrite")
    frozen, profile, _ = validate_evaluation_freeze(method_path)
    controls = []
    for pair in profile["relocation_control"]:
        equal = pair["original"]["sha256"] == pair["produced"]["sha256"]
        require(equal, "H10 relocation control entire native Parquet bytes differ")
        controls.append({**pair, "entire_parquet_bytes_equal": True})
    parents = {(cell["day"], cell["symbol"]): cell for cell in profile["parents"]}
    cells = []
    for cell in profile["cells"]:
        if cell["status"] == "missing_fixed_input_skip":
            cells.append(cell)
            continue
        require(cell["status"] == "observed_native", "unsupported native cell disposition")
        parent = parents[(cell["day"], cell["symbol"])]
        # Inspect whole schema first, so selecting declared columns cannot hide
        # a regenerated label or an accidentally exported diagnostic feature.
        for folder in ("data", "events"):
            require(set(pq.read_schema(cell[folder]["path"]).names) == set(REQUIRED_COLUMNS), "undeclared native output columns/labels")
        tables = [pq.read_table(Path(cell[folder]["path"]), columns=list(REQUIRED_COLUMNS), use_threads=False) for folder in ("data", "events")]
        original = pq.read_table(parent["parent_origins"]["path"], columns=[*KEYS, "OriginMidPrice"], use_threads=False)
        expected = pq.read_table(parent["expected_native_keys"]["path"], columns=list(KEYS), use_threads=False)
        cells.append(evaluate_cell(cell["day"], cell["symbol"], tables[0], original, expected, tables[1]))
    validate_evaluation_freeze(method_path)
    result = {
        "schema": "h13-native-source-support-validation-v1",
        "method_identity": frozen["identity"],
        "profile_identity": digest(profile),
        "method": record(method_path),
        "method_anchor": record(method_path.with_suffix(method_path.suffix + ".identity")),
        "sources": frozen["sources"],
        "inputs": frozen["inputs"],
        "producer_identity": PRODUCER_IDENTITY,
        "native_output_artifacts": [cell[folder] for cell in profile["cells"] if cell["status"] == "observed_native" for folder in ("data", "events")],
        "relocation_control": controls,
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
    mode.add_argument("--bind-profile", action="store_true", help="Hash native output artifacts/status metadata only; never decode any values")
    mode.add_argument("--evaluate", type=Path, metavar="FROZEN_EVALUATION_METHOD")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.evaluate and args.output is None:
        parser.error("--evaluate requires --output")
    print(bind_profile() if args.bind_profile else evaluate(args.evaluate.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
