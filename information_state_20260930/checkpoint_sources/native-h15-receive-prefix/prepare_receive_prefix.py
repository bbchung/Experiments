"""Prospective real H15 receive-prefix selection and exact native validation.

Root alone runs bind/prepare/check stages, freezes both methods, and replays the
recorded command. Imports and synthetic tests never inspect research data.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import ctypes as ct
import hashlib
import importlib.util
import sys
from array import array
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]
PARENT = OUTPUT.parent / "native-h15-peer-trade-v2"
PROFILE = OUTPUT / "selection-profile.yaml"
PLAN = OUTPUT / "PREFIX_PLAN.md"
PRODUCER_IDENTITY = "00540736c9defcabf2eb9e577bee50840dd3eedd62e00b3fdd981572118a8b61"
SUPPORT_IDENTITY = "0430f280dc5f18d71050507e878952c6e46cb11360006d59a5b0d8882d8614a5"
DAY = "20260119"
KEYS = ("SampleTime", "SampleBookTime", "SampleBookSeq")
CONTRACT = {
    "schema": "h15-real-receive-prefix-selection-profile-v1",
    "day": DAY,
    "target_order": ["2308", "2317"],
    "leg_order": ["peer", "target"],
    "origin_seconds": 30,
    "origin_order": "ascending original native SampleTime within each fixed target/leg",
    "pending": "continuous positive own-E group containing a book; last R<S; first actual increasing continuous next-distinct own-E A>S; no EOF closure",
    "state": "qualified historical native mark, own closed phase quote/flow, positive own processedE<pendingE and processedA<=R; no E<=R/S assumption",
    "carrier": "earliest other-symbol real loader-effective receive D in[S,A), ties by receive/rawseq/T-before-B/symbol/sourcepath",
    "source_policy": "all16 original fixed BIN streams; byte-identical full warm-in prefix effectiveR<=D; no fallback/synthesis/replacement",
    "max_uncompressed_spool_bytes": 536870912,
    "validation": "exact complete original periodic/event key populations throughD; all native fields including11Alpha27Info/rawmid; S mandatory; no EOF origin>D",
    "models_permitted": 0,
    "labels_permitted": False,
}
spec = importlib.util.spec_from_file_location("h15_prefix_parent", PARENT / "prepare_native.py")
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)
require, record, check_record = producer.require, producer.record, producer.check_record
canonical_method, unique_records = producer.canonical_method, producer.unique_records
digest, read_yaml, write_yaml = producer.digest, producer.read_yaml, producer.write_yaml


class Header(ct.Structure):
    _fields_ = [("seq", ct.c_int64), ("time", ct.c_int64), ("type", ct.c_char)]


class Book(ct.Structure):
    _fields_ = [
        ("exchange", ct.c_int64),
        ("bid", ct.c_double * 5),
        ("ask", ct.c_double * 5),
        ("bid_qty", ct.c_int32 * 5),
        ("ask_qty", ct.c_int32 * 5),
        ("status", ct.c_int32),
        ("symbol", ct.c_char * 16),
        ("bid_depth", ct.c_uint8),
        ("ask_depth", ct.c_uint8),
    ]


class Trade(ct.Structure):
    _fields_ = [
        ("exchange", ct.c_int64),
        ("price", ct.c_double),
        ("turnover", ct.c_double),
        ("quantity", ct.c_int32),
        ("total_quantity", ct.c_int32),
        ("status", ct.c_int32),
        ("symbol", ct.c_char * 16),
        ("side", ct.c_uint8),
    ]


def abi():
    expected = {
        "byte_order": "little",
        "header_bytes": 24,
        "book_bytes": 152,
        "trade_bytes": 56,
        "header_time_offset": 8,
        "header_type_offset": 16,
        "book_symbol_offset": 132,
        "trade_symbol_offset": 36,
    }
    actual = {
        "byte_order": sys.byteorder,
        "header_bytes": ct.sizeof(Header),
        "book_bytes": ct.sizeof(Book),
        "trade_bytes": ct.sizeof(Trade),
        "header_time_offset": Header.time.offset,
        "header_type_offset": Header.type.offset,
        "book_symbol_offset": Book.symbol.offset,
        "trade_symbol_offset": Trade.symbol.offset,
    }
    require(actual == expected, "unsupported native ABI; do not reinterpret records")
    return actual


def exact_read(stream, count):
    parts, size = [], 0
    while size < count:
        part = stream.read(count - size)
        if not part:
            break
        parts.append(part)
        size += len(part)
    return b"".join(parts)


def read_record(stream, symbol, previous, ordinal):
    header_bytes = exact_read(stream, ct.sizeof(Header))
    if not header_bytes:
        return None
    require(len(header_bytes) == ct.sizeof(Header), "truncated native header")
    header = Header.from_buffer_copy(header_bytes)
    payload_type = Book if header.type == b"B" else Trade if header.type == b"T" else None
    require(payload_type is not None, "unsupported native message type")
    payload_bytes = exact_read(stream, ct.sizeof(payload_type))
    require(len(payload_bytes) == ct.sizeof(payload_type), "truncated native payload")
    payload = payload_type.from_buffer_copy(payload_bytes)
    require(bytes(payload.symbol).decode("ascii") == symbol, "native symbol/ABI mismatch")
    # BinLoader only clamps R. It does NOT replace nonpositive payload E by R.
    return {
        "ordinal": ordinal,
        "seq": int(header.seq),
        "type": header.type.decode(),
        "raw_receive": int(header.time),
        "receive": max(int(header.time), previous),
        "exchange": int(payload.exchange),
        "status": int(payload.status),
        "ignored_h15_observation": header.type == b"T" and payload.quantity <= 0 or header.type == b"B" and payload.bid_depth == 0 and payload.ask_depth == 0,
        "bytes": header_bytes + payload_bytes,
    }


def evidence(row):
    return {key: value for key, value in row.items() if key != "bytes"}


def positive_clock(value):
    return type(value) is int and 0 < value < 2**53


def closed_window(group, following, samples):
    if not group or not group["continuous"] or not group["has_book"]:
        return []
    last = group["last"]
    if following["status"] & 7 or following["exchange"] <= last["exchange"]:
        return []
    if not all(positive_clock(v) for v in (last["exchange"], last["receive"], following["receive"], following["exchange"])):
        return []
    r, a = last["receive"], following["receive"]
    selected = samples[bisect.bisect_right(samples, r) : bisect.bisect_left(samples, a)]
    return [
        {"S": s, "R": r, "A": a, "pending_exchange": last["exchange"], "pending_first": group["first"], "pending_last": evidence(last), "next_distinct": evidence(following)}
        for s in selected
    ]


def scan_stream(path, symbol, destination, samples, remaining_budget):
    """Exactly one compressed-source decode; spool unmodified records for cutting."""
    abi()
    require(not destination.exists(), "fresh raw spool required")
    previous, ordinal, size, group = 0, 0, 0, None
    times, ends, windows = array("q"), array("Q"), []
    h = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as stream, destination.open("wb") as spool:
        while (row := read_record(stream, symbol, previous, ordinal + 1)) is not None:
            previous, ordinal = row["receive"], row["ordinal"]
            size += len(row["bytes"])
            require(size <= remaining_budget, "fixed spool budget exceeded; no replacement scan")
            spool.write(row["bytes"])
            h.update(row["bytes"])
            times.append(row["receive"])
            ends.append(size)
            if symbol not in producer.TARGETS:
                continue
            if row["ignored_h15_observation"]:
                continue
            if group and row["exchange"] != group["last"]["exchange"]:
                windows.extend(closed_window(group, row, samples))
                group = None
            if group is None:
                group = {"first": evidence(row), "last": row, "has_book": False, "continuous": True}
            group["last"] = row
            group["has_book"] |= row["type"] == "B"
            group["continuous"] &= row["status"] & 7 == 0 and positive_clock(row["exchange"]) and positive_clock(row["receive"])
    # Final group deliberately remains open. Neither EOF nor timers create A.
    require(ordinal > 0, "empty fixed native stream; no invented warm-in")
    return {
        "symbol": symbol,
        "spool": record(destination),
        "records": ordinal,
        "uncompressed_bytes": size,
        "uncompressed_sha256": h.hexdigest(),
        "times": times,
        "ends": ends,
        "windows": windows,
    }


def carrier_at(scanned, index):
    offset = 0 if index == 0 else scanned["ends"][index - 1]
    with Path(scanned["spool"]["path"]).open("rb") as stream:
        stream.seek(offset)
        row = read_record(stream, scanned["symbol"], int(scanned["times"][index]), index + 1)
    return {"symbol": scanned["symbol"], **evidence(row)}


def select_carrier(scans, own_symbol, s, a):
    candidates = []
    for symbol in producer.CARRIERS:
        if symbol == own_symbol:
            continue
        scanned = scans[symbol]
        index = bisect.bisect_left(scanned["times"], s)
        if index < len(scanned["times"]) and scanned["times"][index] < a:
            row = carrier_at(scanned, index)
            candidates.append(row)
    return min(candidates, key=lambda row: (row["receive"], row["seq"], -ord(row["type"]), row["symbol"], scans[row["symbol"]]["source"]["path"])) if candidates else None


def same_bits(left, right):
    if left.type != right.type or len(left) != len(right) or left.null_count or right.null_count:
        return False
    if pa.types.is_float64(left.type):
        return np.array_equal(left.to_numpy().view(np.uint64), right.to_numpy().view(np.uint64))
    if pa.types.is_floating(left.type):
        return False
    return left.equals(right)


def native_keys(table):
    require(all(table[name].type == pa.int64() and table[name].null_count == 0 for name in KEYS), "native origin keys must remain nonnullable int64")
    rows = list(zip(*(table[name].to_pylist() for name in KEYS), strict=True))
    require(len(rows) == len(set(rows)), "duplicate native origin keys")
    require(all(type(v) is int and 0 <= v < 2**53 for row in rows for v in row) and all(0 < row[0] and row[1] <= row[0] for row in rows), "unsafe/future native book key")
    return rows


def compare_tables(full, prefix, cutoff):
    require(full.schema.equals(prefix.schema, check_metadata=True), "native field/schema metadata drift")
    full_keys, prefix_keys = native_keys(full), native_keys(prefix)
    expected_indices = [i for i, key in enumerate(full_keys) if key[0] <= cutoff]
    require(prefix_keys == [full_keys[i] for i in expected_indices], "prefix invented/dropped/reordered original origins or emitted afterD")
    expected = full.take(pa.array(expected_indices, type=pa.int64()))
    checks = {name: same_bits(expected[name], prefix[name]) for name in full.column_names}
    require(all(checks.values()), "native common-origin bits changed, including sentinel/category/diagnostics")
    return {"common_rows": len(prefix_keys), "field_bit_checks": checks}


def write_cut(scanned, cutoff, path):
    require(not path.exists(), "fresh native cut required")
    count = bisect.bisect_right(scanned["times"], cutoff)
    require(count > 0, "empty warm-in source prefix; do not synthesize a row")
    length, h = int(scanned["ends"][count - 1]), hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(scanned["spool"]["path"]).open("rb") as stream, path.open("wb") as sink, zstd.ZstdCompressor(level=3).stream_writer(sink) as writer:
        remaining = length
        while remaining:
            block = exact_read(stream, min(1024 * 1024, remaining))
            require(block, "spool truncated while preserving prefix")
            writer.write(block)
            h.update(block)
            remaining -= len(block)
    return {
        "symbol": scanned["symbol"],
        "source": scanned["source"],
        **record(path),
        "records": count,
        "uncompressed_prefix_bytes": length,
        "uncompressed_prefix_sha256": h.hexdigest(),
        "last_effective_receive": int(scanned["times"][count - 1]),
    }


def prefix_config(original, raw_dir):
    config = copy.deepcopy(original)
    group = next(g for g in config["Modules"] if g["Gid"] == "")
    md = next(d for d in group["Decl"] if d["Desc"] == "TradeBookMd.0")
    md["Spec"]["Dirs"] = [str(raw_dir)]
    return config


def parent_dependencies():
    producer_path, support_path = PARENT / "frozen-method.yaml", PARENT / "frozen-support-evaluation-method.yaml"
    production = canonical_method(producer_path, "h15-native-producer-method-v2")
    support = canonical_method(support_path, "h15-native-support-evaluation-method-v2")
    require(production["identity"] == PRODUCER_IDENTITY and support["identity"] == SUPPORT_IDENTITY, "fixed H15 parent method identity changed")
    check_record(production["preparation"])
    preparation = read_yaml(Path(production["preparation"]["path"]))
    check_record(preparation["producer"])
    capsule = read_yaml(Path(preparation["producer"]["path"]))
    require(len(capsule["sources"]) == 1648 and preparation["binary"]["sha256"] == producer.BINARY_SHA256, "fixed compiled native/source capsule required")
    validation_path = PARENT / "source-support-validation.yaml"
    validation = read_yaml(validation_path)
    require(
        validation["schema"] == "h15-native-source-support-validation-v2"
        and validation["producer_identity"] == PRODUCER_IDENTITY
        and validation["method_identity"] == SUPPORT_IDENTITY
        and validation["native_artifact_integrity_passed"] is True
        and validation["support_gate"]["passed"] is True
        and validation["labels_read"] is False
        and validation["model_fits"] == 0,
        "actual qualified H15 native support receipt required",
    )
    source_files = [record(Path(__file__)), record(OUTPUT / "test_receive_prefix.py"), record(PLAN), record(ROOT / "AstraResearch/astra/io.py")]
    sources = unique_records([*source_files, *production["sources"], *support["sources"]])
    inputs = unique_records(
        [
            *production["inputs"],
            *support["inputs"],
            *validation["inputs"],
            record(PROFILE),
            record(validation_path),
            record(producer_path),
            record(producer_path.with_suffix(".yaml.identity")),
            record(support_path),
            record(support_path.with_suffix(".yaml.identity")),
        ]
    )
    return preparation, sources, inputs


def bind_plan():
    require(not (OUTPUT / "bound-plan.yaml").exists(), "fresh prospective prefix plan required")
    profile = read_yaml(PROFILE)
    require(profile == CONTRACT, "prospective selector cannot change after support/data inspection")
    preparation, sources, inputs = parent_dependencies()
    body = {
        "schema": "h15-real-receive-prefix-bound-plan-v1",
        "profile": record(PROFILE),
        "profile_identity": digest(profile),
        "sources": sources,
        "inputs": inputs,
        "producer_identity": PRODUCER_IDENTITY,
        "support_identity": SUPPORT_IDENTITY,
        "binary": preparation["binary"],
        "native_replays_launched": 0,
        "raw_streams_decoded": 0,
        "labels_read": False,
        "model_fits": 0,
    }
    write_yaml(OUTPUT / "bound-plan.yaml", body)
    return body


def verify_selection(path):
    method = canonical_method(path, "h15-real-receive-prefix-selection-method-v1")
    require(read_yaml(PROFILE) == CONTRACT and method["profile_identity"] == digest(CONTRACT), "frozen prefix selector/profile changed")
    bound_path = OUTPUT / "bound-plan.yaml"
    require(method["plan"] == record(bound_path), "different prospective bound-plan")
    bound = read_yaml(bound_path)
    require(
        bound.get("raw_streams_decoded") == 0
        and type(bound.get("raw_streams_decoded")) is int
        and bound.get("native_replays_launched") == 0
        and bound.get("labels_read") is False
        and bound.get("model_fits") == 0,
        "selection plan must precede data/replay/model access",
    )
    preparation, sources, inputs = parent_dependencies()
    require(bound["profile"] == record(PROFILE) and bound["profile_identity"] == digest(CONTRACT), "bound profile changed")
    require(bound["sources"] == sources and bound["inputs"] == inputs, "prospective parent evidence closure changed")
    for scope, nodes in (("sources", sources), ("inputs", [*inputs, record(bound_path)])):
        producer.require_closure(method, scope, nodes)
        for node in method[scope]:
            check_record(node)
    abi()
    return method, preparation


def prepare(method_path):
    method, preparation = verify_selection(method_path)
    destination = OUTPUT / "preparation-receipt.yaml"
    require(not destination.exists() and not (OUTPUT / "spool").exists() and not (OUTPUT / "truncated").exists(), "fresh prefix cut required; no overwrite/resume")
    native_job = next(j for j in preparation["jobs"] if j["day"] == DAY)
    require(native_job["symbols"] == list(producer.TARGETS), "fixed Jan19 pair unavailable; no replacement date")
    needed = [
        *KEYS,
        "H15_mark_is_qualified",
        "H15_qualified_mark_available_time",
        "H15_source_mark_id",
        *[f"H15_processed_{leg}_{field}_time" for leg in ("peer", "target") for field in ("exchange", "available")],
        *[f"PeerTradeInformation.0.k_{leg}_phase.0" for leg in ("peer", "target")],
    ]
    tables = {target: pq.read_table(Path(native_job["work"]) / "data" / DAY / target / "values.parquet", columns=needed, use_threads=False) for target in producer.TARGETS}
    keys = {target: native_keys(table) for target, table in tables.items()}
    sample_times = sorted({k[0] for rows in keys.values() for k in rows if k[0] % 30_000_000 == 0})
    raw = {n["symbol"]: n for n in preparation["inputs"] if n.get("day") == DAY and n["path"].endswith(".bin.zst")}
    require(set(raw) == set(producer.CARRIERS), "exact original16 BIN streams required")
    scans, budget = {}, CONTRACT["max_uncompressed_spool_bytes"]
    for symbol in producer.CARRIERS:
        scanned = scan_stream(Path(raw[symbol]["path"]), symbol, OUTPUT / "spool" / f"{symbol}.bin", sample_times, budget)
        scanned["source"] = raw[symbol]
        scans[symbol] = scanned
        budget -= scanned["uncompressed_bytes"]
    selected = None
    for target in producer.TARGETS:
        table, index_by_time = tables[target], {key[0]: i for i, key in enumerate(keys[target])}
        for leg in ("peer", "target"):
            symbol = producer.PEERS[target] if leg == "peer" else target
            for window in sorted(scans[symbol]["windows"], key=lambda w: w["S"]):
                i = index_by_time.get(window["S"])
                if i is None:
                    continue
                value = lambda name, source=table, row=i: source[name][row].as_py()
                if not (
                    value("H15_mark_is_qualified") == 1.0
                    and value("H15_source_mark_id") > 0
                    and 0 < value("H15_qualified_mark_available_time") <= window["S"]
                    and value(f"PeerTradeInformation.0.k_{leg}_phase.0") in ("quote", "flow")
                    and 0 < value(f"H15_processed_{leg}_exchange_time") < window["pending_exchange"]
                    and 0 < value(f"H15_processed_{leg}_available_time") <= window["R"]
                ):
                    continue
                carrier = select_carrier(scans, symbol, window["S"], window["A"])
                if carrier:
                    selected = {
                        **window,
                        "target": target,
                        "peer": producer.PEERS[target],
                        "leg": leg,
                        "pending_symbol": symbol,
                        "D": carrier["receive"],
                        "carrier": carrier,
                        "origin_keys": dict(zip(KEYS, keys[target][i], strict=True)),
                    }
                    break
            if selected:
                break
        if selected:
            break
    spool_nodes = [scans[s]["spool"] for s in producer.CARRIERS]
    base = {
        "schema": "h15-real-receive-prefix-preparation-v1",
        "selection_method": record(method_path),
        "selection_identity": method["identity"],
        "producer_identity": PRODUCER_IDENTITY,
        "support_identity": SUPPORT_IDENTITY,
        "binary": preparation["binary"],
        "abi": abi(),
        "genuine_straddle_found": selected is not None,
        "generated_events": 0,
        "labels_read": False,
        "model_fits": 0,
        "native_replays_launched": 0,
        "spools": spool_nodes,
        "sources": method["sources"],
        "inputs": unique_records([*method["inputs"], record(method_path), record(method_path.with_suffix(".yaml.identity")), *spool_nodes]),
    }
    if selected is None:
        write_yaml(destination, base)
        return {"prepared": False, "genuine_straddle_found": False, "receipt": str(destination)}
    r, s, d, a = (selected[name] for name in ("R", "S", "D", "A"))
    require(r < s <= d < a, "real straddle inequality changed")
    cuts = []
    for symbol in producer.CARRIERS:
        scanned = scans[symbol]
        cut = OUTPUT / "raw" / symbol / f"{DAY}.bin.zst"
        cuts.append(write_cut(scanned, d, cut))
    work = OUTPUT / "truncated"
    work.mkdir()
    config = work / "config.yaml"
    original = read_yaml(Path(native_job["config"]["path"]))
    write_yaml(config, prefix_config(original, OUTPUT / "raw"))
    full_outputs = [record(Path(p)) for p in native_job["outputs"]]
    base.update(
        selected=selected,
        cuts=cuts,
        config=record(config),
        original_config=native_job["config"],
        day=DAY,
        full_outputs=full_outputs,
        command=[preparation["binary"]["path"], "-d", DAY, "-C", str(work), "--trading-calendar", str(producer.CALENDAR), "--run-status-dir", str(work / "status"), str(config)],
        environment=native_job["environment"],
        prefix_outputs=[str(work / folder / DAY / symbol / "values.parquet") for folder in ("data", "events") for symbol in producer.TARGETS],
    )
    base["inputs"] = unique_records([*base["inputs"], *full_outputs, *[record(Path(c["path"])) for c in cuts], record(config), native_job["config"]])
    write_yaml(destination, base)
    return {"prepared": True, "selection": selected, "receipt": str(destination), "command": base["command"], "native_replays_launched": 0}


def bind_comparison(selection_path):
    selection, _ = verify_selection(selection_path)
    path = OUTPUT / "preparation-receipt.yaml"
    receipt = read_yaml(path)
    require(receipt["genuine_straddle_found"] is True and receipt["selection_identity"] == selection["identity"], "genuine frozen source prefix required")
    require(
        receipt["selected"]["R"] < receipt["selected"]["S"] <= receipt["selected"]["D"] < receipt["selected"]["A"], "genuine prefix inequality required before comparison freeze"
    )
    require([c["symbol"] for c in receipt["cuts"]] == list(producer.CARRIERS), "all16 original carrier prefixes required")
    require(all(c["last_effective_receive"] <= receipt["selected"]["D"] for c in receipt["cuts"]), "a source cut contains future receive information")
    require(not (OUTPUT / "bound-comparison.yaml").exists(), "fresh comparison bind required")
    inputs = unique_records([*receipt["inputs"], record(path)])
    for node in inputs:
        check_record(node)
    body = {
        "schema": "h15-real-receive-prefix-bound-comparison-v1",
        "preparation": record(path),
        "preparation_identity": digest(receipt),
        "sources": receipt["sources"],
        "inputs": inputs,
        "selection_identity": selection["identity"],
        "outputs_read": False,
        "labels_read": False,
        "model_fits": 0,
    }
    write_yaml(OUTPUT / "bound-comparison.yaml", body)
    return body


def check(comparison_path):
    method = canonical_method(comparison_path, "h15-real-receive-prefix-comparison-method-v1")
    require(method["plan"] == record(OUTPUT / "bound-comparison.yaml"), "wrong comparison bind")
    bound = read_yaml(OUTPUT / "bound-comparison.yaml")
    receipt_path = OUTPUT / "preparation-receipt.yaml"
    receipt = read_yaml(receipt_path)
    selection_path = Path(receipt["selection_method"]["path"])
    selection, preparation = verify_selection(selection_path)
    require(
        bound["selection_identity"] == receipt["selection_identity"] == selection["identity"]
        and bound["preparation"] == record(receipt_path)
        and bound["preparation_identity"] == digest(receipt),
        "frozen prefix/preparation identity drift",
    )
    for scope, nodes in (("sources", selection["sources"]), ("inputs", [*receipt["inputs"], record(receipt_path), record(OUTPUT / "bound-comparison.yaml")])):
        producer.require_closure(method, scope, nodes)
        for node in method[scope]:
            check_record(node)
    destination = OUTPUT / "validation.yaml"
    require(not destination.exists(), "fresh real prefix validation receipt required")
    status_path = OUTPUT / "truncated/status" / f"{DAY}.yaml"
    status = read_yaml(status_path)
    require(status.get("status") == "completed" and status.get("fatal_error") is False, "root native prefix replay not completed")
    original_config = read_yaml(Path(receipt["original_config"]["path"]))
    require(read_yaml(Path(receipt["config"]["path"])) == prefix_config(original_config, OUTPUT / "raw"), "native prefix config altered more than source directory")
    require(receipt["binary"] == preparation["binary"] and receipt["binary"]["sha256"] == producer.BINARY_SHA256, "different native binary used for prefix")
    execution_path = OUTPUT / "truncated/execution-receipt.yaml"
    execution = read_yaml(execution_path)
    require(
        execution.get("method_identity") == method["identity"]
        and execution.get("job") == "receive_prefix"
        and execution.get("exit_code") == 0
        and type(execution.get("exit_code")) is int
        and execution.get("command") == receipt["command"]
        and [n["path"] for n in execution.get("outputs", [])] == receipt["prefix_outputs"],
        "prefix replay not bound to exact frozen comparison command/outputs",
    )
    for node in [execution["log"], *execution["outputs"]]:
        check_record(node)
    selected, checks, inputs = receipt["selected"], [], [record(status_path), record(execution_path), execution["log"], *execution["outputs"]]
    require(selected["R"] < selected["S"] <= selected["D"] < selected["A"], "genuine receive straddle changed")
    for full_node, prefix_path in zip(receipt["full_outputs"], receipt["prefix_outputs"], strict=True):
        full = pq.read_table(full_node["path"], use_threads=False)
        prefix = pq.read_table(prefix_path, use_threads=False)
        expected = (
            set(KEYS)
            | {f"PeerTradeInformation.0.{name}.0" for name in producer.ALPHA}
            | {f"H15_{name}" for name in producer.INFO}
            | {"OriginMidPrice", "OriginMidTicks", "OriginBidTicks", "OriginAskTicks"}
        )
        require(set(full.column_names) == set(prefix.column_names) == expected, "undeclared native fields/labels or missing H15 representation")
        result = compare_tables(full, prefix, selected["D"])
        result.update(full_path=full_node["path"], prefix_path=prefix_path)
        if Path(prefix_path).parent.name == selected["target"] and "/data/" in prefix_path:
            origin = tuple(selected["origin_keys"][name] for name in KEYS)
            keys = native_keys(prefix)
            require(origin in keys and origin[0] == selected["S"], "selected original30s origin absent")
            index = keys.index(origin)
            require(
                0 < prefix[f"H15_processed_{selected['leg']}_exchange_time"][index].as_py() < selected["pending_exchange"]
                and prefix[f"H15_processed_{selected['leg']}_available_time"][index].as_py() <= selected["R"],
                "pending own-E published or backdated by EOF/timer",
            )
            result["selected_common_origin_present"] = True
        checks.append(result)
        inputs.append(record(Path(prefix_path)))
    require(any(c.get("selected_common_origin_present") is True for c in checks), "mandatory S comparison missing")
    result = {
        "schema": "h15-real-receive-prefix-validation-v1",
        "passed": True,
        "genuine_straddle_found": True,
        "common_origin_present": True,
        "all_common_origin_bits_exact": True,
        "pending_cluster_not_published_at_origin": True,
        "producer_identity": PRODUCER_IDENTITY,
        "binary_sha256": receipt["binary"]["sha256"],
        "selection_method_identity": selection["identity"],
        "comparison_method_identity": method["identity"],
        "selected": selected,
        "tables": checks,
        "sources": method["sources"],
        "inputs": unique_records([*method["inputs"], *inputs, record(comparison_path), record(comparison_path.with_suffix(".yaml.identity"))]),
        "labels_read": False,
        "model_fits": 0,
        "generated_events": 0,
        "predictive_admission": False,
    }
    write_yaml(destination, result)
    return {"passed": True, "receipt": str(destination), "common_native_tables": len(checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--bind-plan", action="store_true")
    modes.add_argument("--prepare", type=Path)
    modes.add_argument("--bind-comparison", type=Path)
    modes.add_argument("--check", type=Path)
    args = parser.parse_args()
    result = (
        bind_plan()
        if args.bind_plan
        else prepare(args.prepare.resolve())
        if args.prepare
        else bind_comparison(args.bind_comparison.resolve())
        if args.bind_comparison
        else check(args.check.resolve())
    )
    if args.bind_plan or args.bind_comparison:
        result = {
            "bound_only": True,
            "receipt": str(OUTPUT / ("bound-plan.yaml" if args.bind_plan else "bound-comparison.yaml")),
            "sources": len(result["sources"]),
            "inputs": len(result["inputs"]),
            "native_replays_launched": 0,
            "model_fits": 0,
        }
    print(result)


if __name__ == "__main__":
    main()
