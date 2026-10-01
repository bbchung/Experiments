"""Draft root-only H16 real receive-prefix causality study.

Selection uses only original keys and existing complete native spool headers.
All16 raw byte prefixes are replayed unchanged by root; only a separately
frozen comparison decodes native feature values. No Python feature is formed.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
from array import array
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

OUTPUT = Path(__file__).resolve().parent
PILOT = OUTPUT.parent / "native-h16-auction-cycle"
OLD_PREFIX = OUTPUT.parent / "native-h15-receive-prefix"
ROOT = OUTPUT.parents[3]
PLAN = ROOT / "AstraResearch/experiments/information_state_20260930/H16_PREFIX_PLAN.md"
DAY, TARGET = "20260119", "3481"
SELECTION_SCHEMA = "h16-real-receive-prefix-selection-method-v1"
COMPARISON_SCHEMA = "h16-real-receive-prefix-comparison-method-v1"
CONTRACT = {
    "day": DAY,
    "target": TARGET,
    "origin_seconds": 30,
    "origin_order": "ascending original target keys S>=09:10; earliest S with R<S<D<A",
    "R": "last loader-effective target receive strictly before S",
    "A": "first loader-dispatched target record with effective receive strictly after S",
    "D": "earliest global loader-dispatched record with effective receive strictly after S; native comparator ties",
    "records": "all original16 full warm-in raw bytes with loader-effective R<=D; no filtering, synthesis or EOF/timer closure",
    "state_eligibility": "none; source headers and original keys only; no Alpha/category/quantity/label gate",
    "absence": "missing native output allowed only with frozen original common key count zero; no synthesized empty file",
    "comparison": "all16 exact ordered common key populations<=D and all36 native fields, schema metadata, uint64 float bits and strings",
    "max_total_existing_spool_bytes": 536870912,
    "labels_read": False,
    "model_fits": 0,
    "python_features_generated": False,
}


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = import_file("h16_prefix_immutable_pilot", PILOT / "native_pilot.py")
reader = import_file("h16_prefix_immutable_abi_reader", OLD_PREFIX / "prepare_receive_prefix.py")
require, record, check_record = pilot.require, pilot.record, pilot.check_record
canonical_method, unique_records, require_closure = pilot.canonical_method, pilot.unique_records, pilot.require_closure
digest, read_yaml, write_yaml = pilot.digest, pilot.read_yaml, pilot.write_yaml
freeze_payload = pilot.freeze_payload
KEYS, CARRIERS = pilot.KEYS, pilot.CARRIERS
ABI_SOURCES = ("src/msg/md_msg.h", "src/oms/modules/md/trade_book_md/trade_book_md.h", "src/oms/modules/md/trade_book_md/trade_book_md.cpp")


def own_sources():
    return [record(path) for path in (Path(__file__), OUTPUT / "test_receive_prefix.py", PLAN, OLD_PREFIX / "prepare_receive_prefix.py")]


def snapshot_record(capsule, relative):
    matches = [node for node in capsule["sources"] if node["path"].endswith("/" + relative)]
    require(len(matches) == 1, "one compiled ABI/loader source record required")
    return matches[0]


def parent_context(producer_path, support_path, support_receipt_path=None):
    """Root metadata/hash bind only; no keys, headers or feature values yet."""
    producer_path, support_path = Path(producer_path).resolve(), Path(support_path).resolve()
    support, _ = pilot.verify_evaluation(support_path)
    producer = canonical_method(producer_path, pilot.PRODUCER_SCHEMA)
    require(support["producer_method"] == record(producer_path) and support["producer_identity"] == producer["identity"], "different H16 native pilot producer/support")
    preparation = read_yaml(check_record(producer["preparation"]))
    validation_path = Path(support_receipt_path).resolve() if support_receipt_path else PILOT / "source-support-validation.yaml"
    validation = read_yaml(validation_path)
    require(
        validation.get("schema") == "h16-native-source-support-validation-v1"
        and validation.get("passed") is True
        and validation.get("native_artifact_integrity_passed") is True
        and validation.get("all_original_cells_present") is True
        and validation.get("support_gate", {}).get("passed") is True
        and validation.get("producer_identity") == producer["identity"]
        and validation.get("method_identity") == support["identity"]
        and validation.get("labels_read") is False
        and type(validation.get("model_fits")) is int
        and validation["model_fits"] == 0
        and validation.get("python_features_generated") is False,
        "actual complete H16 pilot support PASS required before any prefix job",
    )
    for scope in ("sources", "inputs"):
        require_closure(validation, scope, support[scope])
    jobs = [job for job in preparation["jobs"] if job["day"] == DAY]
    require(len(jobs) == 1 and jobs[0]["symbols"] == list(CARRIERS), "fixed Jan19 full16 original native job required; no replacement")
    native_job = jobs[0]
    executed, execution_nodes = pilot.execution(native_job, producer["identity"])
    full_outputs = executed["outputs"]
    old_selection_path = OLD_PREFIX / "frozen-selection-method.yaml"
    old_selection, old_preparation = reader.verify_selection(old_selection_path)
    old_comparison_path = OLD_PREFIX / "frozen-comparison-method.yaml"
    old_comparison = canonical_method(old_comparison_path, "h15-real-receive-prefix-comparison-method-v1")
    spool_receipt_path = OLD_PREFIX / "preparation-receipt.yaml"
    spool_receipt = read_yaml(spool_receipt_path)
    old_bound_path = OLD_PREFIX / "bound-comparison.yaml"
    old_bound = read_yaml(check_record(old_comparison["plan"]))
    require(
        old_comparison["plan"] == record(old_bound_path)
        and old_bound["preparation"] == record(spool_receipt_path)
        and old_bound["preparation_identity"] == digest(spool_receipt)
        and old_bound["selection_identity"] == old_selection["identity"],
        "complete-spool receipt was not sealed by the original comparison method",
    )
    require_closure(old_comparison, "sources", old_selection["sources"])
    require_closure(old_comparison, "inputs", [*spool_receipt["inputs"], record(spool_receipt_path), record(old_bound_path)])
    require(
        spool_receipt.get("schema") == "h15-real-receive-prefix-preparation-v1"
        and spool_receipt.get("selection_method") == record(old_selection_path)
        and spool_receipt.get("selection_identity") == old_selection["identity"]
        and spool_receipt.get("abi") == reader.abi()
        and spool_receipt.get("labels_read") is False
        and type(spool_receipt.get("model_fits")) is int
        and spool_receipt["model_fits"] == 0,
        "complete immutable original-spool provenance required",
    )
    require([node["path"] for node in spool_receipt["spools"]] == [str(OLD_PREFIX / "spool" / f"{symbol}.bin") for symbol in CARRIERS], "original full16 spool order/path changed")
    require(sum(node["size"] for node in spool_receipt["spools"]) <= CONTRACT["max_total_existing_spool_bytes"], "existing spool budget exceeded; no alternate inputs")
    new_raw = {node["symbol"]: node for node in preparation["raw_files"] if node["day"] == DAY}
    old_raw = {node["symbol"]: node for node in old_preparation["inputs"] if node.get("day") == DAY and node["path"].endswith(".bin.zst")}
    require(set(new_raw) == set(old_raw) == set(CARRIERS), "all16 original source streams required")
    require(all(new_raw[symbol] == old_raw[symbol] for symbol in CARRIERS), "existing spools correspond to another original raw source")
    require_closure(spool_receipt, "inputs", [*old_raw.values(), *spool_receipt["spools"]])
    old_capsule = read_yaml(check_record(old_preparation["producer"]))
    new_capsule = read_yaml(check_record(preparation["producer"]))
    for relative in ABI_SOURCES:
        require(
            snapshot_record(old_capsule, relative)["sha256"] == snapshot_record(new_capsule, relative)["sha256"],
            "native binary loader ABI differs from existing complete spool reader",
        )
    sources = unique_records([*own_sources(), *producer["sources"], *support["sources"], *old_selection["sources"], *old_comparison["sources"]])
    inputs = unique_records(
        [
            *producer["inputs"],
            *support["inputs"],
            *validation["inputs"],
            *old_selection["inputs"],
            *old_comparison["inputs"],
            *spool_receipt["inputs"],
            *execution_nodes,
            *full_outputs,
            record(validation_path),
            record(producer_path),
            record(producer_path.with_suffix(producer_path.suffix + ".identity")),
            record(support_path),
            record(support_path.with_suffix(support_path.suffix + ".identity")),
            record(old_selection_path),
            record(old_selection_path.with_suffix(old_selection_path.suffix + ".identity")),
            record(old_comparison_path),
            record(old_comparison_path.with_suffix(old_comparison_path.suffix + ".identity")),
            record(old_bound_path),
            record(spool_receipt_path),
        ]
    )
    for node in [*sources, *inputs]:
        check_record(node)
    return {
        "producer_method": record(producer_path),
        "support_method": record(support_path),
        "support_receipt": record(validation_path),
        "producer_identity": producer["identity"],
        "support_identity": support["identity"],
        "binary": preparation["binary"],
        "spool_receipt": record(spool_receipt_path),
        "spools": spool_receipt["spools"],
        "raw_sources": [new_raw[symbol] for symbol in CARRIERS],
        "native_job": native_job,
        "full_outputs": full_outputs,
        "abi": reader.abi(),
        "sources": sources,
        "inputs": inputs,
    }


def bind_selection(producer_path, support_path, support_receipt_path=None):
    context = parent_context(producer_path, support_path, support_receipt_path)
    path = OUTPUT / "bound-selection.yaml"
    require(not path.exists(), "fresh selection bind required")
    bound = {
        "schema": "h16-real-receive-prefix-bound-selection-v1",
        "contract": CONTRACT,
        **context,
        "raw_headers_read": False,
        "original_keys_read": False,
        "native_values_read": False,
        "native_replays_launched": 0,
    }
    write_yaml(path, bound)
    return {
        "schema": SELECTION_SCHEMA,
        "status": "draft",
        "plan": record(path),
        "contract_identity": digest(CONTRACT),
        "sources": context["sources"],
        "inputs": unique_records([*context["inputs"], record(path)]),
    }


def verify_selection(path):
    method = canonical_method(path, SELECTION_SCHEMA)
    bound_path = OUTPUT / "bound-selection.yaml"
    require(method["plan"] == record(bound_path) and method["contract_identity"] == digest(CONTRACT), "selection contract/plan drift")
    bound = read_yaml(bound_path)
    require(
        bound["schema"] == "h16-real-receive-prefix-bound-selection-v1"
        and bound["contract"] == CONTRACT
        and bound["raw_headers_read"] is False
        and bound["original_keys_read"] is False
        and bound["native_values_read"] is False
        and type(bound["native_replays_launched"]) is int
        and bound["native_replays_launched"] == 0,
        "selector must be frozen before headers/keys/values/replay",
    )
    current = parent_context(bound["producer_method"]["path"], bound["support_method"]["path"], bound["support_receipt"]["path"])
    require(all(bound[key] == value for key, value in current.items()), "prefix parent/spool lineage drift")
    require_closure(method, "sources", current["sources"])
    require_closure(method, "inputs", [*current["inputs"], record(bound_path)])
    for scope in ("sources", "inputs"):
        for node in method[scope]:
            check_record(node)
    return method, bound


@lru_cache(maxsize=16)
def scan_index(path, sha256, expected_size, symbol):
    """Immutable byte identity caches one header pass in a root process."""
    reader.abi()
    previous, ordinal, size = 0, 0, 0
    times, ends = array("q"), array("Q")
    with Path(path).open("rb") as stream:
        while (row := reader.read_record(stream, symbol, previous, ordinal + 1)) is not None:
            previous, ordinal = row["receive"], row["ordinal"]
            size += len(row["bytes"])
            times.append(previous)
            ends.append(size)
    require(size == expected_size and ordinal > 0, "complete original spool truncated/empty")
    return times, ends, ordinal, size


def scan_spool(node, symbol, source):
    """Root-only existing spool scan; no feature eligibility/duplicate spool."""
    check_record(node)
    times, ends, ordinal, size = scan_index(node["path"], node["sha256"], node["size"], symbol)
    return {"symbol": symbol, "spool": node, "source": source, "times": times, "ends": ends, "records": ordinal, "uncompressed_bytes": size}


def header_at(scan, index):
    offset = 0 if index == 0 else scan["ends"][index - 1]
    prior_r = 0 if index == 0 else int(scan["times"][index - 1])
    with Path(scan["spool"]["path"]).open("rb") as stream:
        stream.seek(offset)
        row = reader.read_record(stream, scan["symbol"], prior_r, index + 1)
    require(row is not None and row["receive"] == scan["times"][index], "spool header/index drift")
    return {key: row[key] for key in ("ordinal", "seq", "type", "raw_receive", "receive", "exchange", "status")}


def native_order(symbol, header, source):
    return header["receive"], header["seq"], 0 if header["type"] == "T" else 1, symbol, source["path"]


def select(scans, keys, start):
    """Prospective deterministic selector; no Alpha/price/quantity reads."""
    require(set(scans) == set(CARRIERS), "all16 scans required")
    target = scans[TARGET]
    for key in keys:
        s = key[0]
        if s < start or s % 30_000_000:
            continue
        before = bisect.bisect_left(target["times"], s) - 1
        after = bisect.bisect_right(target["times"], s)
        if before < 0 or after == len(target["times"]):
            continue
        r, a = int(target["times"][before]), int(target["times"][after])
        candidates = []
        for symbol in CARRIERS:
            scan = scans[symbol]
            index = bisect.bisect_right(scan["times"], s)
            if index == len(scan["times"]):
                continue
            header = header_at(scan, index)
            candidates.append((native_order(symbol, header, scan["source"]), symbol, header))
        if not candidates:
            continue
        _, carrier, header = min(candidates)
        d = int(header["receive"])
        if all(type(v) is int and 0 < v < 2**53 for v in (r, s, d, a)) and r < s < d < a:
            return {
                "day": DAY,
                "target": TARGET,
                "R": r,
                "S": s,
                "D": d,
                "A": a,
                "origin_keys": dict(zip(KEYS, key, strict=True)),
                "previous_target_record": header_at(target, before),
                "next_target_record": header_at(target, after),
                "carrier_symbol": carrier,
                "carrier_record": header,
            }
    return None


def selection_inputs(bound):
    node = bound["full_outputs"][list(CARRIERS).index(TARGET)]
    schema_identity(pq.read_schema(node["path"]))
    keys = reader.native_keys(pq.read_table(node["path"], columns=list(KEYS), use_threads=False))
    require(
        len(keys) == pilot.CONTRACT["expected_native_rows"] and all(keys[i][0] < keys[i + 1][0] for i in range(len(keys) - 1)),
        "fixed complete original periodic target keys required",
    )
    raw = {node["symbol"]: node for node in bound["raw_sources"]}
    scans = {symbol: scan_spool(node, symbol, raw[symbol]) for symbol, node in zip(CARRIERS, bound["spools"], strict=True)}
    return keys, scans


@lru_cache(maxsize=32)
def source_prefix_hash(path, sha256, length):
    remaining, h = length, hashlib.sha256()
    with Path(path).open("rb") as stream:
        while remaining:
            block = reader.exact_read(stream, min(1024 * 1024, remaining))
            require(block, "original byte prefix truncated")
            h.update(block)
            remaining -= len(block)
    return h.hexdigest()


@lru_cache(maxsize=32)
def cut_content_identity(path, sha256):
    size, h = 0, hashlib.sha256()
    with Path(path).open("rb") as stream, zstd.ZstdDecompressor().stream_reader(stream) as decompressed:
        while block := decompressed.read(1024 * 1024):
            size += len(block)
            require(size <= CONTRACT["max_total_existing_spool_bytes"], "cut exceeds fixed source budget")
            h.update(block)
    return size, h.hexdigest()


def validate_cuts(cuts, scans, cutoff):
    require([cut["symbol"] for cut in cuts] == list(CARRIERS), "all16 source cut order required")
    for cut in cuts:
        scan = scans[cut["symbol"]]
        count = bisect.bisect_right(scan["times"], cutoff)
        require(count > 0 and cut["source"] == scan["source"], "original full warm-in/source identity required")
        length = int(scan["ends"][count - 1])
        require(
            cut["path"] == str(OUTPUT / "raw" / cut["symbol"] / f"{DAY}.bin.zst")
            and type(cut["records"]) is int
            and cut["records"] == count
            and cut["uncompressed_prefix_bytes"] == length
            and cut["last_effective_receive"] == scan["times"][count - 1],
            "raw prefix dropped/invented/tied records or wrong cutoff",
        )
        check_record(cut)
        expected_hash = source_prefix_hash(scan["spool"]["path"], scan["spool"]["sha256"], length)
        actual_size, actual_hash = cut_content_identity(cut["path"], cut["sha256"])
        require(actual_size == length and actual_hash == expected_hash == cut["uncompressed_prefix_sha256"], "compressed cut bytes differ from exact original received prefix")


def schema_identity(schema):
    require(len(schema.names) == len(pilot.COLUMNS) and set(schema.names) == set(pilot.COLUMNS), "exact36 H16 fields required; no labels or unknown diagnostics")
    for name in pilot.COLUMNS:
        expected = pa.int64() if name in KEYS else pa.string() if name in pilot.CAT_COLUMNS else pa.float64()
        role = b"time" if name == "SampleTime" else b"context" if name in KEYS else b"feature" if name in pilot.ALPHA_COLUMNS else b"metadata"
        field = schema.field(name)
        require(field.type == expected and (field.metadata or {}).get(b"coco.role") == role, "native schema dtype/semantic-role drift")
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def key_proofs(full_outputs, cutoff):
    """Only schema and original keys; root binds before comparison freeze."""
    require(len(full_outputs) == len(CARRIERS), "full original16 writer outputs required")
    proofs = []
    for symbol, node in zip(CARRIERS, full_outputs, strict=True):
        check_record(node)
        schema = pq.read_schema(node["path"])
        identity = schema_identity(schema)
        keys = reader.native_keys(pq.read_table(node["path"], columns=list(KEYS), use_threads=False))
        require(all(keys[i][0] < keys[i + 1][0] for i in range(len(keys) - 1)), "periodic original key order changed")
        common = [key for key in keys if key[0] <= cutoff]
        proofs.append(
            {
                "symbol": symbol,
                "full": node,
                "full_key_rows": len(keys),
                "expected_common_rows": len(common),
                "expected_common_keys_identity": digest(common),
                "schema_identity": identity,
                "prefix_path": str(OUTPUT / "truncated" / "data" / DAY / symbol / "values.parquet"),
                "absence_allowed": len(common) == 0,
            }
        )
    return proofs


def prepare(selection_path):
    """ROOT ONLY after selection freeze: headers/keys, byte cuts, no native values."""
    selection_path = Path(selection_path).resolve()
    method, bound = verify_selection(selection_path)
    receipt_path = OUTPUT / "preparation-receipt.yaml"
    require(not receipt_path.exists() and not (OUTPUT / "raw").exists() and not (OUTPUT / "truncated").exists(), "fresh cut/receipt/work required; no overwrite/resume")
    keys, scans = selection_inputs(bound)
    selected = select(scans, keys, pilot.session_bounds(DAY)[0])
    receipt = {
        "schema": "h16-real-receive-prefix-preparation-v1",
        "selection_method": record(selection_path),
        "selection_identity": method["identity"],
        "producer_identity": bound["producer_identity"],
        "support_identity": bound["support_identity"],
        "binary": bound["binary"],
        "abi": bound["abi"],
        "genuine_straddle_found": selected is not None,
        "selected": selected,
        "existing_complete_spools_reused": True,
        "decompressed_new_sources": 0,
        "generated_events": 0,
        "native_values_read": False,
        "labels_read": False,
        "model_fits": 0,
        "native_replays_launched": 0,
        "sources": method["sources"],
        "inputs": unique_records([*method["inputs"], record(selection_path), record(selection_path.with_suffix(selection_path.suffix + ".identity"))]),
    }
    if selected is None:
        write_yaml(receipt_path, receipt)
        return {"prepared": False, "genuine_straddle_found": False, "receipt": str(receipt_path)}
    cuts = [reader.write_cut(scans[symbol], selected["D"], OUTPUT / "raw" / symbol / f"{DAY}.bin.zst") for symbol in CARRIERS]
    work = OUTPUT / "truncated"
    work.mkdir()
    original_config = bound["native_job"]["config"]
    config_path = work / "config.yaml"
    write_yaml(config_path, reader.prefix_config(read_yaml(check_record(original_config)), OUTPUT / "raw"))
    proofs = key_proofs(bound["full_outputs"], selected["D"])
    target_proof = proofs[list(CARRIERS).index(TARGET)]
    require(target_proof["expected_common_rows"] > 0, "mandatory original target S cannot be an empty prefix")
    command = [bound["binary"]["path"], "-d", DAY, "-C", str(work), "--trading-calendar", str(pilot.CALENDAR), "--run-status-dir", str(work / "status"), str(config_path)]
    receipt.update(
        cuts=cuts,
        config=record(config_path),
        original_config=original_config,
        command=command,
        environment=bound["native_job"]["environment"],
        prefix_outputs=[proof["prefix_path"] for proof in proofs],
        proofs=proofs,
    )
    receipt["inputs"] = unique_records([*receipt["inputs"], record(config_path), original_config, *[record(cut["path"]) for cut in cuts], *bound["full_outputs"]])
    write_yaml(receipt_path, receipt)
    return {"prepared": True, "selected": selected, "receipt": str(receipt_path), "command": command, "native_values_read": False, "native_replays_launched": 0}


def verify_preparation(selection_path):
    method, bound = verify_selection(selection_path)
    path = OUTPUT / "preparation-receipt.yaml"
    receipt = read_yaml(path)
    require(
        receipt["schema"] == "h16-real-receive-prefix-preparation-v1"
        and receipt["selection_method"] == record(selection_path)
        and receipt["selection_identity"] == method["identity"]
        and receipt["producer_identity"] == bound["producer_identity"]
        and receipt["support_identity"] == bound["support_identity"]
        and receipt["binary"] == bound["binary"]
        and receipt["abi"] == reader.abi()
        and receipt["genuine_straddle_found"] is True
        and receipt["native_values_read"] is False
        and receipt["labels_read"] is False
        and type(receipt["model_fits"]) is int
        and receipt["model_fits"] == 0,
        "prefix preparation lineage/phase changed",
    )
    selected = receipt["selected"]
    require(
        selected["day"] == DAY
        and selected["target"] == TARGET
        and all(type(selected[k]) is int and 0 < selected[k] < 2**53 for k in ("R", "S", "D", "A"))
        and selected["R"] < selected["S"] < selected["D"] < selected["A"],
        "strict genuine fixed source receive straddle required",
    )
    require(
        [cut["symbol"] for cut in receipt["cuts"]] == list(CARRIERS) and all(cut["last_effective_receive"] <= selected["D"] for cut in receipt["cuts"]),
        "all16 byte cuts throughD required",
    )
    keys, scans = selection_inputs(bound)
    require(selected == select(scans, keys, pilot.session_bounds(DAY)[0]), "selected origin/carrier is not frozen deterministic first source case")
    validate_cuts(receipt["cuts"], scans, selected["D"])
    require(receipt["proofs"] == key_proofs(bound["full_outputs"], selected["D"]), "original key-only/schema proofs changed")
    require(
        read_yaml(check_record(receipt["config"])) == reader.prefix_config(read_yaml(check_record(bound["native_job"]["config"])), OUTPUT / "raw"),
        "cut config changed beyond raw source directory",
    )
    require(
        receipt["original_config"] == bound["native_job"]["config"] and receipt["environment"] == bound["native_job"]["environment"], "original config/runtime ownership changed"
    )
    work = OUTPUT / "truncated"
    expected_command = [
        bound["binary"]["path"],
        "-d",
        DAY,
        "-C",
        str(work),
        "--trading-calendar",
        str(pilot.CALENDAR),
        "--run-status-dir",
        str(work / "status"),
        str(work / "config.yaml"),
    ]
    require(receipt["command"] == expected_command and receipt["prefix_outputs"] == [proof["prefix_path"] for proof in receipt["proofs"]], "prefix command/output path changed")
    require_closure(receipt, "sources", method["sources"])
    require_closure(
        receipt,
        "inputs",
        [
            *method["inputs"],
            record(selection_path),
            record(Path(selection_path).with_suffix(Path(selection_path).suffix + ".identity")),
            receipt["config"],
            receipt["original_config"],
            *[record(cut["path"]) for cut in receipt["cuts"]],
            *bound["full_outputs"],
        ],
    )
    for scope in ("sources", "inputs"):
        for node in receipt[scope]:
            check_record(node)
    return method, bound, receipt


def bind_comparison(selection_path):
    selection_path = Path(selection_path).resolve()
    selection, bound, receipt = verify_preparation(selection_path)
    plan_path = OUTPUT / "bound-comparison.yaml"
    require(not plan_path.exists(), "fresh comparison bind required")
    plan = {
        "schema": "h16-real-receive-prefix-bound-comparison-v1",
        "contract": CONTRACT,
        "selection_method": record(selection_path),
        "selection_identity": selection["identity"],
        "preparation": record(OUTPUT / "preparation-receipt.yaml"),
        "preparation_identity": digest(receipt),
        "selected": receipt["selected"],
        "proofs": receipt["proofs"],
        "binary": bound["binary"],
        "native_values_read": False,
        "native_replays_launched": 0,
        "labels_read": False,
        "model_fits": 0,
    }
    write_yaml(plan_path, plan)
    return {
        "schema": COMPARISON_SCHEMA,
        "status": "draft",
        "plan": record(plan_path),
        "selection_identity": selection["identity"],
        "sources": selection["sources"],
        "inputs": unique_records([*receipt["inputs"], record(OUTPUT / "preparation-receipt.yaml"), record(plan_path)]),
    }


def verify_comparison(comparison_path, *, before_replay=False):
    method = canonical_method(comparison_path, COMPARISON_SCHEMA)
    plan_path = OUTPUT / "bound-comparison.yaml"
    require(method["plan"] == record(plan_path), "wrong frozen comparison plan")
    plan = read_yaml(plan_path)
    selection, bound, receipt = verify_preparation(plan["selection_method"]["path"])
    require(
        plan["schema"] == "h16-real-receive-prefix-bound-comparison-v1"
        and plan["contract"] == CONTRACT
        and plan["selection_method"] == receipt["selection_method"]
        and plan["selection_identity"] == method["selection_identity"] == selection["identity"]
        and plan["preparation"] == record(OUTPUT / "preparation-receipt.yaml")
        and plan["preparation_identity"] == digest(receipt)
        and plan["selected"] == receipt["selected"]
        and plan["proofs"] == receipt["proofs"]
        and plan["binary"] == bound["binary"]
        and plan["native_values_read"] is False
        and plan["labels_read"] is False
        and type(plan["model_fits"]) is int
        and plan["model_fits"] == 0,
        "frozen comparison key/ABI/parent/phase drift",
    )
    require_closure(method, "sources", selection["sources"])
    require_closure(method, "inputs", [*receipt["inputs"], record(OUTPUT / "preparation-receipt.yaml"), record(plan_path)])
    for scope in ("sources", "inputs"):
        for node in method[scope]:
            check_record(node)
    if before_replay:
        require(
            not (OUTPUT / "truncated" / "execution-receipt.yaml").exists() and not any(Path(path).exists() for path in receipt["prefix_outputs"]),
            "fresh prefix job required; no rerun or overwrite",
        )
    return method, plan, receipt


def validate_execution(receipt, identity):
    work = OUTPUT / "truncated"
    execution_path = work / "execution-receipt.yaml"
    execution = read_yaml(execution_path)
    require(
        execution.get("method_identity") == identity
        and execution.get("job") == "receive_prefix"
        and type(execution.get("exit_code")) is int
        and execution["exit_code"] == 0
        and execution.get("command") == receipt["command"],
        "prefix execution ownership/exit/command changed",
    )
    configured = receipt["prefix_outputs"]
    present = [path for path in configured if Path(path).is_file()]
    require([node["path"] for node in execution["outputs"]] == present and len(present) == len(set(present)), "executed native output presence/order changed")
    require(execution["log"]["path"] == str(work / "native.log"), "different prefix native log")
    nodes = [record(execution_path), execution["log"], *execution["outputs"]]
    status_path = work / "status" / f"{DAY}.yaml"
    status = read_yaml(status_path)
    require(status.get("status") == "completed" and status.get("fatal_error") is False, "prefix native completion status required")
    nodes += [record(status_path)]
    for node in nodes:
        check_record(node)
    return execution, nodes


def absence_allowed(proof, present):
    require(
        present or proof["absence_allowed"] is True and type(proof["expected_common_rows"]) is int and proof["expected_common_rows"] == 0,
        "missing nonempty original native periodic prefix",
    )
    return not present


def compare_tables(full, truncated, proof, cutoff):
    require(
        schema_identity(full.schema) == schema_identity(truncated.schema) == proof["schema_identity"] and full.schema.equals(truncated.schema, check_metadata=True),
        "native schema/metadata mismatch",
    )
    expected_keys = [key for key in reader.native_keys(full) if key[0] <= cutoff]
    require(len(expected_keys) == proof["expected_common_rows"] and digest(expected_keys) == proof["expected_common_keys_identity"], "frozen common-key population drift")
    result = reader.compare_tables(full, truncated, cutoff)
    require(set(result["field_bit_checks"]) == set(pilot.COLUMNS), "all36 fields must be compared")
    return result


def check(comparison_path):
    comparison_path = Path(comparison_path).resolve()
    method, plan, receipt = verify_comparison(comparison_path)
    destination = OUTPUT / "validation.yaml"
    require(not destination.exists(), "fresh prefix validation receipt required")
    execution, execution_nodes = validate_execution(receipt, method["identity"])
    executed = {node["path"]: node for node in execution["outputs"]}
    checks = []
    for proof in plan["proofs"]:
        present = Path(proof["prefix_path"]).is_file()
        empty = absence_allowed(proof, present)
        result = {"symbol": proof["symbol"], "prefix_present": present, "common_rows": proof["expected_common_rows"], "absence_proven_from_original_keys": empty}
        if not empty:
            require(executed.get(proof["prefix_path"]) == record(proof["prefix_path"]), "native prefix artifact not bound by actual execution")
            full = pq.read_table(proof["full"]["path"], columns=list(pilot.COLUMNS), use_threads=False)
            truncated = pq.read_table(proof["prefix_path"], columns=list(pilot.COLUMNS), use_threads=False)
            pilot.validate_table(full)
            pilot.validate_table(truncated)
            result.update(compare_tables(full, truncated, proof, plan["selected"]["D"]))
            if proof["symbol"] == TARGET:
                keys = reader.native_keys(truncated)
                origin = tuple(plan["selected"]["origin_keys"][name] for name in KEYS)
                require(origin in keys and origin[0] == plan["selected"]["S"], "selected original S mandatory at same exact key")
                result["selected_common_origin_present"] = True
        checks.append(result)
    require(len(checks) == len(CARRIERS) and any(node.get("selected_common_origin_present") is True for node in checks), "all16 cells/mandatory original S proof incomplete")
    result = {
        "schema": "h16-real-receive-prefix-validation-v1",
        "passed": True,
        "comparison_method_identity": method["identity"],
        "selection_method_identity": plan["selection_identity"],
        "producer_identity": receipt["producer_identity"],
        "support_identity": receipt["support_identity"],
        "binary_sha256": receipt["binary"]["sha256"],
        "selected": plan["selected"],
        "genuine_straddle_found": True,
        "common_origin_present": True,
        "all_common_origin_bits_exact": True,
        "all16_original_streams_preserved": True,
        "fields_per_present_cell": len(pilot.COLUMNS),
        "checks": checks,
        "sources": method["sources"],
        "inputs": unique_records([*method["inputs"], *execution_nodes, record(comparison_path), record(comparison_path.with_suffix(comparison_path.suffix + ".identity"))]),
        "existing_complete_spools_reused": True,
        "labels_read": False,
        "model_fits": 0,
        "generated_events": 0,
        "python_features_generated": False,
        "native_replays_launched": 0,
        "predictive_admission": False,
    }
    write_yaml(destination, result)
    return {"passed": True, "receipt": str(destination), "cells": len(checks), "fields_per_cell": len(pilot.COLUMNS), "predictive_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bind-selection", type=Path, metavar="FROZEN_H16_PRODUCER")
    mode.add_argument("--prepare", type=Path, metavar="FROZEN_SELECTION")
    mode.add_argument("--bind-comparison", type=Path, metavar="FROZEN_SELECTION")
    mode.add_argument("--preflight", type=Path, metavar="FROZEN_COMPARISON")
    mode.add_argument("--check", type=Path, metavar="FROZEN_COMPARISON")
    parser.add_argument("--support", type=Path, metavar="FROZEN_H16_SUPPORT")
    parser.add_argument("--support-receipt", type=Path, metavar="ACTUAL_H16_SUPPORT_RECEIPT")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bind_selection:
        if not args.support:
            parser.error("--bind-selection requires --support")
        payload = bind_selection(args.bind_selection, args.support, args.support_receipt)
        path = args.output or OUTPUT / "draft-selection-method.yaml"
        write_yaml(path, payload)
        result = {"selection_payload_path": str(path), "headers_keys_values_read": False}
    elif args.prepare:
        result = prepare(args.prepare)
    elif args.bind_comparison:
        payload = bind_comparison(args.bind_comparison)
        path = args.output or OUTPUT / "draft-comparison-method.yaml"
        write_yaml(path, payload)
        result = {"comparison_payload_path": str(path), "native_values_read": False}
    elif args.preflight:
        method, _, _ = verify_comparison(args.preflight, before_replay=True)
        result = {"preflight_passed": True, "method_identity": method["identity"], "native_replays_launched": 0}
    else:
        result = check(args.check)
    print(result)


if __name__ == "__main__":
    main()
