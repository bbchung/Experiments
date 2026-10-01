"""Prepare a real native BIN receive-prefix causal replay; never launch coco."""

from __future__ import annotations

import bisect
import copy
import ctypes as ct
import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
import zstandard as zstd

OUTPUT = Path(__file__).resolve().parent
DAY, TARGET = "20260119", "2308"
DESTINATION = OUTPUT / "receive-prefix"


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


def read(path):
    return yaml.load(path.read_text(), Loader=yaml.CSafeLoader)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def records(path, symbol):
    if sys.byteorder != "little" or [ct.sizeof(item) for item in (Header, Book, Trade)] != [24, 152, 56]:
        raise RuntimeError("Unsupported native ABI; never reinterpret a different packed source")
    previous, ordinal = 0, 0
    with path.open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as stream:
        while header_bytes := stream.read(ct.sizeof(Header)):
            if len(header_bytes) != ct.sizeof(Header):
                raise RuntimeError("Truncated native header")
            header = Header.from_buffer_copy(header_bytes)
            payload_type = Book if header.type == b"B" else Trade if header.type == b"T" else None
            if payload_type is None:
                raise RuntimeError("Unsupported native message type")
            payload_bytes = stream.read(ct.sizeof(payload_type))
            if len(payload_bytes) != ct.sizeof(payload_type):
                raise RuntimeError("Truncated native payload")
            payload = payload_type.from_buffer_copy(payload_bytes)
            if payload.symbol.decode() != symbol:
                raise RuntimeError("Native ABI or declared symbol mismatch")
            received = max(header.time, previous)
            previous = received
            ordinal += 1
            yield {
                "ordinal": ordinal,
                "seq": header.seq,
                "type": header.type.decode(),
                "raw_receive": header.time,
                "receive": received,
                "exchange": payload.exchange if payload.exchange > 0 else received,
                "status": payload.status,
                "bytes": header_bytes + payload_bytes,
            }


def evidence(row):
    return {key: value for key, value in row.items() if key != "bytes"}


def main():
    if DESTINATION.exists():
        raise RuntimeError("Fresh receive-prefix preparation required; no overwrite")
    plan = read(OUTPUT / "full-preparation-receipt.yaml")
    pilot_plan = read(OUTPUT / "preparation-receipt.yaml")
    pilot = next(job for job in pilot_plan["pilot_jobs"] if (job["day"], job["symbol"]) == (DAY, TARGET))
    full_native = Path(pilot["work"]) / "data" / DAY / TARGET / "values.parquet"
    full = pq.read_table(
        full_native, columns=["SampleTime", "SampleBookTime", "SampleBookSeq", "DemandRepairRenewal.0.phase.0", "H10_ordered_processed_cluster_exchange_time"], use_threads=False
    )
    sample_times = full["SampleTime"].to_numpy()
    phases = full["DemandRepairRenewal.0.phase.0"].to_numpy()
    processed = full["H10_ordered_processed_cluster_exchange_time"].to_numpy()
    sources = {item["symbol"]: item for item in plan["raw_inputs"] if item["day"] == DAY}
    if set(sources) != set(plan["universe_order"]) or any(not item["path"].endswith(".bin.zst") for item in sources.values()):
        raise RuntimeError("Exact sixteen native BIN sources required")
    candidates, previous, has_book, continuous = [], None, False, True
    for row in records(Path(sources[TARGET]["path"]), TARGET):
        if previous and row["exchange"] != previous["exchange"]:
            index = bisect.bisect_right(sample_times, previous["receive"])
            while index < len(full) and sample_times[index] < row["receive"]:
                sample = int(sample_times[index])
                if (
                    sample % 30_000_000 == 0
                    and has_book
                    and continuous
                    and phases[index] in ("contained", "repaired", "renewed", "resumed")
                    and processed[index] < previous["exchange"]
                ):
                    candidates.append(
                        {
                            "index": index,
                            "S": sample,
                            "R": previous["receive"],
                            "A": row["receive"],
                            "pending_exchange": previous["exchange"],
                            "target_last": evidence(previous),
                            "target_next_distinct": evidence(row),
                        }
                    )
                index += 1
            has_book, continuous = False, True
        has_book |= row["type"] == "B"
        continuous &= row["status"] & 7 == 0
        previous = row
    chosen, carrier = None, None
    for candidate in candidates:
        for symbol in plan["universe_order"]:
            if symbol == TARGET:
                continue
            for row in records(Path(sources[symbol]["path"]), symbol):
                if row["receive"] >= candidate["A"]:
                    break
                if row["receive"] >= candidate["S"] and (carrier is None or row["receive"] < carrier["receive"]):
                    carrier = {"symbol": symbol, **evidence(row)}
                    break
        if carrier is not None:
            chosen = candidate
            break
    DESTINATION.mkdir()
    if chosen is None:
        (DESTINATION / "preparation-receipt.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "native-real-receive-prefix-preparation-v1",
                    "genuine_straddle_found": False,
                    "candidate_count": len(candidates),
                    "native_replays_launched": 0,
                    "generated_events": 0,
                }
            )
        )
        print("NO_GENUINE_RECEIVE_STRADDLE")
        return
    cutoff = carrier["receive"]
    assert chosen["R"] < chosen["S"] <= cutoff < chosen["A"]
    cuts = []
    for symbol in plan["universe_order"]:
        original = Path(sources[symbol]["path"])
        if digest(original) != sources[symbol]["sha256"]:
            raise RuntimeError("Original native source changed since full prep")
        output = DESTINATION / "raw" / symbol / (DAY + ".bin.zst")
        output.parent.mkdir(parents=True)
        count, retained_bytes, prefix_hash, last = 0, 0, hashlib.sha256(), None
        with output.open("wb") as sink, zstd.ZstdCompressor(level=3).stream_writer(sink) as writer:
            for row in records(original, symbol):
                if row["receive"] > cutoff:
                    break
                writer.write(row["bytes"])
                prefix_hash.update(row["bytes"])
                count += 1
                retained_bytes += len(row["bytes"])
                last = evidence(row)
        if not count:
            raise RuntimeError("Missing clock-carrier prefix; cannot synthesize a row")
        cuts.append(
            {
                "symbol": symbol,
                "source": sources[symbol],
                "path": str(output),
                "sha256": digest(output),
                "records": count,
                "uncompressed_prefix_bytes": retained_bytes,
                "uncompressed_prefix_sha256": prefix_hash.hexdigest(),
                "last": last,
            }
        )
    config = copy.deepcopy(read(Path(pilot["work"]) / "config.yaml"))
    global_group = next(group for group in config["Modules"] if group["Gid"] == "")
    md = next(item for item in global_group["Decl"] if item["Desc"] == "TradeBookMd.0")
    md["Spec"]["Dirs"] = [str(DESTINATION / "raw")]
    work = DESTINATION / "truncated"
    work.mkdir()
    config_path = work / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    index = chosen.pop("index")
    receipt = {
        "schema": "native-real-receive-prefix-preparation-v1",
        "genuine_straddle_found": True,
        "selection": "earliest actual fixed30s origin with active published state, continuous pending book cluster and real other-symbol receive before closure; no labels/performance",
        "day": DAY,
        "symbol": TARGET,
        **chosen,
        "D": cutoff,
        "carrier": carrier,
        "origin_keys": {name: int(full[name][index].as_py()) for name in ("SampleTime", "SampleBookTime", "SampleBookSeq")},
        "source_full_native": {"path": str(full_native), "sha256": digest(full_native)},
        "full_plan_sha256": digest(OUTPUT / "full-preparation-receipt.yaml"),
        "script_sha256": digest(Path(__file__)),
        "config": {"path": str(config_path), "sha256": digest(config_path)},
        "cuts": cuts,
        "abi": {
            "byte_order": sys.byteorder,
            "header_bytes": 24,
            "book_payload_bytes": 152,
            "trade_payload_bytes": 56,
            "header_time_offset": Header.time.offset,
            "payload_exchange_offset": Book.exchange.offset,
        },
        "binary": plan["binary"],
        "calendar": plan["calendar"],
        "runtime_snapshot": plan["runtime_snapshot"],
        "source_files": [
            {"path": str(OUTPUT.parents[3] / name), "sha256": digest(OUTPUT.parents[3] / name)}
            for name in ("src/msg/md_msg.h", "src/sdk/types/coco_type.h", "src/oms/modules/md/trade_book_md/trade_book_md.cpp", "src/oms/modules/md/trade_book_md/trade_book_md.h")
        ],
        "command": [plan["binary"]["path"], "-d", DAY, "-C", str(work), "--trading-calendar", plan["calendar"]["path"], "--run-status-dir", str(work / "status"), str(config_path)],
        "environment": {"LD_LIBRARY_PATH": str(OUTPUT / "runtime"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        "labels_read": False,
        "generated_events": 0,
        "native_replays_launched": 0,
        "required_after_root_replay": "Every common full/truncated exact5key origin throughD must retain all H10 numeric/phase/diagnostic and physicalmid bits; S must exist unchanged and EOF cannot close pending_exchange beforeA",
    }
    (DESTINATION / "preparation-receipt.yaml").write_text(yaml.safe_dump(receipt, sort_keys=False))
    print("REAL_BIN_PREFIX_PREPARED", "R<S<=D<A", chosen["R"], chosen["S"], cutoff, chosen["A"], "carrier", carrier["symbol"], "records", sum(item["records"] for item in cuts))


if __name__ == "__main__":
    main()
