"""Frozen native pair features and receive-causal quote mechanism study; no learner.

Prepare reads metadata and hashes recordings, pilot checks technical contracts,
export materializes the preregistered panel, and analyze consumes frozen outputs.
"""

from __future__ import annotations

import sys as _astra_sys
from pathlib import Path as _AstraPath

_astra_repo_root = next(parent.parent for parent in _AstraPath(__file__).resolve().parents if parent.name == "AstraResearch")
_astra_sys.path.insert(0, str(_astra_repo_root))

import argparse
import copy
import json
import math
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml
import zstandard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import venue_transfer as quotes

from AstraResearch.code_guard import source_files
from AstraResearch.engine_identity import identity
from AstraResearch.Experiments.instruments import stock_futures
from AstraResearch.io import ContractError, file_hash, file_stamp, lock, read_yaml, write_yaml
from AstraResearch.native_contract import sample_summary, validate_receipt
from AstraResearch.store import ExposureLedger

FIELDS = ("basis_bps", "basis_deviation_bps", "target_age_s", "reference_age_s", "target_latency_s", "reference_latency_s", "reference_spread_bps", "basis_history_fraction")
FEATURES = {field: f"CrossMarketBasis.1.{field}.0" for field in FIELDS}
KEYS = ["SampleTime", "SampleBookTime", "SampleBookSeq"]
OUTPUT = ROOT / "runs/research/basis-mechanism-20260926"


def path(value):
    return (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value)


def run_native(binary, day, work, config, calendar):
    work.mkdir(parents=True, exist_ok=False)
    config_path = work / "config.yaml"
    write_yaml(config_path, config)
    command = [str(binary), "-d", day, "-C", str(work), "--quiet", "--run-status-dir", str(work / "native-status"), "--trading-calendar", calendar, str(config_path)]
    with (work / "process.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=False)
    if result.returncode:
        raise ContractError(f"Native process failed ({result.returncode}): {work}")
    validate_receipt(read_yaml(work / "native-status" / f"{day}.yaml"), day)


def metadata(config, template, day, output):
    pairs = [config["scope"]["primary_pair"], *config["scope"]["fixed_comparators"]]
    work = output / "metadata" / day
    declarations = [copy.deepcopy(template["Modules"][0]["Decl"][0])]
    for i, pair in enumerate(pairs):
        declarations.append(
            {
                "Desc": f"TaifexInfo.{i}",
                "Spec": {"Product": pair["product"], "TickLadder": "stock_future", "PointValue": 2000, "BasicInfo": config["mapping"]["contract_root"] + "/taifex/futures"},
            }
        )
    declarations.append(
        {
            "Desc": "MarketInfoWriter.0",
            "Spec": {"OutputPath": str(work / "contracts.yaml"), "Contracts": [name for pair in pairs for name in (pair["product"] + "@1", pair["stock"])]},
        }
    )
    run_native(path(config["native_engine"]["binary"]), day, work, {"Modules": [{"Gid": "", "Decl": declarations}]}, config["dates"]["calendar"])
    return yaml.load((work / "contracts.yaml").read_text(), Loader=yaml.BaseLoader)


def render(template, row, work):
    text = yaml.safe_dump(template, sort_keys=False)
    text = text.replace("CDF@1", row["product"] + "@1").replace("CDFF6", row["contract"]).replace("CDF", row["product"]).replace("2330", row["stock"])
    config = yaml.safe_load(text)
    for group in config["Modules"]:
        for declaration in group["Decl"]:
            if declaration["Desc"].startswith("TradeBookMd."):
                declaration["Spec"]["Dirs"] = [str(work / "input")]
    return config


def prepare(registration, output):
    config = read_yaml(registration)
    if config["dates"]["protected_from"] != "20260828" or config["dates"]["validation"][1] != "20260826":
        raise ContractError("Unexpected study date boundary")
    if output.exists():
        raise ContractError("Preparation requires a new output directory")
    output.mkdir(parents=True)
    template_path = registration.parent / config["native_export"]["config_template"]
    template = read_yaml(template_path)
    for key, expected in [("binary", "binary_sha256"), ("source_patch", "source_patch_sha256")]:
        if file_hash(path(config["native_engine"][key])) != config["native_engine"][expected]:
            raise ContractError("Native freeze mismatch: " + key)
    if file_hash(Path(config["dates"]["calendar"])) != config["dates"]["calendar_sha256"]:
        raise ContractError("Calendar changed")
    calendar = read_yaml(Path(config["dates"]["calendar"]))
    days = [d for d in calendar["trading_days"] if "20260601" <= d <= "20260826" and d not in calendar["unscheduled_closures"]]
    if len(days) != config["dates"]["expected_planned_days"]:
        raise ContractError("Unexpected planned date count")
    write_yaml(output / "registration.yaml", config)
    write_yaml(output / "native-template.yaml", template)
    write_yaml(output / "planned-dates.yaml", days)
    for source in source_files(ROOT):
        target = output / "source/AstraResearch" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    instruments_source = Path(sys.modules["AstraResearch.Experiments.instruments"].__file__)
    instruments_target = output / "source/AstraResearch/Experiments/instruments.py"
    instruments_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(instruments_source, instruments_target)
    for source in [Path(__file__), Path(quotes.__file__), ROOT / "engine_identity.py"]:
        shutil.copy2(source, output / "source" / source.name)
    shutil.copy2(path(config["native_engine"]["source_manifest"]), output / "native-source-manifest.json")
    shutil.copy2(path(config["native_engine"]["source_patch"]), output / "native-source.patch")
    write_yaml(
        output / "environment.yaml",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "engine": identity(path(config["native_engine"]["binary"])),
        },
    )
    pairs = [config["scope"]["primary_pair"], *config["scope"]["fixed_comparators"]]
    ledger = ExposureLedger(ROOT / "runs/.cache/dataset")
    for scope in ["TWSE", *("TAIFEX:" + pair["product"] for pair in pairs)]:
        ledger.claim(scope, days, "development", config["study"])
    write_yaml(output / "exposure.yaml", {"study": config["study"], "days": days, "role": "development", "scopes": ["TWSE", *("TAIFEX:" + pair["product"] for pair in pairs)]})
    inventory, metadata_hashes = [], {}
    for i, day in enumerate(days):
        native = metadata(config, template, day, output)
        mapping = {r["product"]: r for r in stock_futures(Path(config["mapping"]["contract_root"]), day)}
        meta_paths = [Path(config["mapping"]["contract_root"]) / exchange / kind / f"{day}.csv" for exchange, kind in [("tse", "stock"), ("otc", "stock"), ("taifex", "futures")]]
        meta_paths += [Path("/mnt/data0/info/tse/stock") / name / f"{day}.csv" for name in ["twt84u", "twtb4u", "disposition"]]
        for source in meta_paths:
            metadata_hashes[str(source)] = file_hash(source) if source.is_file() else None
        for pair in pairs:
            product, stock = pair["product"], pair["stock"]
            future, spot = native["contracts"][product + "@1"], native["contracts"][stock]
            relation = mapping.get(product, {})
            valid = (
                future["available"] == "true"
                and spot["available"] == "true"
                and relation.get("underlying") == stock
                and relation.get("rule") in config["mapping"]["permitted_relationships"]
                and relation.get("shares_per_contract") == 2000
            )
            if valid and (future["tick_ladder"] != "stock_future" or float(future["trading_unit"]) != 2000):
                raise ContractError("Native SSF coordinate mismatch")
            row = {
                "day": day,
                **pair,
                "contract": future.get("symbol"),
                "mapping": relation,
                "native_future": future,
                "native_stock": spot,
                "available": valid,
                "missing": [],
                "sources": {},
            }
            for leg, symbol, directory in [("target", future.get("symbol"), config["mapping"]["future_root"]), ("reference", stock, config["mapping"]["stock_root"])]:
                source = Path(directory) / str(symbol) / f"{day}.csv.zst"
                row["sources"][leg] = {
                    "path": str(source),
                    "sha256": file_hash(source) if source.is_file() else None,
                    "stamp": list(file_stamp(source)) if source.is_file() else None,
                }
                if not source.is_file():
                    row["available"] = False
                    row["missing"].append(leg + "_csv_missing")
            if not valid:
                row["missing"].append("metadata_ineligible")
            inventory.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(days):
            print(f"metadata prepared {i + 1}/{len(days)}", flush=True)
    write_yaml(output / "inventory.yaml", inventory)
    files = [p for p in (output / "source").rglob("*") if p.is_file()]
    files += [
        output / n
        for n in [
            "registration.yaml",
            "native-template.yaml",
            "planned-dates.yaml",
            "environment.yaml",
            "inventory.yaml",
            "exposure.yaml",
            "native-source-manifest.json",
            "native-source.patch",
        ]
    ]
    write_yaml(
        output / "freeze.yaml",
        {
            "files": {str(p.relative_to(output)): file_hash(p) for p in sorted(files)},
            "metadata_hashes": metadata_hashes,
            "native_binary_sha256": config["native_engine"]["binary_sha256"],
        },
    )
    write_yaml(
        output / "status.yaml",
        {
            "state": "prepared",
            "planned_pair_days": len(inventory),
            "available_pair_days": sum(r["available"] for r in inventory),
            "missing": [{k: r[k] for k in ["day", "product", "missing"]} for r in inventory if not r["available"]],
        },
    )
    print(json.dumps(read_yaml(output / "status.yaml"), indent=2), flush=True)


def verify_freeze(output):
    frozen_source = (output / "source").resolve()
    amendment_path = output / "amendments/001-empty-flip-traces.yaml"
    if amendment_path.is_file():
        amendment = read_yaml(amendment_path)
        if file_hash(output / "freeze.yaml") != amendment["original_freeze_sha256"]:
            raise ContractError("Original freeze changed after the recorded correction")
        frozen_source = (output / amendment["source_directory"]).resolve()
        for name, expected in amendment["files"].items():
            if file_hash(output / name) != expected:
                raise ContractError("Amended implementation changed: " + name)
    if Path(__file__).resolve() != frozen_source / "basis_mechanism.py":
        raise ContractError("Execute the frozen output/source/basis_mechanism.py for all non-prepare stages")
    for name in ["AstraResearch.io", "AstraResearch.Experiments.instruments", "AstraResearch.store", "AstraResearch.native_contract", "engine_identity", "venue_transfer"]:
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(frozen_source):
            raise ContractError("Active dependency is outside the frozen source: " + name)
    freeze = read_yaml(output / "freeze.yaml")
    for name, expected in freeze["files"].items():
        if file_hash(output / name) != expected:
            raise ContractError("Frozen implementation/input changed: " + name)
    for name, expected in freeze["metadata_hashes"].items():
        source = Path(name)
        if (file_hash(source) if source.is_file() else None) != expected:
            raise ContractError("Metadata changed: " + name)
    config = read_yaml(output / "registration.yaml")
    if file_hash(path(config["native_engine"]["binary"])) != freeze["native_binary_sha256"]:
        raise ContractError("Frozen native binary changed")
    if identity(path(config["native_engine"]["binary"]))["fingerprint"] != read_yaml(output / "environment.yaml")["engine"]["fingerprint"]:
        raise ContractError("Frozen native runtime dependency changed")
    if file_hash(Path(config["dates"]["calendar"])) != config["dates"]["calendar_sha256"]:
        raise ContractError("External native calendar changed")
    return config


def read_flip_trace(work, directory, day):
    """A completed zero-capture native trace is empty; a missing artifact is not."""
    trace = work / directory
    parquet = trace / "keys.parquet"
    if parquet.is_file():
        frame = pd.read_parquet(parquet)
        if set(frame) != set(KEYS):
            raise ContractError("Unexpected flip-trace schema: " + str(parquet))
        return frame
    validate_receipt(read_yaml(work / "native-status" / f"{day}.yaml"), day)
    summary = sample_summary(trace / "sample_summary.yaml")
    if any(summary[name] != 0 for name in ["sampled_rows", "emitted_rows", "pending_rows", "emitted_unresolved_rows", "unresolved_label_cells"]):
        raise ContractError("Missing nonempty native flip trace: " + str(trace))
    expected_sampler = {
        "Mode": "price_flip",
        "QtyChangeRatio": 0.25,
        "QtyChangeMin": 0,
        "MaxSpreadTick": 0,
        "MinLimitPos": 0,
        "MaxLimitPos": 1,
        "MinTotalVolume": 0,
        "MinInterval": "0s",
        "MinIntervalClock": "last_accepted",
        "StartTime": "091000",
        "UntilTime": "130000",
    }
    if read_yaml(trace / "sampler_contract.yaml") != expected_sampler or read_yaml(trace / "feature_semantic_manifest.yaml") != []:
        raise ContractError("Missing flip trace lacks the registered native semantics: " + str(trace))
    return pd.DataFrame({key: pd.Series(dtype=np.int64) for key in KEYS})


def read_raw(source):
    source_path = Path(source["path"])
    if list(file_stamp(source_path)) != source["stamp"] or file_hash(source_path) != source["sha256"]:
        raise ContractError("Frozen raw recording changed")
    columns = quotes.RAW_COLUMNS + ["Price", "TradeVolume", "BidPrice2", "AskPrice2", "BidVol2", "AskVol2"]
    types = {name: pa.string() if name in {"Type", "StatusMask"} else pa.float64() if "Price" in name else pa.int64() for name in columns}
    with source_path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decompressed:
        frame = pacsv.read_csv(
            decompressed, read_options=pacsv.ReadOptions(use_threads=False), convert_options=pacsv.ConvertOptions(include_columns=columns, column_types=types)
        ).to_pandas()
    if not frame.Type.isin(["B", "T"]).all():
        raise ContractError("Unexpected raw event type")
    return frame


def current_book_source(raw, *, stock=False, upper=math.nan, lower=math.nan):
    """Independent CurrentBook.mid source tape, deliberately not executable quotes."""
    times, values, book_times, book_seqs = [], [], [], []
    halted, latest_book, latest_seq, clock = True, 0, 0, 0
    for r in raw.itertuples(index=False):
        clock = max(clock, int(r.Timestamp))
        if r.Type == "T" and ((not stock and not math.isfinite(r.Price)) or not r.Price > 0 or r.TradeVolume <= 0):
            continue
        status = int(str(r.StatusMask), 16)
        halt_now = bool(status & 5)
        if halt_now and not halted:
            times.append(clock)
            values.append(math.nan)
            book_times.append(latest_book)
            book_seqs.append(latest_seq)
        halted = halt_now
        if r.Type != "B":
            continue
        bid, ask = float(r.BidPrice1), float(r.AskPrice1)
        bid_volume, ask_volume = r.BidVol1, r.AskVol1
        if stock and (status & 7) == 0:
            if r.BidDepth > 0 and bid == 0:
                bid = upper
                if r.BidDepth > 1 and getattr(r, "BidPrice2", math.nan) == upper:
                    bid_volume += r.BidVol2
            if r.AskDepth > 0 and ask == 0:
                ask = lower
                if r.AskDepth > 1 and getattr(r, "AskPrice2", math.nan) == lower:
                    ask_volume += r.AskVol2
        b = r.BidDepth > 0 and bid > 0 and bid_volume > 0
        a = r.AskDepth > 0 and ask > 0 and ask_volume > 0
        mid = (bid + ask) / 2 if b and a else bid if b else ask if a else math.nan
        latest_book, latest_seq = clock, latest_seq + 1
        times.append(clock)
        values.append(mid)
        book_times.append(latest_book)
        book_seqs.append(latest_seq)
    return tuple(np.asarray(v, dtype=np.int64 if i != 1 else float) for i, v in enumerate([times, values, book_times, book_seqs]))


def values_at(source, query, *, inclusive):
    times, values, *_ = source
    indices = np.searchsorted(times, query, side="right" if inclusive else "left") - 1
    result = np.full(len(query), np.nan)
    valid = indices >= 0
    result[valid] = values[indices[valid]]
    return result, indices


def check_labels(frame, target, reference):
    origins = frame.SampleTime.to_numpy(np.int64)
    result = {}
    for name, source in [("OriginMidPrice", target), ("ReferenceMidPrice", reference)]:
        expected, _ = values_at(source, origins, inclusive=False)
        actual = frame[name].to_numpy(float)
        finite = np.isfinite(expected) & np.isfinite(actual)
        mismatch = (np.isfinite(expected) != np.isfinite(actual)) | (finite & (actual != expected))
        result[name] = {"checked": len(frame), "mismatches": int(mismatch.sum())}
    _, origin_idx = values_at(target, origins, inclusive=False)
    for name, field in [("SampleBookTime", 2), ("SampleBookSeq", 3)]:
        valid = origin_idx >= 0
        expected = np.zeros(len(frame), dtype=np.int64)
        expected[valid] = target[field][origin_idx[valid]]
        result[name] = {"checked": len(frame), "mismatches": int((frame[name].to_numpy(np.int64) != expected).sum())}
    y0 = frame.OriginMidPrice.to_numpy(float)
    for horizon in [1, 5, 30]:
        deadline = origins + horizon * 1000000
        y1, _ = values_at(target, deadline, inclusive=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected = 10000 * (y1 - y0) / y0
        expected[(y0 <= 0) | (deadline >= target[0][-1] if len(target[0]) else np.ones(len(deadline), dtype=bool))] = np.nan
        actual = frame[f"ssf_mid_return_bps[{horizon}s]"].to_numpy(float)
        mismatch = (np.isfinite(expected) != np.isfinite(actual)) | (np.isfinite(expected) & ~np.isclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True))
        finite_both = np.isfinite(actual) & np.isfinite(expected)
        result[f"label_{horizon}s"] = {
            "checked": len(frame),
            "known_native": int(np.isfinite(actual).sum()),
            "mismatches": int(mismatch.sum()),
            "maximum_finite_error": float(np.max(np.abs(actual[finite_both] - expected[finite_both]))) if finite_both.any() else None,
        }
    result["passed"] = all(r["mismatches"] == 0 for r in result.values())
    return result


def export_job(row, config, template, output, *, repeat=False):
    if not row["available"]:
        return {"day": row["day"], "product": row["product"], "status": "missing", "reasons": row["missing"]}
    key = row["day"] + "-" + row["product"]
    work = output / ("pilot-repeat" if repeat else "native") / key
    if work.exists() and not (work / "receipt.yaml").is_file():
        raise ContractError("Incomplete native attempt retained; explicit correction required: " + str(work))
    # Build the private source root beside the future work directory; no binary
    # recording or global fallback can be selected by the native reader.
    inputs = output / "inputs" / key
    for leg, symbol in [("target", row["contract"]), ("reference", row["stock"])]:
        source = row["sources"][leg]
        if file_hash(Path(source["path"])) != source["sha256"]:
            raise ContractError("Raw recording changed: " + source["path"])
        link = inputs / symbol / f"{row['day']}.csv.zst"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(source["path"])
        if not link.is_symlink() or link.resolve() != Path(source["path"]).resolve():
            raise ContractError("Private source link changed: " + str(link))
    native = render(template, row, work)
    for declaration in native["Modules"][0]["Decl"]:
        if declaration["Desc"].startswith("TradeBookMd."):
            declaration["Spec"]["Dirs"] = [str(inputs)]
    if (work / "receipt.yaml").is_file():
        receipt = read_yaml(work / "receipt.yaml")
        if read_yaml(work / "config.yaml") != native or receipt.get("sources") != row["sources"]:
            raise ContractError("Resumed native identity changed: " + str(work))
        for name, expected in receipt["files"].items():
            if file_hash(work / name) != expected:
                raise ContractError("Resumed native artifact changed: " + str(work / name))
        return receipt
    run_native(path(config["native_engine"]["binary"]), row["day"], work, native, config["dates"]["calendar"])
    frame = pd.read_parquet(work / "grid/values.parquet")
    required = {*KEYS, *FEATURES.values(), "OriginMidPrice", "ReferenceMidPrice", "TargetBidTicks", "TargetAskTicks", *(f"ssf_mid_return_bps[{h}s]" for h in [1, 5, 30])}
    if set(frame) != required:
        raise ContractError("Unexpected native grid schema: " + str(sorted(set(frame) ^ required)))
    summaries = {}
    for directory, filename in [("grid", "values.parquet"), ("target-flips", "keys.parquet"), ("reference-flips", "keys.parquet")]:
        summary = sample_summary(work / directory / "sample_summary.yaml")
        count = pq.read_metadata(work / directory / filename).num_rows if directory == "grid" else len(read_flip_trace(work, directory, row["day"]))
        if summary["schema_version"] != 2 or summary["sampled_rows"] != count or summary["emitted_rows"] != count or summary["pending_rows"] != 0:
            raise ContractError("Native origin accounting mismatch: " + str(work / directory))
        summaries[directory] = summary
    if frame[KEYS].duplicated().any() or (frame.SampleBookTime > frame.SampleTime).any():
        raise ContractError("Invalid native decision keys")
    raw_target, raw_reference = read_raw(row["sources"]["target"]), read_raw(row["sources"]["reference"])
    target = current_book_source(raw_target)
    reference = current_book_source(raw_reference, stock=True, upper=float(row["native_stock"]["limit_up"]), lower=float(row["native_stock"]["limit_down"]))
    parity = check_labels(frame, target, reference)
    write_yaml(work / "source-parity.yaml", parity)
    if not parity["passed"]:
        raise ContractError("Native/source parity failed: " + str(work))
    receipt = {
        "day": row["day"],
        "product": row["product"],
        "status": "complete",
        "rows": len(frame),
        "sources": row["sources"],
        "parity": parity,
        "sample_summaries": summaries,
        "files": {str(p.relative_to(work)): file_hash(p) for p in work.rglob("*") if p.is_file()},
    }
    write_yaml(work / "receipt.yaml", receipt)
    return receipt


def export(output, pilot):
    config = verify_freeze(output)
    inventory, template = read_yaml(output / "inventory.yaml"), read_yaml(output / "native-template.yaml")
    if pilot:
        rows = [r for r in inventory if r["day"] == "20260603"]
    else:
        if read_yaml(output / "pilot.yaml").get("passed") is not True:
            raise ContractError("Technical pilot must pass first")
        rows = inventory
    receipts = []
    with lock(output / "export.lock", blocking=False), ThreadPoolExecutor(max_workers=2) as pool:
        for row, result in zip(rows, pool.map(lambda row: export_job(row, config, template, output), rows), strict=True):
            receipts.append(result)
            print(f"{'pilot' if pilot else 'export'} {len(receipts)}/{len(rows)} {row['day']} {row['product']} {result['status']}", flush=True)
            write_yaml(
                output / "status.yaml", {"state": "exporting", "completed": len(receipts), "planned": len(rows), "last": {k: result[k] for k in ["day", "product", "status"]}}
            )
    if pilot:
        for row in rows:
            repeated = export_job(row, config, template, output, repeat=True)
            key = row["day"] + "-" + row["product"]
            a, b = pd.read_parquet(output / "native" / key / "grid/values.parquet"), pd.read_parquet(output / "pilot-repeat" / key / "grid/values.parquet")
            if list(a) != list(b) or not np.array_equal(a.to_numpy().view(np.uint64), b.to_numpy().view(np.uint64)):
                raise ContractError("Repeated native export is not bit-identical: " + key)
            if repeated["status"] != "complete":
                raise ContractError("Missing technical pilot pair")
        write_yaml(
            output / "pilot.yaml",
            {"passed": True, "days": ["20260603"], "pairs": [r["product"] for r in rows], "repeat": "all numeric feature/context/label bits identical", "receipts": receipts},
        )
    write_yaml(output / ("pilot-receipts.yaml" if pilot else "export-receipts.yaml"), receipts)
    write_yaml(output / "status.yaml", {"state": "pilot_complete" if pilot else "export_complete", "completed": len(receipts)})


def eligible(frame):
    return np.isfinite(frame[[FEATURES["basis_bps"], FEATURES["basis_deviation_bps"]]]).all(axis=1).to_numpy()


def make_intents(frame, calibration):
    """Shared union clock is fixed by current features before any outcome masks."""
    good = eligible(frame)
    r = frame[FEATURES["basis_bps"]].to_numpy() - calibration["raw_center"]
    d = frame[FEATURES["basis_deviation_bps"]].to_numpy()
    ra = good & (np.abs(r) >= calibration["raw_q90"]) & (r != 0)
    da = good & (np.abs(d) >= calibration["deviation_q90"]) & (d != 0)
    retained, after = [], -math.inf
    for i in np.flatnonzero(ra | da):
        if frame.SampleTime.iloc[i] >= after:
            retained.append(i)
            after = frame.SampleTime.iloc[i] + 5000000
    result = frame.iloc[retained].copy()
    result["R_score"], result["D_score"] = r[retained], d[retained]
    result["R_active"], result["D_active"] = ra[retained], da[retained]
    return result


def markout(tape, origin, horizon, delay):
    entry_time, exit_time = int(origin) + delay, int(origin) + delay + horizon
    entry, er = quotes.quote_at(tape, entry_time, 3000000)
    end, xr = quotes.quote_at(tape, exit_time, 3000000)
    if entry is None or end is None:
        return {"known": False, "reason": "entry_" + er if entry is None else "exit_" + xr}
    if min(entry["bid_volume"], entry["ask_volume"], end["bid_volume"], end["ask_volume"]) < 1:
        return {"known": False, "reason": "capacity"}
    start = np.searchsorted(tape[0], entry_time, side="left")
    stop = np.searchsorted(tape[0], exit_time, side="left")
    if (~np.isfinite(tape[1][start:stop]) | ~np.isfinite(tape[2][start:stop])).any():
        return {"known": False, "reason": "invalid_path_episode"}
    costs = {"future_commission_per_side_twd": 50.0, "future_tax_per_side": 0.00002, "stress_multiplier": 1.2}
    buy = quotes.cash_markout(entry, end, "buy", "future", 2000, costs)
    sell = quotes.cash_markout(entry, end, "sell", "future", 2000, costs)
    return {"known": True, "reason": "ok", "entry": entry, "exit": end, "buy": buy, "sell": sell}


def bootstrap_lower(daily, *, column="net_bps"):
    """Circular day blocks within month; all rows were already fixed before PnL."""
    if len(daily) < 6:
        return None, []
    rng = np.random.default_rng(1729)
    arrays = [part[column].to_numpy(float) for _, part in daily.groupby("month", sort=True)]
    if any(not np.isfinite(a).all() for a in arrays):
        raise ContractError("Unknown daily outcomes cannot enter bootstrap as zeros")
    draws = []
    for _ in range(4000):
        sampled = []
        for values in arrays:
            starts = rng.integers(0, len(values), size=math.ceil(len(values) / 3))
            indices = np.concatenate([(start + np.arange(3)) % len(values) for start in starts])[: len(values)]
            sampled.extend(values[indices])
        draws.append(float(np.mean(sampled)))
    return float(np.quantile(draws, 0.025)), draws


def context_columns(intents, target_flips, reference_flips):
    origins = intents.SampleTime.to_numpy(np.int64)
    target_indices = np.searchsorted(target_flips, origins, side="left") - 1
    reference_indices = np.searchsorted(reference_flips, origins, side="left") - 1
    last_target = np.where(target_indices >= 0, target_flips[np.maximum(target_indices, 0)] if len(target_flips) else 0, 0)
    last_reference = np.where(reference_indices >= 0, reference_flips[np.maximum(reference_indices, 0)] if len(reference_flips) else 0, 0)
    intents["quiet_target_after_reference_flip"] = (last_target > 0) & (origins - last_target >= 1000000) & (last_reference > last_target)
    intents["reference_age_cell"] = np.searchsorted([0.05, 0.25], intents[FEATURES["reference_age_s"]], side="right")
    intents["history_cell"] = (intents[FEATURES["basis_history_fraction"]] >= 0.5).astype(int)
    intents["spread_cell"] = ((intents.TargetAskTicks - intents.TargetBidTicks) > 1).astype(int)
    times = pd.to_datetime(origins, unit="us", utc=True).tz_convert("Asia/Taipei")
    intents["time_cell"] = (times.hour * 60 + times.minute >= 630).astype(int)
    return intents


def analyze(output):
    config = verify_freeze(output)
    if read_yaml(output / "status.yaml")["state"] not in {"export_complete", "analysis_complete"}:
        raise ContractError("Complete the fixed export panel before analysis")
    inventory = read_yaml(output / "inventory.yaml")
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    calibration = {}
    for product in ["CDF", "DHF", "DVF"]:
        frames = [
            pd.read_parquet(output / "native" / (r["day"] + "-" + product) / "grid/values.parquet", columns=list(FEATURES.values()))
            for r in inventory
            if r["available"] and r["product"] == product and r["day"].startswith("202606")
        ]
        data = pd.concat(frames, ignore_index=True)
        good = data.loc[eligible(data)]
        raw = good[FEATURES["basis_bps"]].to_numpy()
        if not len(raw):
            raise ContractError("No June calibration support: " + product)
        center = float(np.median(raw))
        calibration[product] = {
            "raw_center": center,
            "raw_q90": float(np.quantile(np.abs(raw - center), 0.9, method="linear")),
            "deviation_q90": float(np.quantile(np.abs(good[FEATURES["basis_deviation_bps"]]), 0.9, method="linear")),
            "origins": len(data),
            "eligible": len(good),
            "uses_outcomes": False,
        }
    write_yaml(analysis / "june-calibration.yaml", calibration)
    intentions, daily_records, ic_records, coverage_records, event_records, clock_records = [], [], [], [], [], []
    for count, row in enumerate(inventory, 1):
        if not row["available"]:
            coverage_records.append({"day": row["day"], "product": row["product"], "observed": False, "reason": "|".join(row["missing"])})
            continue
        work = output / "native" / (row["day"] + "-" + row["product"])
        receipt = read_yaml(work / "receipt.yaml")
        for name, expected in receipt["files"].items():
            if file_hash(work / name) != expected:
                raise ContractError("Native artifact changed: " + str(work / name))
        frame = pd.read_parquet(work / "grid/values.parquet")
        good = eligible(frame)
        coverage_records.append(
            {"day": row["day"], "product": row["product"], "observed": True, "origins": len(frame), "eligible": int(good.sum()), "fraction": float(good.mean()), "reason": "ok"}
        )
        for horizon in [1, 5, 30]:
            label = f"ssf_mid_return_bps[{horizon}s]"
            for feature in ["basis_bps", "basis_deviation_bps"]:
                known = good & np.isfinite(frame[label])
                value = (
                    frame.loc[known, FEATURES[feature]].corr(frame.loc[known, label], method="spearman")
                    if known.sum() >= 3 and frame.loc[known, FEATURES[feature]].nunique() > 1 and frame.loc[known, label].nunique() > 1
                    else math.nan
                )
                ic_records.append({"day": row["day"], "product": row["product"], "horizon": horizon, "feature": feature, "known": int(known.sum()), "ic": value})
        tf = read_flip_trace(work, "target-flips", row["day"]).SampleTime.to_numpy(np.int64)
        rf = read_flip_trace(work, "reference-flips", row["day"]).SampleTime.to_numpy(np.int64)
        frame = context_columns(frame, tf, rf)
        selected = make_intents(frame, calibration[row["product"]])
        cutoffs = calibration[row["product"]]
        raw_signal = frame[FEATURES["basis_bps"]] - cutoffs["raw_center"]
        deviation_signal = frame[FEATURES["basis_deviation_bps"]]
        unspaced_tail = good & (((raw_signal.abs() >= cutoffs["raw_q90"]) & (raw_signal != 0)) | ((deviation_signal.abs() >= cutoffs["deviation_q90"]) & (deviation_signal != 0)))
        selected["day"], selected["product"] = row["day"], row["product"]
        intentions.append(selected)
        next_target = np.searchsorted(tf, rf, side="right")
        tied_target = np.searchsorted(tf, rf, side="right") != np.searchsorted(tf, rf, side="left")
        delay = (tf[np.minimum(next_target, max(0, len(tf) - 1))] - rf) / 1000000 if len(tf) else np.full(len(rf), np.nan)
        resolved = next_target < len(tf)
        clock_records.append(
            {
                "day": row["day"],
                "product": row["product"],
                "target_flips": len(tf),
                "reference_flips": len(rf),
                "reference_next_target_censored": int((~resolved).sum()),
                "reference_target_same_receipt_ambiguous": int(tied_target.sum()),
                "reference_next_target_median_s": float(np.median(delay[resolved])) if resolved.any() else None,
                "reference_next_target_above_1s_fraction": float((delay[resolved] > 1).mean()) if resolved.any() else None,
                "eligible_union_intents": len(selected),
                "quiet_target_intents": int(selected.quiet_target_after_reference_flip.sum()),
                "eligible_grid_origins": int(good.sum()),
                "eligible_quiet_target_grid_origins": int((good & frame.quiet_target_after_reference_flip.to_numpy()).sum()),
                "unspaced_eligible_tail_grid_origins": int(unspaced_tail.sum()),
                "unspaced_quiet_target_tail_origins": int((unspaced_tail & frame.quiet_target_after_reference_flip).sum()),
            }
        )
        # Write causal selections before opening target quote outcomes for this day.
        selected.to_parquet(analysis / f"intents-{row['day']}-{row['product']}.parquet", index=False)
        tape, quote_receipt = quotes.read_quotes(Path(row["sources"]["target"]["path"]), row["sources"]["target"]["stamp"])
        if quote_receipt["sha256"] != row["sources"]["target"]["sha256"]:
            raise ContractError("Quote/native source mismatch")
        if len(rf):
            target_end = tf[np.minimum(next_target, len(tf) - 1)] if len(tf) else np.full(len(rf), rf[-1])
            prefix = np.r_[0, np.cumsum(~np.isfinite(tape[1]) | ~np.isfinite(tape[2]))]
            begin = np.searchsorted(tape[0], rf, side="left")
            end = np.searchsorted(tape[0], target_end, side="right")
            prior = np.maximum(0, begin - 1)
            interrupted = (prefix[end] - prefix[prior]) > 0
            supported = resolved & ~interrupted & ~tied_target
            clock_records[-1].update(
                reference_next_target_invalid_episode=int((resolved & interrupted).sum()),
                reference_next_target_median_s=float(np.median(delay[supported])) if supported.any() else None,
                reference_next_target_above_1s_fraction=float((delay[supported] > 1).mean()) if supported.any() else None,
            )
        for horizon in [1000000, 5000000, 30000000]:
            for delay_us in [50000, 250000]:
                outcomes = [markout(tape, int(t), horizon, delay_us) for t in selected.SampleTime]
                for view in ["R", "D"]:
                    active = selected[view + "_active"].to_numpy(bool)
                    values, rows = [], []
                    for i, (sample, outcome) in enumerate(zip(selected.to_dict("records"), outcomes, strict=True)):
                        event = {
                            k: sample[k]
                            for k in ["SampleTime", "day", "product", "reference_age_cell", "history_cell", "spread_cell", "time_cell", "quiet_target_after_reference_flip"]
                        }
                        event.update(
                            view=view,
                            horizon_us=horizon,
                            delay_us=delay_us,
                            active=bool(active[i]),
                            known=bool(outcome["known"]),
                            policy_known=bool(not active[i] or outcome["known"]),
                            reason=outcome["reason"],
                            score=sample[view + "_score"],
                        )
                        if not active[i]:
                            for name in ["net_bps", "stress_net_bps", "gross_bps", "gross_cash", "net_cash", "fee_bps", "directional_excess_bps", "directional_gross_excess_bps"]:
                                event[name] = 0.0
                        if outcome["known"]:
                            side = "buy" if sample[view + "_score"] > 0 else "sell"
                            own, buy, sell = outcome[side], outcome["buy"], outcome["sell"]
                            for name in ["net_bps", "stress_net_bps", "gross_bps", "gross_cash", "net_cash", "fee_bps"]:
                                event[name] = own[name] if active[i] else 0.0
                            event["long_net_bps"], event["short_net_bps"] = buy["net_bps"], sell["net_bps"]
                            event["random_net_bps"] = (buy["net_bps"] + sell["net_bps"]) / 2
                            event["directional_excess_bps"] = own["net_bps"] - event["random_net_bps"] if active[i] else 0.0
                            event["directional_gross_excess_bps"] = own["gross_bps"] - (buy["gross_bps"] + sell["gross_bps"]) / 2 if active[i] else 0.0
                            for which in ["entry", "exit"]:
                                for name, value in outcome[which].items():
                                    event[which + "_" + name] = value
                        rows.append(event)
                        if active[i] and outcome["known"]:
                            values.append(event)
                    event_records.extend(rows)
                    observed = pd.DataFrame(values)
                    record = {
                        "day": row["day"],
                        "month": row["day"][:6],
                        "product": row["product"],
                        "view": view,
                        "horizon_us": horizon,
                        "delay_us": delay_us,
                        "union_intents": len(selected),
                        "active": int(active.sum()),
                        "known": len(values),
                        "unknown": int(active.sum()) - len(values),
                    }
                    for metric in ["net_bps", "stress_net_bps", "directional_excess_bps", "directional_gross_excess_bps", "long_net_bps", "short_net_bps"]:
                        record[metric] = float(observed[metric].mean()) if len(values) else 0.0 if not active.any() else math.nan
                    record["net_cash"] = float(observed.net_cash.sum()) if len(values) else 0.0 if not active.any() else math.nan
                    record["common_union_net_bps"] = (
                        float(sum(v.get("net_bps", 0) for v in rows if v["known"]) / sum(v["known"] for v in rows))
                        if any(v["known"] for v in rows)
                        else 0.0
                        if not len(rows)
                        else math.nan
                    )
                    daily_records.append(record)
        if count % 10 == 0 or count == len(inventory):
            print(f"analyzed {count}/{len(inventory)} pair-days", flush=True)
    pd.concat(intentions, ignore_index=True).to_parquet(analysis / "all-intents-before-outcomes.parquet", index=False)
    daily, events, ics, coverage = pd.DataFrame(daily_records), pd.DataFrame(event_records), pd.DataFrame(ic_records), pd.DataFrame(coverage_records)
    for column in ["net_bps", "stress_net_bps", "gross_bps", "net_cash", "directional_excess_bps", "directional_gross_excess_bps", "long_net_bps", "short_net_bps"]:
        if column not in events:
            events[column] = np.nan
    if not len(events):
        for column in ["active", "known", "policy_known"]:
            events[column] = pd.Series(dtype=bool)
        for column in [
            "day",
            "product",
            "view",
            "horizon_us",
            "delay_us",
            "reference_age_cell",
            "history_cell",
            "spread_cell",
            "time_cell",
            "quiet_target_after_reference_flip",
            "SampleTime",
        ]:
            events[column] = pd.Series(dtype=object)
    events["month"] = events.day.str[:6]
    daily.to_csv(analysis / "daily.csv", index=False)
    events.to_parquet(analysis / "all-events.parquet", index=False)
    ics.to_csv(analysis / "daily-ic.csv", index=False)
    coverage.to_csv(analysis / "coverage.csv", index=False)
    pd.DataFrame(clock_records).to_csv(analysis / "sampler-coverage.csv", index=False)
    known_active = events.loc[events.active & events.known]
    context_keys = ["month", "product", "view", "horizon_us", "delay_us", "reference_age_cell", "history_cell", "spread_cell", "time_cell"]
    cells = known_active.groupby(context_keys, dropna=False)[["net_bps", "directional_excess_bps"]].agg(["count", "mean"])
    cells = cells.reindex(
        pd.MultiIndex.from_product(
            [["202606", "202607", "202608"], ["CDF", "DHF", "DVF"], ["R", "D"], [1000000, 5000000, 30000000], [50000, 250000], [0, 1, 2], [0, 1], [0, 1], [0, 1]],
            names=context_keys,
        )
    )
    for metric in ["net_bps", "directional_excess_bps"]:
        cells[(metric, "count")] = cells[(metric, "count")].fillna(0).astype(int)
    cells.to_csv(analysis / "context-cells.csv")
    known_active.groupby(["month", "product", "view", "horizon_us", "delay_us", "quiet_target_after_reference_flip"])[["net_bps", "directional_excess_bps"]].agg(
        ["count", "mean"]
    ).to_csv(analysis / "quiet-target-economics.csv")
    common_context = events.loc[events.known].groupby(context_keys)["net_bps"].mean().unstack("view").reindex(columns=["R", "D"])
    common_context["D_minus_R_common_union_bps"] = common_context.D - common_context.R
    common_context.to_csv(analysis / "context-common-union-comparison.csv")
    decisions, draws = [], {}
    for view in ["R", "D"]:
        part = daily.loc[(daily["product"] == "CDF") & (daily.view == view) & (daily.horizon_us == 5000000) & (daily.delay_us == 50000) & (daily.day >= "20260701")].copy()
        monthly = []
        for month in ["202607", "202608"]:
            m = part.loc[part.month == month]
            month_coverage = coverage.loc[(coverage["product"] == "CDF") & coverage.day.str.startswith(month)]
            monthly.append(
                {
                    "month": month,
                    "days": len(m),
                    "planned_pair_days": len(month_coverage),
                    "observed_pair_days": int(month_coverage.observed.sum()),
                    "unknown_pair_days": int((~month_coverage.observed).sum()),
                    "planned_day_coverage": float(month_coverage.observed.mean()),
                    "active": int(m.active.sum()),
                    "known": int(m.known.sum()),
                    "known_fraction": float(m.known.sum() / m.active.sum()) if m.active.sum() else 0.0,
                    "net_bps": float(m.net_bps.mean()),
                    "net_cash": float(m.net_cash.sum()),
                    "directional_excess_bps": float(m.directional_excess_bps.mean()),
                }
            )
        # A completely unresolved active day is not silently dropped from an interval.
        lower, values = bootstrap_lower(part) if np.isfinite(part.net_bps).all() else (None, [])
        draws[view] = values
        primary_events = events.loc[(events["product"] == "CDF") & (events.view == view) & (events.horizon_us == 5000000) & (events.day >= "20260701") & events.active]
        a = primary_events.loc[(primary_events.delay_us == 50000) & primary_events.known]
        b = primary_events.loc[(primary_events.delay_us == 250000) & primary_events.known]
        common = a.merge(b, on=["day", "SampleTime"], suffixes=("_50", "_250"))
        e50 = float(common.directional_gross_excess_bps_50.mean()) if len(common) else math.nan
        e250 = float(common.directional_gross_excess_bps_250.mean()) if len(common) else math.nan
        retention = e250 / e50 if e50 > 0 else None
        positive_days = part.net_cash.clip(lower=0)
        concentration = float(positive_days.max() / positive_days.sum()) if positive_days.sum() > 0 else None
        checks = {
            "support": all(m["known"] >= 100 and m["known_fraction"] >= 0.95 and m["planned_day_coverage"] >= 0.90 for m in monthly),
            "both_months_positive": all(m["net_bps"] > 0 and m["net_cash"] > 0 for m in monthly),
            "stress_positive": float(part.stress_net_bps.mean()) > 0,
            "adjusted_lower_positive": lower is not None and lower > 0,
            "directional_controls": all(m["directional_excess_bps"] > 0 for m in monthly) and part.net_bps.mean() > max(part.long_net_bps.mean(), part.short_net_bps.mean()),
            "delay_retention": retention is not None and retention >= 1 / 3 and len(common) > 0 and common.net_bps_250.mean() > 0,
            "day_concentration": concentration is not None and concentration <= 0.3 and part.net_cash.sum() - part.net_cash.max() > 0,
        }
        checks = {name: bool(value) for name, value in checks.items()}
        decisions.append(
            {
                "view": view,
                "status": "advance_to_native_execution_confirmation"
                if all(checks.values())
                else "inconclusive_support"
                if not checks["support"]
                else "rejected_frozen_economic_formulation",
                "checks": checks,
                "monthly": monthly,
                "net_lower_97_5": lower,
                "delay_retention": retention,
                "max_positive_day_share": concentration,
            }
        )
    write_yaml(analysis / "bootstrap-draws.yaml", draws)
    validation = daily.loc[(daily.horizon_us == 5000000) & (daily.delay_us == 50000) & (daily.day >= "20260701")]
    paired = validation.pivot(index=["day", "month", "product"], columns="view", values=["net_cash", "common_union_net_bps"])
    paired["D_minus_R_cash"] = paired[("net_cash", "D")] - paired[("net_cash", "R")]
    paired["D_minus_R_common_union_bps"] = paired[("common_union_net_bps", "D")] - paired[("common_union_net_bps", "R")]
    paired.to_csv(analysis / "history-attribution-daily.csv")
    history_monthly = paired[["D_minus_R_cash", "D_minus_R_common_union_bps"]].groupby(["month", "product"]).mean()
    history_monthly.to_csv(analysis / "history-attribution-monthly.csv")
    history_positive = True
    for month in ["202607", "202608"]:
        portion = paired.xs((month, "CDF"), level=("month", "product"))["D_minus_R_cash"].to_numpy()
        history_positive &= bool(len(portion) and np.isfinite(portion).all() and portion.mean() > 0)
    history_claim = decisions[1]["status"] == "advance_to_native_execution_confirmation" and history_positive
    write_yaml(
        analysis / "summary.yaml",
        {
            "study": config["study"],
            "evidence_role": config["evidence_role"],
            "decisions": decisions,
            "historical_representation_claim_supported": history_claim,
            "planned_pair_days": len(inventory),
            "observed_pair_days": int(coverage.observed.sum()),
            "limitations": [
                "Quote markouts are not native execution or final OOS.",
                "Missing whole pair-days are unknown and excluded from numeric series; coverage.csv retains every planned row.",
                "Native endpoint labels and economic path/freshness rules intentionally differ.",
            ],
        },
    )
    write_yaml(output / "status.yaml", {"state": "analysis_complete", "decisions": decisions})
    print(json.dumps(decisions, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "pilot", "export", "analyze"])
    parser.add_argument("--registration", type=Path, default=ROOT / "Experiments/e2e_20260926/basis_mechanism.yaml")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.registration.resolve(), args.output.resolve())
    elif args.action == "analyze":
        analyze(args.output.resolve())
    else:
        export(args.output.resolve(), args.action == "pilot")


if __name__ == "__main__":
    main()
