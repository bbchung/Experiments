"""Reusable native FutureTruth exports with content-addressed provenance."""

from __future__ import annotations

import datetime as dt
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ...io import ContractError, digest, file_hash, lock, read_yaml, write_yaml
from ...kernels.data import base_config, declaration, market_path, surface
from ...market import futures, instrument, metadata_paths, resolve_contract, scope
from ...native_contract import native_descriptors, validate_captured_origins, validate_receipt
from ...profile import load_profile
from ...store import ExposureLedger
from .config import validate_data
from .targets import truth_columns

CONTRACT = "native-future-truth-ticks-v3-all-origins"
SEMANTICS = {
    "price": "book_sticky_ticks",
    "clock": "logical_receive_us",
    "endpoint": "inclusive",
    "ties": "native_event_order",
    "extrema_ties": "first",
    "no_hit": -1,
    "invalid_path": "NaN",
    "anchor_in_extrema": True,
    "incomplete_max_horizon": "retain_origin_with_unknown_truth",
}


def render(profile, data, truth):
    config = base_config(profile)
    if "otc_info_root" in data:
        if futures(profile):
            raise ContractError("OTC metadata cannot be used with a futures profile")
        root = Path(data["otc_info_root"])
        if not root.is_absolute():
            raise ContractError("TD otc_info_root must be absolute")
        config["Modules"][0]["Decl"].insert(
            1, declaration("OtcInfo", 0, {"intraday": str(root / "intraday"), "aftertrading": str(root / "aftertrading"), "disposal": str(root / "disposition"), "HistoryDays": 0})
        )
    for symbol in data["symbols"]:
        declarations, _ = surface(profile, data["modules"], symbol)
        labelers = []
        labelers.append(
            {
                "Type": "FutureTruth",
                "Spec": {
                    "Dep": {"Book": [f"CurrentBook.0@{symbol}"]},
                    "Y": f"CurrentBook.0.book_sticky_ticks.0@{symbol}",
                    "HorizonsSeconds": truth["horizons_seconds"],
                    "UpTicks": truth["up_ticks"],
                    "DownTicks": truth["down_ticks"],
                },
            }
        )
        declarations.append(
            declaration(
                "DatasetWriter",
                0,
                {
                    "Subscribe": [{"Book": [instrument(profile, symbol)]}],
                    "Exports": [f"{f}@{symbol}" for f in data["features"]],
                    "Labelers": labelers,
                    "OutputPath": "${cwd}/data/" + symbol + "/values.parquet",
                    "Format": "parquet",
                    "UseTmp": True,
                    "RequireAlphaFactorExports": True,
                    "EmitSampleContext": True,
                    "BookFlipSampler": {
                        "Mode": "all_book",
                        "MinInterval": f"{data['sample_interval_seconds']}s",
                        "StartTime": profile["data"]["start_time"],
                        "UntilTime": profile["data"]["until_time"],
                    },
                },
            )
        )
        config["Modules"].append({"Gid": symbol, "Decl": declarations})
    return config


def native_jobs(config, data):
    """Bound feature slots while keeping every feed on the same market clock."""
    size = data.get("native_symbols_per_job", len(data["symbols"]))
    for start in range(0, len(data["symbols"]), size):
        symbols = data["symbols"][start : start + size]
        groups = []
        for group in config["Modules"]:
            if not group["Gid"] or group["Gid"] in symbols:
                groups.append(group)
            else:
                feeds = [d for d in group["Decl"] if d["Desc"] == "CurrentBook.0"]
                if len(feeds) != 1:
                    raise ContractError("TD sharding requires each symbol's native market subscription")
                groups.append({**group, "Decl": feeds})
        yield {**config, "Modules": groups}, symbols


def day_panel(data, day):
    """Freeze exclusions before replay; every target and shard uses this panel."""
    excluded = {r["symbol"] for r in data.get("excluded_partitions", []) if r["day"] == day}
    panel = {**data, "symbols": [s for s in data["symbols"] if s not in excluded]}
    if "otc_symbols" in panel:
        panel["otc_symbols"] = [s for s in panel["otc_symbols"] if s not in excluded]
        if not panel["otc_symbols"]:
            del panel["otc_symbols"], panel["otc_info_root"]
    return panel


def source_snapshot(profile, data, day):
    paths = [market_path(profile["paths"]["market_data"], resolve_contract(profile, s, day), day) for s in data["symbols"]]
    paths += metadata_paths(profile, day)
    optional = []
    if "otc_info_root" in data:
        root = Path(data["otc_info_root"])
        paths += [root / name / f"{day}.csv" for name in ["intraday", "disposition"]]
        # OtcInfo tries the previous 14 weekdays for a usable aftertrading file.
        # Record absent paths too: a newly appearing fallback changes identity.
        date = dt.date.fromisoformat(day)
        for _ in range(14):
            date -= dt.timedelta(days=1)
            while date.weekday() >= 5:
                date -= dt.timedelta(days=1)
            optional.append(root / "aftertrading" / f"{date:%Y%m%d}.csv")
    sources = {str(p.resolve()): file_hash(p) for p in paths}
    sources.update({str(p.resolve()): file_hash(p) if p.is_file() else None for p in optional})
    return sources


def verify_sources(sources):
    for name, expected in sources.items():
        path = Path(name)
        if (file_hash(path) if path.is_file() else None) != expected:
            raise ContractError("TD native inputs changed during replay")


def verify_files(root, files):
    if any(not (root / name).is_file() or file_hash(root / name) != checksum for name, checksum in files.items()):
        raise ContractError(f"TD native cache content changed: {root}")


def read_partition(path, data, truth):
    features = list(native_descriptors((path.parent / "feature_semantic_manifest.yaml").read_bytes()))
    schema = pq.read_schema(path)
    if set(features) != set(data["features"]) or b"coco.dataset_writer_schema_version" not in (schema.metadata or {}):
        raise ContractError("TD feature manifest differs from native AlphaFactor exports")
    if set(schema.names) != {*features, *truth_columns(truth), "SampleTime", "SampleBookTime", "SampleBookSeq"}:
        raise ContractError("TD native schema differs from frozen feature/truth contract")
    validate_captured_origins(path)
    summary = read_yaml(path.parent / "sample_summary.yaml")
    frame = pd.read_parquet(path)
    # CatBoost already consumes numeric features as float32. Compact at
    # ingestion to avoid holding the full panel twice at float64 width.
    numeric = {name: np.float32 for name in features if pd.api.types.is_numeric_dtype(frame[name].dtype)}
    frame = frame.astype(numeric).copy()
    if summary["emitted_rows"] != len(frame) or summary["sampled_rows"] != len(frame) or summary["pending_rows"] != 0:
        raise ContractError("TD native sample accounting mismatch")
    return frame, summary


def materialize(cfg, base, cache, *, exposure_owner=None):
    data = cfg["data"]
    validate_data(data)
    profile = load_profile((base / data["profile"]).resolve())
    # Reading cached truth consumes the same development information as a new
    # replay. Claim before either path, and retain the claim if the run fails.
    exposure = {
        "scope": scope(profile),
        "days": list(data["dates"]),
        "owner": exposure_owner or "target-discovery:" + digest({"protocol": cfg, "base": str(base.resolve())}),
        "role": "development",
    }
    ExposureLedger(Path(profile["paths"]["store"])).claim(exposure["scope"], exposure["days"], exposure["role"], exposure["owner"])
    config = render(profile, data, cfg["truth"])
    binary = Path(profile["paths"]["coco_binary"])
    engine = file_hash(binary)
    receipts, frames = [], []
    cache.mkdir(parents=True, exist_ok=True)
    for day in data["dates"]:
        panel = day_panel(data, day)
        daily_config = render(profile, panel, cfg["truth"])
        sources = source_snapshot(profile, panel, day)
        for job, symbols in native_jobs(daily_config, panel):
            request = {
                "contract": CONTRACT,
                "semantics": SEMANTICS,
                "truth": cfg["truth"],
                "day": day,
                "engine": engine,
                "config": job,
                "sources": sources,
                "producer": file_hash(Path(__file__)),
            }
            key = digest(request)
            root = cache / key
            with lock(cache / (key + ".lock")):
                if not root.exists():
                    work = Path(tempfile.mkdtemp(prefix=key[:12] + "-", dir=cache)).resolve()
                    write_yaml(work / "config.yaml", job)
                    argv = [str(binary), "-d", day, "-C", str(work), "--run-status-dir", str(work / "status"), "--log-dir", str(work / "logs")]
                    if futures(profile):
                        argv += ["--trading-calendar", profile["paths"]["calendar"]]
                    with (work / "stdout.log").open("w") as log:
                        subprocess.run([*argv, str(work / "config.yaml")], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=profile["resources"]["task_timeout_seconds"])
                    reports = list((work / "status").glob("*.yaml"))
                    if len(reports) != 1:
                        raise ContractError(f"TD needs exactly one native receipt: {work}")
                    validate_receipt(read_yaml(reports[0]), day)
                    verify_sources(sources)
                    if file_hash(binary) != engine:
                        raise ContractError("TD native inputs changed during replay")
                    files = {str(p.relative_to(work)): file_hash(p) for p in work.rglob("*") if p.is_file()}
                    write_yaml(work / "manifest.yaml", {"request": request, "files": files})
                    work.rename(root)
                manifest = read_yaml(root / "manifest.yaml")
                if manifest["request"] != request:
                    raise ContractError("TD native cache request mismatch")
                verify_files(root, manifest["files"])
            summaries = {}
            for symbol in symbols:
                path = root / "data" / symbol / "values.parquet"
                if not path.exists():
                    raise ContractError(f"TD requires every configured day/symbol partition: {path}")
                frame, summary = read_partition(path, data, cfg["truth"])
                summaries[symbol] = summary
                frame["day"], frame["symbol"] = day, symbol
                frames.append(frame)
            receipts.append({"day": day, "symbols": symbols, "cache_key": key, "manifest_hash": file_hash(root / "manifest.yaml"), "root": str(root), "samples": summaries})
        print(f"TD native truth: {day}", flush=True)
    return pd.concat(frames, ignore_index=True), {
        "contract": CONTRACT,
        "semantics": SEMANTICS,
        "profile": profile,
        "config": config,
        "receipts": receipts,
        "excluded_partitions": data.get("excluded_partitions", []),
        "exposure": exposure,
    }


def population(frame, cfg):
    """Causal captured origins; future availability is a diagnostic mask only."""
    validate_data(cfg["data"])
    frame = frame.sort_values(["day", "symbol", "SampleTime", "SampleBookSeq"]).reset_index(drop=True)
    keys = ["day", "symbol", "SampleTime", "SampleBookSeq"]
    if frame.duplicated(keys).any() or set(cfg["data"]["features"]) & set(truth_columns(cfg["truth"])):
        raise ContractError("Duplicate samples or future columns in feature manifest")
    valid = np.isfinite(frame[truth_columns(cfg["truth"])].to_numpy(dtype=float)).all(axis=1)
    audit = frame[keys].copy()
    audit["eligible"] = True
    audit["reason"] = "captured_origin"
    audit["complete_common_truth"] = valid
    excluded = {(r["day"], r["symbol"]): r["reason"] for r in cfg["data"].get("excluded_partitions", [])}
    reasons = pd.Series([excluded.get(key) for key in zip(frame.day, frame.symbol, strict=True)], index=frame.index, dtype=object)
    blocked = reasons.notna()
    audit.loc[blocked, "eligible"] = False
    audit.loc[blocked, "reason"] = "excluded_partition: " + reasons[blocked]
    audit["sample_id"] = [digest(list(row))[:24] for row in frame[keys].itertuples(index=False, name=None)]
    frame["sample_id"] = audit.sample_id
    eligible = frame.loc[audit.eligible].reset_index(drop=True)
    if eligible.empty:
        raise ContractError("No eligible captured origins")
    return eligible, audit
