"""Native material: one coco replay per (day, symbol), DatasetWriter parquet out.

Each replay holds the target's feature modules and DatasetWriter. Unless a set is
`solo`, every universe symbol's CurrentBook is subscribed too, so time windows
and cross-symbol modules share one market clock. Contracts come from the dated
contract CSVs; no eligibility or disposition metadata is read. Missing inputs
skip the job, they are never filled.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ...expansion import compile_expansion, ordered_modules
from ...io import read_yaml
from .profile import ROOT, work

MARKET_SUFFIXES = (".bin.zst", ".bin", ".csv.zst", ".csv")


def guide(profile) -> Path:
    """Feature guide of the configured binary (cached per binary mtime)."""
    binary = Path(profile["paths"]["coco_binary"])
    out = work(profile) / f"feature-guide-{int(binary.stat().st_mtime)}.yaml"
    if not out.exists():
        subprocess.run([str(binary), "feature-guide", "--output", str(out)], check=True, capture_output=True)
    return out


def pool(profile, feature_set: dict) -> list[dict]:
    """Module variants of one feature set: the native template pool and/or explicit modules."""
    modules = []
    if feature_set.get("pool"):
        entries = {m["type"]: m for m in read_yaml(guide(profile))["module_types"]}
        variants, _, _ = compile_expansion(ROOT / "workflows/profiles/config_expansion", entries)
        only = feature_set.get("only")
        modules += [v for v in variants if only is None or v["type"] in only]
    modules += copy.deepcopy(feature_set.get("modules", []))
    return modules


def _render(value, symbol):
    if isinstance(value, str):
        return value.replace("{Target}", symbol)
    if isinstance(value, list):
        return [_render(v, symbol) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, symbol) for k, v in value.items()}
    return value


def config(profile, feature_set: dict, symbol: str, universe: list[str]) -> dict:
    paths, session = profile["paths"], profile["session"]
    solo = bool(feature_set.get("solo"))
    subscribe = [{"Book": [symbol], "Trade": [symbol]}]
    decls = [
        {"Desc": "TwseFilter.0", "Spec": {"Subscribe": subscribe, "RequireTradable": False}},
        {"Desc": "CurrentBook.0", "Spec": {"Subscribe": subscribe}},
    ]
    modules = pool(profile, feature_set)
    futures = any(m.get("references") for m in modules)
    exports = []
    for module in ordered_modules(copy.deepcopy(modules)):
        spec = _render(copy.deepcopy(module["spec"]), symbol)
        if not spec.get("Dep"):
            spec["Subscribe"] = subscribe
        if module.get("peers"):
            names = [symbol, *[u for u in universe if u != symbol], *module.get("references", [])]
            spec["Subscribe"] = [{"Book": names, **({"Trade": names} if module.get("peer_trades") else {})}]
        decls.append({"Desc": f"{module['type']}.{module['id']}", "Spec": spec})
    for module in modules:
        exports.extend(f"{module['type']}.{module['id']}.{family}.*@{symbol}" for family in module["exports"])
    labelers = []
    if feature_set.get("labels"):
        labelers.append(
            {
                "Type": "ForwardReturnLabeler",
                "Spec": {
                    "Dep": {"Book": [f"CurrentBook.0@{symbol}"]},
                    "Labels": [{"Y": f"CurrentBook.0.book_mid_ticks.0@{symbol}", "Horizon": f"{h}s", "Name": f"mid_ret_{h}"} for h in profile["target"]["horizons"]],
                },
            }
        )
    writer = {
        "Exports": exports,
        "MetadataExports": [{"Feature": f"CurrentBook.0.book_mid_ticks.0@{symbol}", "Name": "origin_mid_ticks"}],
        "Labelers": labelers,
        "OutputPath": "${cwd}/data/${trading_date}/" + symbol + "/values.parquet",
        "UseTmp": True,
        "Format": "parquet",
        "RequireAlphaFactorExports": True,
        "EmitSampleContext": True,
        "PeriodicSampler": {"StartTime": session["start"], "UntilTime": session["until"], "SampleInterval": session["interval"]},
        "Subscribe": [{"Book": [symbol]}],
    }
    decls.append({"Desc": "DatasetWriter.0", "Spec": writer})
    infra = [
        {"Desc": "ContractImporter.0", "Spec": {"BasicInfo": [paths["contracts"], *([paths["futures_contracts"]] if futures else [])]}},
        {"Desc": "TradeBookMd.0", "Spec": {"Dirs": [paths["market_data"], *([paths["futures_data"]] if futures else [])]}},
    ]
    if feature_set.get("daily_stats"):
        infra.append({"Desc": "DailyMdStatsInfo.0", "Spec": {"Dir": str(work(profile) / "daily_md_stats"), "HistoryDays": 21}})
    groups = [{"Gid": "", "Decl": infra}]
    if not solo:
        groups += [{"Gid": other, "Decl": [{"Desc": "CurrentBook.0", "Spec": {"Subscribe": [{"Book": [other], "Trade": [other]}]}}]} for other in universe if other != symbol]
    groups.append({"Gid": symbol, "Decl": decls})
    return {"Users": [{"UserId": 1, "Credit": 1e7}], "Modules": groups}


def compact(source: Path, target: Path):
    """float64 -> float32 (NaN/inf preserved), zstd."""
    table = pq.read_table(source)
    fields = [pa.field(f.name, pa.float32()) if pa.types.is_float64(f.type) else f for f in table.schema]
    table = table.cast(pa.schema(fields))
    tmp = target.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    tmp.rename(target)


def has_market(profile, symbol, day):
    base = Path(profile["paths"]["market_data"]) / symbol
    return any((base / f"{day}{s}").is_file() and (base / f"{day}{s}").stat().st_size for s in MARKET_SUFFIXES)


def run_job(profile, cfg: dict, out_root: Path, day: str, symbol: str, keep_logs=False, fallback: dict | None = None) -> dict:
    out = out_root / "data" / day / symbol / "values.parquet"
    if out.is_file():
        return {"day": day, "symbol": symbol, "status": "reused"}
    if not (Path(profile["paths"]["contracts"]) / f"{day}.csv").is_file():
        return {"day": day, "symbol": symbol, "status": "skip_no_contract"}
    if not has_market(profile, symbol, day):
        return {"day": day, "symbol": symbol, "status": "skip_no_market"}
    job = out_root / "jobs" / day / symbol
    shutil.rmtree(job, ignore_errors=True)
    job.mkdir(parents=True)
    (job / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    t0 = time.time()
    argv = [profile["paths"]["coco_binary"], "-d", day, "-C", str(job), "--run-status-dir", str(job / "status"), "--log-dir", str(job / "logs")]
    argv += ["--trading-calendar", profile["paths"]["calendar"], str(job / "config.yaml")]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    produced = job / "data" / day / symbol / "values.parquet"
    status = "ok" if proc.returncode == 0 and produced.is_file() else f"fail_{proc.returncode}"
    if status == "ok":
        out.parent.mkdir(parents=True, exist_ok=True)
        compact(produced, out)
        shutil.move(str(produced.parent / "sample_summary.yaml"), out.parent / "sample_summary.yaml")
        manifest = out_root / "feature_semantic_manifest.yaml"
        if not manifest.exists():
            shutil.copy2(produced.parent / "feature_semantic_manifest.yaml", manifest)
        if not keep_logs:
            shutil.rmtree(job)
    else:
        (job / "stderr.txt").write_text(proc.stderr[-20000:])
        if fallback is not None:
            # A declared fallback (e.g. no reference instrument that day) replaces the failed replay.
            result = run_job(profile, fallback, out_root, day, symbol, keep_logs)
            return {**result, "status": f"fallback_{result['status']}"}
    return {"day": day, "symbol": symbol, "status": status, "seconds": round(time.time() - t0, 1)}


def snapshot_binary(profile) -> str:
    """Run a content-addressed copy so a rebuild during a long generation cannot change or break it."""
    import hashlib

    source = Path(profile["paths"]["coco_binary"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = work(profile) / "bin" / f"coco-{digest}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        shutil.copy2(source, tmp)
        tmp.rename(target)
    return str(target)


def generate(profile, set_name: str, days: list[str], universe: list[str], symbols=None, workers=24, keep_logs=False) -> dict:
    profile = {**profile, "paths": {**profile["paths"], "coco_binary": snapshot_binary(profile)}}
    feature_set = profile["features"]["sets"][set_name]
    out_root = work(profile) / "material" / set_name
    out_root.mkdir(parents=True, exist_ok=True)
    symbols = symbols or universe
    configs = {s: config(profile, feature_set, s, universe) for s in symbols}
    fallback_set = profile["features"]["sets"].get(feature_set.get("fallback", ""), None)
    fallbacks = {s: config(profile, fallback_set, s, universe) for s in symbols} if fallback_set else {}
    (out_root / "set.yaml").write_text(yaml.safe_dump({"set": feature_set, "binary": profile["paths"]["coco_binary"]}, sort_keys=False))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool_:
        futures = [pool_.submit(run_job, profile, configs[s], out_root, d, s, keep_logs, fallbacks.get(s)) for d in days for s in symbols]
        for i, f in enumerate(futures):
            r = f.result()
            results.append(r)
            if r["status"] not in ("ok", "reused", "fallback_ok") or i % 200 == 0:
                print(json.dumps(r), f"{i + 1}/{len(futures)}", flush=True)
    with (out_root / "jobs.jsonl").open("a") as stream:
        for r in results:
            stream.write(json.dumps(r) + "\n")
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts
