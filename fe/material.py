"""Native material: one coco replay per (day, symbol) and feature set.

Every set shares the DatasetWriter PeriodicSampler of the study session, so rows
of different sets align by SampleTime. Only the ``anchor`` set carries labels;
it defines the row cohort. Other universe symbols' CurrentBook are subscribed in
every replay (the shared market clock and peers). Sets that reference futures
add the TAIFEX contracts and tapes; a day without the reference contract runs
the declared fallback set instead. Missing inputs skip the job; nothing is
filled. Outputs are compacted to float32 (NaN/inf preserved) zstd parquet.
"""

from __future__ import annotations

import copy
import hashlib
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
from ..signal.material import MARKET_SUFFIXES
from .profile import ROOT, work


def snapshot_binary(profile) -> str:
    """Content-addressed copy so rebuilding the engine cannot change a running generation."""
    source = Path(profile["paths"]["coco_binary"])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = work(profile) / "bin" / f"coco-{digest}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        shutil.copy2(source, tmp)
        tmp.rename(target)
    return str(target)


def guide(profile, binary) -> Path:
    out = work(profile) / "guides" / f"{Path(binary).name}.yaml"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([binary, "feature-guide", "--output", str(out)], check=True, capture_output=True)
    return out


def modules(profile, binary, feature_set: dict) -> list[dict]:
    """Module declarations of a set: the native template pool and/or explicit modules."""
    out = []
    if feature_set.get("pool"):
        entries = {m["type"]: m for m in read_yaml(guide(profile, binary))["module_types"]}
        excluded = set(feature_set.get("exclude_subcategories", []))
        allowed = {k: v for k, v in entries.items() if v.get("subcategory") not in excluded}
        variants, _, _ = compile_expansion(ROOT / "workflows/profiles/config_expansion", allowed)
        out += variants
    out += copy.deepcopy(feature_set.get("modules", []))
    return out


def _render(value, symbol, futures=""):
    if isinstance(value, str):
        return value.replace("{Target}", symbol).replace("{Futures}", futures)
    if isinstance(value, list):
        return [_render(v, symbol, futures) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, symbol, futures) for k, v in value.items()}
    return value


def futures_map(profile) -> dict:
    path = work(profile) / "universe" / "futures_map.yaml"
    return read_yaml(path)["roots"] if path.is_file() else {}


def config(profile, binary, feature_set: dict, symbol: str, universe: list[str], day: str | None = None) -> dict:
    paths, session = profile["paths"], profile["session"]
    subscribe = [{"Book": [symbol], "Trade": [symbol]}]
    decls = [
        {"Desc": "TwseFilter.0", "Spec": {"Subscribe": subscribe, "RequireTradable": False}},
        {"Desc": "CurrentBook.0", "Spec": {"Subscribe": subscribe}},
    ]
    mods = modules(profile, binary, feature_set)
    # A module naming references (e.g. TXF@1) subscribes them itself; the
    # replay only needs the TAIFEX contracts and tapes.
    futures = any(m.get("references") or m.get("stock_futures") or m.get("breadth_pairs") for m in mods)
    exports = []
    root = futures_map(profile).get(symbol)
    for module in ordered_modules(copy.deepcopy(mods)):
        spec = _render(copy.deepcopy(module["spec"]), symbol, f"{root}@1" if root else "")
        if module.get("breadth_pairs"):
            # Day-resolved pairs with an existing futures tape: a missing tape would fail the replay.
            from .instruments import day_pairs

            pairs = day_pairs(profile, day)
            spec["Pairs"] = [f"{s}:{f}" for s, f in pairs] or [f"{symbol}:NONE"]
            spec["Subscribe"] = [{"Book": sorted({symbol, *[s for s, _ in pairs], *[f for _, f in pairs]})}]
        if not spec.get("Dep") and "Subscribe" not in spec:
            spec["Subscribe"] = subscribe
        if module.get("peers"):
            names = [symbol, *[u for u in universe if u != symbol]]
            spec["Subscribe"] = [{"Book": names}]
        decls.append({"Desc": f"{module['type']}.{module['id']}", "Spec": spec})
    for module in mods:
        exports.extend(f"{module['type']}.{module['id']}.{family}.*@{symbol}" for family in module["exports"])
    labelers = []
    if feature_set.get("labels"):
        labelers.append(
            {
                "Type": "ForwardReturnLabeler",
                "Spec": {
                    "Dep": {"Book": [f"CurrentBook.0@{symbol}"]},
                    "Labels": [{"Y": f"CurrentBook.0.book_mid_ticks.0@{symbol}", "Horizon": f"{h}s", "Name": f"mid_ret_{h}"} for h in profile["labels"]["horizons"]],
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
        infra.append({"Desc": "DailyMdStatsInfo.0", "Spec": {"Dir": str(work(profile) / "daily_md_stats"), "HistoryDays": int(feature_set["daily_stats"])}})
    groups = [{"Gid": "", "Decl": infra}]
    groups += [{"Gid": other, "Decl": [{"Desc": "CurrentBook.0", "Spec": {"Subscribe": [{"Book": [other], "Trade": [other]}]}}]} for other in universe if other != symbol]
    groups.append({"Gid": symbol, "Decl": decls})
    return {"Users": [{"UserId": 1, "Credit": 1e7}], "Modules": groups}


def compact(source: Path, target: Path):
    table = pq.read_table(source)
    fields = [pa.field(f.name, pa.float32()) if pa.types.is_float64(f.type) and not f.name.startswith("mid_ret_") else f for f in table.schema]
    table = table.cast(pa.schema(fields))
    tmp = target.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    tmp.rename(target)


def has_market(profile, symbol, day):
    base = Path(profile["paths"]["market_data"]) / symbol
    return any((base / f"{day}{s}").is_file() and (base / f"{day}{s}").stat().st_size for s in MARKET_SUFFIXES)


def run_job(profile, binary, cfg: dict, out_root: Path, day: str, symbol: str, fallback: dict | None = None) -> dict:
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
    argv = [binary, "-d", day, "-C", str(job), "--run-status-dir", str(job / "status"), "--log-dir", str(job / "logs")]
    argv += ["--trading-calendar", profile["paths"]["calendar"], str(job / "config.yaml")]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    produced = job / "data" / day / symbol / "values.parquet"
    status = "ok" if proc.returncode == 0 and produced.is_file() else f"fail_{proc.returncode}"
    if status == "ok":
        out.parent.mkdir(parents=True, exist_ok=True)
        compact(produced, out)
        manifest = out_root / "feature_semantic_manifest.yaml"
        if not manifest.exists():
            shutil.copy2(produced.parent / "feature_semantic_manifest.yaml", manifest)
        shutil.rmtree(job)
    else:
        (job / "stderr.txt").write_text(proc.stderr[-20000:])
        if fallback is not None:
            result = run_job(profile, binary, fallback, out_root, day, symbol)
            return {**result, "status": f"fallback_{result['status']}"}
    return {"day": day, "symbol": symbol, "status": status, "seconds": round(time.time() - t0, 1)}


def generate(profile, set_name: str, jobs: list[tuple[str, str]], universe: list[str], workers=22) -> dict:
    """Replay ``jobs`` = [(day, symbol)] for one feature set of the profile."""
    binary = snapshot_binary(profile)
    feature_set = profile["feature_sets"][set_name]
    out_root = work(profile) / "material" / set_name
    out_root.mkdir(parents=True, exist_ok=True)
    symbols = sorted({s for _, s in jobs})
    fallback_set = profile["feature_sets"].get(feature_set.get("fallback", ""))
    roots = futures_map(profile)
    # A set that needs the target's stock futures runs its fallback for symbols without one.
    primary = {s: fallback_set if feature_set.get("stock_futures") and s not in roots else feature_set for s in symbols}
    configs = {s: config(profile, binary, primary[s], s, universe) for s in symbols}
    fallbacks = {s: config(profile, binary, fallback_set, s, universe) for s in symbols} if fallback_set else {}
    record = {"set": feature_set, "binary": binary, "binary_sha256": Path(binary).name.split("-")[-1]}
    (out_root / "set.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    results = []
    per_day = any(m.get("breadth_pairs") for m in feature_set.get("modules", []))

    def cfg(d, s):
        return config(profile, binary, primary[s], s, universe, d) if per_day else configs[s]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, profile, binary, cfg(d, s), out_root, d, s, fallbacks.get(s)) for d, s in jobs]
        for i, f in enumerate(futures):
            r = f.result()
            results.append(r)
            if r["status"] not in ("ok", "reused", "fallback_ok") or i % 250 == 0:
                print(json.dumps(r), f"{i + 1}/{len(futures)}", flush=True)
    with (out_root / "jobs.jsonl").open("a") as stream:
        for r in results:
            stream.write(json.dumps(r) + "\n")
    counts: dict = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return counts
