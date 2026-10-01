"""Replay parity, native endpoint classes and future-prefix invariance."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

from AstraResearch.engine_identity import identity
from AstraResearch.expansion import compile_expansion
from AstraResearch.io import read_yaml, write_yaml
from AstraResearch.kernels.data import ConfigKernel
from AstraResearch.material import same_values
from AstraResearch.native_contract import validate_receipt
from AstraResearch.profile import load_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    parser.add_argument("--prefix-cutter", type=Path, required=True)
    parser.add_argument("--attempt", default="validation")
    args = parser.parse_args()
    root = args.root.resolve()
    work = root / args.attempt
    work.mkdir(parents=True, exist_ok=False)
    profile = load_profile(Path(__file__).with_name("material.yaml"))
    engine = identity(Path(profile["paths"]["coco_binary"]))["fingerprint"]
    guide_text = subprocess.check_output([profile["paths"]["coco_binary"], "feature-guide"], text=True)
    (work / "feature-guide.yaml").write_text(guide_text)
    guide = {module["type"]: module for module in read_yaml(work / "feature-guide.yaml")["module_types"]}
    variants, _, _ = compile_expansion(Path.cwd(), guide)
    flow = next(module for module in variants if module["type"] == "FlowResponseSurprise")
    day = "20260302"
    timings = {}

    def run(document, name):
        folder = work / name
        folder.mkdir()
        config = folder / "config.yaml"
        write_yaml(config, document)
        argv = [
            profile["paths"]["coco_binary"],
            "-d",
            day,
            "-C",
            str(folder),
            "--trading-calendar",
            profile["paths"]["calendar"],
            "--run-status-dir",
            str(folder / "status"),
            str(config),
        ]
        before = time.monotonic()
        with (folder / "native.log").open("wb") as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
        timings[name] = time.monotonic() - before
        validate_receipt(read_yaml(folder / "status" / f"{day}.yaml"), day)
        return pq.read_table(folder / "data" / day / "2330" / "values.parquet")

    document = read_yaml(root / "benchmark" / "config-2.yaml")
    target = next(group for group in document["Modules"] if group["Gid"] == "2330")
    native = copy.deepcopy(flow["spec"])
    native["Dep"] = {kind: ["TwseFilter.0@2330"] for kind in ("Book", "Trade")}
    target["Decl"].insert(-1, {"Desc": "FlowResponseSurprise.0", "Spec": native})
    target["Decl"][-1]["Spec"]["Exports"].extend(f"FlowResponseSurprise.0.{family}.*@2330" for family in flow["exports"])
    parity = run(document, "parity")
    oracle = pq.read_table(root / "benchmark/replay-2/data" / day / "2330/values.parquet")
    mismatches = [name for name in oracle.column_names if not same_values(oracle[name], parity[name])]
    if mismatches:
        raise AssertionError(f"Unchanged native columns differ: {mismatches}")

    modules = copy.deepcopy(read_yaml(root / "benchmark/metadata.yaml")["modules"]) + [flow]
    cross = copy.deepcopy(profile["node_parameters"]["plan"]["additional_modules"][0])
    cross["spec"]["Subscribe"] = [{"Book": ["2330", "2317", "2454"]}]
    modules.append(cross)
    plan = SimpleNamespace(
        metadata={
            "engine": engine,
            "modules": modules,
            "phase": "baseline_diagnostic",
            "target_kind": "mid_return",
            "horizon_seconds": 300,
            "material_horizons": [60, 120, 180, 300],
            "sample_interval_seconds": 10,
            "label_margin_ticks": 0.0,
        }
    )
    inventory = SimpleNamespace(root=work, metadata={"engine": engine, "symbols": ["2330"]})
    ConfigKernel().execute(SimpleNamespace(profile=profile, work=work), {"inventory": inventory, "plan": plan}, {})
    document = read_yaml(work / "config.yaml")
    full = run(document, "full")
    endpoint_checks = {}
    for horizon in (60, 120, 180, 300):
        move = full[f"mid_return_ticks[{horizon}s]"].to_numpy()
        up = full[f"mid_endpoint.up.5[{horizon}s]"].to_numpy()
        down = full[f"mid_endpoint.down.5[{horizon}s]"].to_numpy()
        if not np.array_equal(np.isfinite(move), np.isfinite(up)) or not np.array_equal(np.isfinite(up), np.isfinite(down)):
            raise AssertionError("Native endpoint availability differs")
        valid = np.isfinite(move)
        np.testing.assert_array_equal(up[valid], (move[valid] >= 5).astype(float))
        np.testing.assert_array_equal(down[valid], (move[valid] <= -5).astype(float))
        endpoint_checks[horizon] = {"rows": int(valid.sum()), "up": int(up[valid].sum()), "down": int(down[valid].sum())}

    cutoff = int(dt.datetime(2026, 3, 2, 11, 0, tzinfo=ZoneInfo("Asia/Taipei")).timestamp() * 1e6)
    prefix_root = work / "source-prefix"
    prefix_counts = {}
    for symbol in ("2330", "2317", "2454"):
        destination = prefix_root / symbol / f"{day}.bin"
        destination.parent.mkdir(parents=True)
        source = Path(profile["paths"]["market_data"][0]) / symbol / f"{day}.bin.zst"
        decompressed = destination.with_suffix(".full.bin")
        with decompressed.open("wb") as stream:
            subprocess.run(["zstd", "-dc", str(source)], check=True, stdout=stream)
        prefix_counts[symbol] = int(subprocess.check_output([str(args.prefix_cutter.resolve()), str(decompressed), str(destination), str(cutoff)], text=True))
        decompressed.unlink()
    document["Modules"][0]["Decl"][-1]["Spec"]["Dirs"] = [str(prefix_root)]
    prefix = run(document, "prefix")
    # The clock stops at the final prefix event, before an untriggered future timer.
    last_origin = int(np.asarray(prefix["SampleTime"])[-1])
    before = full.filter(np.asarray(full["SampleTime"]) <= last_origin)
    # A prefix keeps pending endpoints but cannot invent missing future labels.
    feature_names = [name for name in full.column_names if name not in {name for name in full.column_names if "[" in name}]
    if len(prefix) != len(before):
        raise AssertionError(f"Prefix origins differ: {len(prefix)} vs {len(before)}")
    bad = [name for name in feature_names if not same_values(before[name], prefix[name])]
    if bad:
        raise AssertionError(f"Future-prefix feature differences: {bad}")
    report = {
        "engine": engine,
        "parity": {"rows": len(oracle), "bitwise_identical_columns": len(oracle.column_names)},
        "native_endpoint_checks": endpoint_checks,
        "prefix": {"rows": len(prefix), "bitwise_identical_feature_and_key_columns": len(feature_names), "counts": prefix_counts},
        "timings_seconds": timings,
        "new_features": [name for name in full.column_names if name.startswith(("FlowResponseSurprise.", "CrossReturnContext."))],
    }
    write_yaml(work / "results.yaml", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
