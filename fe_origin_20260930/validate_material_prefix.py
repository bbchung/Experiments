"""Future truncation against exact canonical native exports, on more symbols."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

from AstraResearch.contracts import ArtifactRef
from AstraResearch.engine_identity import identity
from AstraResearch.io import ContractError, file_hash, read_yaml, write_yaml
from AstraResearch.material import KEYS, fragments, same_values
from AstraResearch.native_contract import validate_receipt
from AstraResearch.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--native-day", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--cutoff", default="110000")
    parser.add_argument("--prefix-cutter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = read_yaml(args.material / "profile.yaml")
    store = Store(Path(profile["paths"]["store"]))
    dataset = store.resolve(ArtifactRef("native_material_day", args.native_day))
    day = dataset.metadata["days"][0]
    binary = profile["paths"]["coco_binary"]
    if identity(Path(binary))["fingerprint"] != dataset.metadata["engine"]:
        raise ContractError("Prefix validation needs the exact producing binary")
    clock = dt.datetime.strptime(day + args.cutoff, "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo("Asia/Taipei"))
    cutoff = int(clock.timestamp() * 1e6)
    parts = {part["symbol"]: part for part in dataset.metadata["partitions"]}
    original = store.resolve(ArtifactRef(**fragments(parts[args.symbols[0]])[0]["artifact"]))
    counts = {}
    prefix_root = args.output.resolve() / "source-prefix"
    for record in original.metadata["native_identity"]["sources"]:
        if not record["symbol"]:
            continue
        source = Path(record["path"])
        destination = prefix_root / source.parent.name / f"{day}.bin"
        destination.parent.mkdir(parents=True, exist_ok=True)
        decompressed = destination.with_suffix(".full.bin")
        with decompressed.open("wb") as output:
            subprocess.run(["zstd", "-dc", str(source)], check=True, stdout=output)
        counts[record["symbol"]] = int(subprocess.check_output([str(args.prefix_cutter.resolve()), str(decompressed), str(destination), str(cutoff)], text=True))
        decompressed.unlink()
    results = {}
    for symbol in args.symbols:
        fragment = fragments(parts[symbol])[0]
        source = dataset.file(fragment)
        config_path = next(parent / "config.yaml" for parent in source.parents if (parent / "config.yaml").is_file())
        document = read_yaml(config_path)
        module = next(declaration for group in document["Modules"] for declaration in group["Decl"] if declaration["Desc"] == "TradeBookMd.0")
        module["Spec"]["Dirs"] = [str(prefix_root)]
        folder = args.output.resolve() / symbol
        folder.mkdir()
        write_yaml(folder / "config.yaml", document)
        started = time.monotonic()
        with (folder / "native.log").open("wb") as log:
            subprocess.run(
                [binary, "-d", day, "-C", str(folder), "--trading-calendar", profile["paths"]["calendar"], "--run-status-dir", str(folder / "status"), str(folder / "config.yaml")],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        validate_receipt(read_yaml(folder / "status" / f"{day}.yaml"), day)
        prefix = pq.read_table(folder / "data" / day / symbol / "values.parquet")
        full = pq.read_table(source)
        times = np.asarray(prefix["SampleTime"])
        before = full.filter(np.asarray(full["SampleTime"]) <= times[-1])
        columns = [*KEYS, *dataset.metadata["feature_columns"], *dataset.metadata.get("metadata_columns", [])]
        if len(before) != len(prefix):
            raise AssertionError(f"Prefix origin population differs: {symbol}")
        mismatches = [name for name in dict.fromkeys(columns) if not same_values(before[name], prefix[name])]
        if mismatches:
            write_yaml(folder / "mismatches.yaml", mismatches)
            raise AssertionError(f"Future truncation changed feature values: {symbol}; {len(mismatches)} columns")
        labels = {}
        for name in dataset.metadata["label_columns"]:
            horizon = int(name.split("[")[1].removesuffix("s]"))
            matured = times + horizon * 1_000_000 < times[-1]
            if not same_values(before[name].filter(matured), prefix[name].filter(matured)):
                raise AssertionError(f"Matured native endpoint changed: {symbol}/{name}")
            labels[name] = int(matured.sum())
        results[symbol] = {
            "feature_columns_compared": len(set(columns)),
            "origins_compared": len(prefix),
            "matured_native_labels_compared": labels,
            "original_config_sha256": file_hash(config_path),
            "seconds": time.monotonic() - started,
        }
        write_yaml(
            args.output / "results.yaml",
            {
                "native_artifact": dataset.ref.document(),
                "engine": dataset.metadata["engine"],
                "day": day,
                "cutoff": args.cutoff,
                "source_prefix_records": counts,
                "symbols": results,
                "all_feature_values_bitwise_equal": True,
            },
        )
        print(symbol, "prefix passed", len(prefix), "origins", flush=True)


if __name__ == "__main__":
    main()
