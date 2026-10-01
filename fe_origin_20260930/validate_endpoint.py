"""Replay the admitted actual three-class model, preserving native inputs."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from catboost import FeaturesData
from study_models import endpoint_model
from validate_observation import compare_values

from AstraResearch.contracts import ArtifactRef
from AstraResearch.engine_identity import identity
from AstraResearch.io import ContractError, digest, file_hash, read_yaml, write_yaml
from AstraResearch.material import KEYS, fragments, same_values
from AstraResearch.native_contract import validate_receipt
from AstraResearch.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--arm", default="current_nominal")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--native-day", default="bc56e95608d67a79a28bad7f52762a8e11a25d5958ca94b8aef64fe70131d751")
    parser.add_argument("--symbols", nargs="+", default=["2609", "2337", "2481"])
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = read_yaml(args.material / "profile.yaml")
    protocol, numeric, nominal, model, model_file, info = endpoint_model(args.models, args.arm, args.horizon)
    if protocol["material_bindings"] != read_yaml(args.material / "baseline-state.yaml")["bindings"]:
        raise ContractError("Endpoint validation requires the exact producing material recipe")
    if any(name.startswith(("FlowResponseSurprise.", "CrossReturnContext.")) for name in numeric + nominal):
        raise ContractError("Corrected producer inputs require new scientific material before admission")
    write_yaml(output / "model-info.yaml", info)
    store = Store(Path(profile["paths"]["store"]))
    dataset = store.resolve(ArtifactRef("native_material_day", args.native_day))
    day = dataset.metadata["days"][0]
    parts = {part["symbol"]: part for part in dataset.metadata["partitions"]}
    binary = args.binary.resolve()
    result = {
        "mode": "endpoint_classifier",
        "day": day,
        "native_artifact": dataset.ref.document(),
        "original_engine": dataset.metadata["engine"],
        "engine": identity(binary)["fingerprint"],
        "study_arm": args.arm,
        "study_horizon": args.horizon,
        "study_protocol_digest": digest(protocol),
        "model_sha256": file_hash(model_file),
        "model_input_columns": numeric + nominal,
        "material_bindings": protocol["material_bindings"],
        "symbols": {},
    }
    for symbol in args.symbols:
        source = dataset.file(fragments(parts[symbol])[0])
        config_path = next(parent / "config.yaml" for parent in source.parents if (parent / "config.yaml").exists())
        document = read_yaml(config_path)
        group = next(group for group in document["Modules"] if group["Gid"] == symbol)
        writer = next(module["Spec"] for module in group["Decl"] if module["Desc"] == "DatasetWriter.0")
        group["Decl"].append(
            {
                "Desc": "CBPredictor.0",
                "Spec": {
                    "Gid": symbol,
                    "EndpointModelPath": str(model_file.resolve()),
                    "EndpointModelInfo": str(output / "model-info.yaml"),
                    "DropNegativeInfinity": False,
                    "DropNaN": False,
                },
            }
        )
        writer["MetadataExports"].extend(
            {"Feature": f"CBPredictor.0.prediction.{index}@{symbol}", "Name": f"NativeProbability{side.title()}"} for index, side in enumerate(("up", "down"))
        )
        folder = output / symbol
        folder.mkdir()
        write_yaml(folder / "config.yaml", document)
        started = time.monotonic()
        with (folder / "native.log").open("wb") as log:
            subprocess.run(
                [
                    str(binary),
                    "-d",
                    day,
                    "-C",
                    str(folder),
                    "--trading-calendar",
                    profile["paths"]["calendar"],
                    "--run-status-dir",
                    str(folder / "status"),
                    str(folder / "config.yaml"),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        validate_receipt(read_yaml(folder / "status" / f"{day}.yaml"), day)
        before, after = pq.read_table(source), pq.read_table(folder / "data" / day / symbol / "values.parquet")
        if len(before) != len(after) or any(not same_values(before[name], after[name]) for name in KEYS):
            raise ContractError("Endpoint scoring changes the original native origins")
        input_changes = compare_values(before, after, numeric + nominal)
        label_changes = compare_values(before, after, dataset.metadata["label_columns"])
        columns = list(dict.fromkeys([*KEYS, *dataset.metadata["feature_columns"], *dataset.metadata.get("metadata_columns", [])]))
        other_changes = compare_values(before, after, [name for name in columns if name not in numeric + nominal])
        if input_changes or label_changes or any(not name.startswith(("FlowResponseSurprise.", "CrossReturnContext.")) for name in other_changes):
            write_yaml(folder / "failed-input-parity.yaml", {"model_inputs": input_changes, "labels": label_changes, "other_changes": other_changes})
            raise ContractError("Endpoint scoring changed an input, label or undocumented producer")
        values = np.column_stack([np.asarray(after[name], dtype=np.float32) for name in numeric])
        if nominal:
            values = FeaturesData(
                num_feature_data=values, cat_feature_data=after.select(nominal).to_pandas().to_numpy(dtype=object), num_feature_names=numeric, cat_feature_names=nominal
            )
        predicted = model.predict_proba(values, thread_count=8)[:, [1, 2]]
        native = np.column_stack([np.asarray(after[f"NativeProbability{side}"]) for side in ("Up", "Down")])
        errors = {side: float(np.abs(native[:, index] - predicted[:, index]).max()) for index, side in enumerate(("up", "down"))}
        if not np.isfinite(native).all() or not np.allclose(native, predicted, rtol=1e-13, atol=1e-13):
            write_yaml(folder / "failed-score-parity.yaml", errors)
            raise ContractError("Actual native endpoint model probabilities differ from Python")
        result["symbols"][symbol] = {
            "common_origins": len(before),
            "model_input_columns_compared": len(numeric) + len(nominal),
            "model_input_bitwise_mismatches": input_changes,
            "native_label_bitwise_mismatches": label_changes,
            "other_native_feature_changes": other_changes,
            "native_score_parity": True,
            "score_maximum_absolute_errors": errors,
            "seconds": time.monotonic() - started,
        }
        write_yaml(output / "results.yaml", result)
        print(symbol, "actual endpoint parity", errors, "excluded producer changes", len(other_changes), flush=True)


if __name__ == "__main__":
    main()
