"""Engineering checks for sampler dependence and native directional scoring."""

from __future__ import annotations

import argparse
import copy
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, FeaturesData

from AstraResearch.contracts import ArtifactRef
from AstraResearch.engine_identity import identity
from AstraResearch.io import ContractError, digest, file_hash, read_yaml, write_yaml
from AstraResearch.material import KEYS, fragments, same_values
from AstraResearch.native_contract import validate_receipt
from AstraResearch.store import Store


def compare_values(before, after, columns):
    changed = {}
    for name in columns:
        if same_values(before[name], after[name]):
            continue
        record = {"before_type": str(before[name].type)}
        if str(before[name].type) in {"double", "float"}:
            x, y = np.asarray(before[name]), np.asarray(after[name])
            finite = np.isfinite(x) & np.isfinite(y)
            equal = (x == y) | (np.isnan(x) & np.isnan(y))
            record.update(
                rows_changed=int((~equal).sum()),
                finite_rows_changed=int((finite & ~equal).sum()),
                maximum_absolute_difference=float(np.abs(x[finite] - y[finite]).max()) if finite.any() else None,
                maximum_relative_difference=float((np.abs(x[finite] - y[finite]) / np.maximum(1, np.abs(x[finite]))).max()) if finite.any() else None,
                availability_changed=int(((np.isfinite(x) != np.isfinite(y)) | (np.isnan(x) != np.isnan(y)) | (np.isneginf(x) != np.isneginf(y))).sum()),
            )
        changed[name] = record
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--native-day", default="bc56e95608d67a79a28bad7f52762a8e11a25d5958ca94b8aef64fe70131d751")
    parser.add_argument("--native-kind", choices=["native_material_day", "native_export_day"], default="native_material_day")
    parser.add_argument("--symbols", nargs="+", default=["2609", "2337"])
    parser.add_argument("--mode", choices=["sampler", "predictor"], required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--study-arm", help="Validate the exact fitted ordinary binary arm, instead of the engineering models")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--binary", type=Path, help="Separate engineering binary; sampler mode only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profile = read_yaml(args.material / "profile.yaml")
    store = Store(Path(profile["paths"]["store"]))
    dataset = store.resolve(ArtifactRef(args.native_kind, args.native_day))
    day = dataset.metadata["days"][0]
    binary = args.binary.resolve() if args.binary else Path(profile["paths"]["coco_binary"])
    engine = identity(binary)["fingerprint"]
    if args.binary and args.mode != "sampler":
        raise ContractError("An amended engineering binary needs its own sampler reference")
    if args.study_arm and args.mode != "predictor":
        raise ContractError("An actual study arm needs predictor validation")
    if not args.binary and engine != dataset.metadata["engine"]:
        raise ContractError("Observation validation needs the exact producing binary")
    parts = {part["symbol"]: part for part in dataset.metadata["partitions"]}
    results = {"native_artifact": dataset.ref.document(), "engine": engine, "original_engine": dataset.metadata["engine"], "day": day, "mode": args.mode, "symbols": {}}
    models, numeric, nominal = {}, [], []
    if args.mode == "predictor":
        if args.models is None:
            raise ContractError("Predictor validation needs original model inputs")
        import json

        if args.study_arm:
            from study_models import binary_heads

            protocol, numeric, nominal, models, model_files, model_infos = binary_heads(args.models, args.study_arm, args.horizon)
            if protocol["material_bindings"]["train"] != read_yaml(args.material / "baseline-state.yaml")["bindings"]["train"]:
                raise ContractError("Study models were fitted on another material recipe")
            results.update(study_arm=args.study_arm, study_horizon=args.horizon, study_protocol_digest=digest(protocol))
        else:
            numeric = json.loads((args.models / "projection.json").read_text())["recipe"]["features"]
            nominal = read_yaml(args.models / "nominal-projection.yaml")["recipe"]["features"]
            model_files = {side: args.models / f"binary-{side}.cbm" for side in ("up", "down")}
            models = {side: CatBoostClassifier().load_model(str(path)) for side, path in model_files.items()}
            model_infos = {
                side: {"model": {"kind": "classifier"}, "features": numeric + nominal, "categorical_features": nominal, "target": {"side": side, "costs": "excluded"}}
                for side in models
            }
        for side in ("up", "down"):
            write_yaml(args.output / f"{side}-model-info.yaml", model_infos[side])
        results["model_sha256"] = {side: file_hash(path) for side, path in model_files.items()}
        results["predictive_evidence"] = False
    variants = ["five_second", "wider_session"] if args.mode == "sampler" else ["directional_classifier"]
    if args.binary:
        variants.insert(0, "reference_ten_second")
    for symbol in args.symbols:
        source = dataset.file(fragments(parts[symbol])[0])
        config_path = next(parent / "config.yaml" for parent in source.parents if (parent / "config.yaml").is_file())
        original = read_yaml(config_path)
        full = pq.read_table(source)
        results["symbols"][symbol] = {}
        for variant in variants:
            document = copy.deepcopy(original)
            group = next(group for group in document["Modules"] if group.get("Gid") == symbol)
            writer = next(module["Spec"] for module in group["Decl"] if module["Desc"] == "DatasetWriter.0")
            if variant == "five_second":
                writer["PeriodicSampler"]["SampleInterval"] = "5s"
            elif variant == "wider_session":
                writer["PeriodicSampler"].update(StartTime="090100", UntilTime="132000")
            elif variant == "directional_classifier":
                spec = {"Gid": symbol, "DropNegativeInfinity": False, "DropNaN": False}
                for side, title in [("up", "Up"), ("down", "Down")]:
                    spec[title + "ModelPath"] = str(model_files[side].resolve())
                    spec[title + "ModelInfo"] = str((args.output / f"{side}-model-info.yaml").resolve())
                group["Decl"].append({"Desc": "CBPredictor.0", "Spec": spec})
                writer["MetadataExports"].extend(
                    {"Feature": f"CBPredictor.0.prediction.{index}@{symbol}", "Name": f"NativeProbability{side.title()}"} for index, side in enumerate(("up", "down"))
                )
            folder = args.output.resolve() / symbol / variant
            folder.mkdir(parents=True)
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
            expanded = pq.read_table(folder / "data" / day / symbol / "values.parquet")
            indexes = np.searchsorted(np.asarray(expanded["SampleTime"]), np.asarray(full["SampleTime"]))
            after = expanded.take(indexes)
            if any(not same_values(full[name], after[name]) for name in KEYS):
                raise AssertionError("Common native origins differ")
            columns = list(dict.fromkeys([*KEYS, *dataset.metadata["feature_columns"], *dataset.metadata.get("metadata_columns", [])]))
            changed = compare_values(full, after, columns)
            label_changes = compare_values(full, after, dataset.metadata["label_columns"])
            record = {
                "original_sampler": next(m["Spec"]["PeriodicSampler"] for g in original["Modules"] for m in g["Decl"] if m["Desc"] == "DatasetWriter.0"),
                "checked_sampler": writer["PeriodicSampler"],
                "original_config_sha256": file_hash(config_path),
                "common_origins": len(full),
                "expanded_origins": len(expanded),
                "columns_compared": len(columns),
                "feature_bitwise_mismatches": changed,
                "native_label_bitwise_mismatches": label_changes,
                "seconds": time.monotonic() - started,
            }
            if args.mode == "predictor":
                numeric_values = np.column_stack([np.asarray(after[name], dtype=np.float32) for name in numeric])
                values = (
                    FeaturesData(
                        num_feature_data=numeric_values,
                        cat_feature_data=after.select(nominal).to_pandas().to_numpy(dtype=object),
                        num_feature_names=numeric,
                        cat_feature_names=nominal,
                    )
                    if nominal
                    else numeric_values
                )
                errors = {}
                for side in models:
                    native = np.asarray(after[f"NativeProbability{side.title()}"])
                    python = models[side].predict_proba(values)[:, 1]
                    if not np.isfinite(native).all():
                        raise AssertionError(f"Native directional scores are missing for {side}")
                    errors[side] = float(np.abs(native - python).max())
                    if not np.allclose(native, python, rtol=1e-13, atol=1e-13):
                        record["score_maximum_absolute_errors"] = errors
                        write_yaml(folder / "failed-comparison.yaml", record)
                        raise AssertionError(f"Native/Python directional score parity failed: {side}; {errors[side]}")
                record["score_maximum_absolute_errors"] = errors
                record["native_score_parity"] = True
                if changed or label_changes:
                    write_yaml(folder / "failed-comparison.yaml", record)
                    raise AssertionError("Predictor evaluation changed original native inputs or endpoints")
            results["symbols"][symbol][variant] = record
            write_yaml(args.output / "results.yaml", results)
            print(symbol, variant, "feature mismatches", len(changed), "label mismatches", len(label_changes), flush=True)
            if variant == "reference_ten_second":
                # Record the versioned change against the old native artifact,
                # then compare amended schedules to this binary's own reference.
                full = expanded


if __name__ == "__main__":
    main()
