"""Validate fitted study heads before native scoring or baseline packaging."""

from __future__ import annotations

from pathlib import Path

from catboost import CatBoostClassifier

from AstraResearch.io import ContractError, digest, file_hash, read_yaml


def binary_heads(directory: Path, arm: str, horizon: int):
    protocol = read_yaml(directory / "protocol.yaml")
    recipe = protocol["arms"].get(arm)
    if not recipe or recipe["kind"] != "binary" or arm.endswith("_permuted_control"):
        raise ContractError("Native scoring requires an ordinary, predeclared binary arm")
    if horizon not in [protocol["primary_horizon"], *protocol["secondary_horizons"]]:
        raise ContractError("Native scoring horizon differs from the frozen study")
    numeric = recipe["features"]
    nominal = protocol["native_nominal_features"] if arm in protocol["native_nominal_arms"] else []
    features = numeric + nominal
    models, files, infos = {}, {}, {}
    for side in ("up", "down"):
        stem = f"{arm}-{horizon}-{side}"
        path = directory / f"{stem}.cbm"
        fit = read_yaml(directory / f"{stem}-fit.yaml")
        expected = {"protocol": digest(protocol), "horizon": horizon, "arm": arm, "native_endpoint_side": side}
        label = f"mid_endpoint.{side}.5[{horizon}s]"
        if (
            fit["recipe"] != expected
            or fit["model_features"] != features
            or fit["feature_digest"] != digest(features)
            or fit["loss"] != "Logloss"
            or fit["native_label"] != label
            or fit["train_dataset"] != protocol["material_bindings"]["train"]
            or fit["tune_dataset"] != protocol["material_bindings"]["tune"]
        ):
            raise ContractError("Fitted head differs from the frozen native feature/label recipe")
        model = CatBoostClassifier().load_model(str(path))
        if list(model.classes_) != [0, 1] or model.feature_names_ != features or model.get_cat_feature_indices() != list(range(len(numeric), len(features))):
            raise ContractError("Fitted head differs from the native binary input/class codebook")
        info = read_yaml(directory / f"{stem}-model-info.yaml")
        if (
            info["model"]["kind"] != "classifier"
            or info["features"] != features
            or info["categorical_features"] != nominal
            or info["target"]["column"] != label
            or info["target"]["side"] != side
            or info["target"]["horizon_seconds"] != horizon
            or info["target"]["contract"] != {"kind": "endpoint_tail", "large_move_ticks": 5, "costs": "excluded"}
        ):
            raise ContractError("Model-info differs from the fitted pure-signal head")
        models[side], files[side], infos[side] = model, path, info
    return protocol, numeric, nominal, models, files, infos


def endpoint_model(directory: Path, arm: str, horizon: int):
    """Use the actual three-class native codebook, not an inferred binary head."""
    protocol = read_yaml(directory / "protocol.yaml")
    recipe = protocol["arms"].get(arm)
    if not recipe or recipe["kind"] != "classifier" or arm.endswith("_permuted_control"):
        raise ContractError("Endpoint scoring requires an ordinary predeclared three-class arm")
    if horizon not in [protocol["primary_horizon"], *protocol["secondary_horizons"]]:
        raise ContractError("Endpoint scoring horizon differs from the frozen comparison")
    numeric = recipe["features"]
    nominal = protocol["native_nominal_features"] if arm in protocol["native_nominal_arms"] else []
    features = numeric + nominal
    stem = f"{arm}-{horizon}"
    path = directory / f"{stem}.cbm"
    fit = read_yaml(directory / f"{stem}-fit.yaml")
    expected = {"protocol": digest(protocol), "horizon": horizon, "arm": arm, "native_endpoint_side": None}
    if (
        fit["recipe"] != expected
        or fit["model_features"] != features
        or fit["feature_digest"] != digest(features)
        or fit["loss"] != "MultiClass"
        or fit["native_label"] != f"native_endpoint_codebook[{horizon}s]"
        or fit["train_dataset"] != protocol["material_bindings"]["train"]
        or fit["tune_dataset"] != protocol["material_bindings"]["tune"]
    ):
        raise ContractError("Endpoint model differs from its frozen native fitting contract")
    model = CatBoostClassifier().load_model(str(path))
    if list(model.classes_) != [0, 1, 2] or model.feature_names_ != features or model.get_cat_feature_indices() != list(range(len(numeric), len(features))):
        raise ContractError("Endpoint model changed its feature order, native categories or class codebook")
    info = {
        "model": {"kind": "classifier", "class_codebook": {"none": 0, "up": 1, "down": 2}},
        "features": features,
        "categorical_features": nominal,
        "target": {
            "column": f"native_endpoint_codebook[{horizon}s]",
            "unit": "probability",
            "horizon_seconds": horizon,
            "contract": {"kind": "endpoint_tail", "large_move_ticks": 5, "costs": "excluded"},
            "native_indicators": {"up": f"mid_endpoint.up.5[{horizon}s]", "down": f"mid_endpoint.down.5[{horizon}s]"},
        },
    }
    return protocol, numeric, nominal, model, path, info


def endpoint_admission(directory: Path, arm: str, parity: dict, model_file: Path):
    completed = read_yaml(directory / "completed.yaml")
    protocol, numeric, nominal, _, expected_file, _ = endpoint_model(directory, arm, 300)
    primary = read_yaml(directory / "primary-decision.yaml")
    if (
        not completed.get("completed")
        or protocol["primary_horizon"] != 300
        or expected_file != model_file
        or primary["challenger"] != arm
        or not primary["predictive_gain_supported"]
        or not primary["checks"]
        or not all(primary["checks"].values())
    ):
        raise ContractError("The frozen nomination did not support this endpoint baseline")
    if any(name.startswith(("FlowResponseSurprise.", "CrossReturnContext.")) for name in numeric + nominal):
        raise ContractError("Amended producer inputs require a fresh scientific material lineage")
    if (
        parity.get("mode") != "endpoint_classifier"
        or parity.get("study_arm") != arm
        or parity.get("study_protocol_digest") != digest(protocol)
        or parity.get("study_horizon") != 300
        or parity.get("model_sha256") != file_hash(model_file)
        or parity.get("model_input_columns") != numeric + nominal
        or parity.get("material_bindings") != protocol["material_bindings"]
        or not parity.get("symbols")
        or any(
            not record.get("native_score_parity")
            or record.get("common_origins", 0) <= 0
            or record.get("model_input_columns_compared") != len(numeric) + len(nominal)
            or "model_input_bitwise_mismatches" not in record
            or record.get("model_input_bitwise_mismatches")
            or "native_label_bitwise_mismatches" not in record
            or record.get("native_label_bitwise_mismatches")
            or any(not name.startswith(("FlowResponseSurprise.", "CrossReturnContext.")) for name in record.get("other_native_feature_changes", {}))
            for record in parity["symbols"].values()
        )
    ):
        raise ContractError("Endpoint baseline admission lacks exact-model native input/score/label parity")
    return primary


def admission(directory: Path, arm: str, parity: dict, model_files: dict):
    """A native-compatible model still needs the study's actual admission evidence."""
    completed = read_yaml(directory / "completed.yaml")
    protocol = read_yaml(directory / "protocol.yaml")
    if not completed.get("completed") or protocol["arms"].get(arm, {}).get("kind") != "binary" or arm.endswith("_permuted_control"):
        raise ContractError("Baseline admission requires a completed ordinary binary comparison")
    if any(name.startswith("CrossReturnContext.") for name in protocol["arms"][arm]["features"]):
        raise ContractError("Affected cross-context inputs require corrected material lineage before integration")
    primary = read_yaml(directory / "primary-decision.yaml")
    objectives = read_yaml(directory / "objective-decisions.yaml")
    evidence = primary if primary.get("challenger") == arm and primary.get("predictive_gain_supported") else objectives.get(arm)
    if not evidence or not evidence.get("predictive_gain_supported") or not evidence.get("checks") or not all(evidence["checks"].values()):
        raise ContractError("The fixed comparison did not support this baseline improvement")
    if (
        parity.get("mode") != "predictor"
        or parity.get("study_arm") != arm
        or parity.get("study_protocol_digest") != digest(protocol)
        or parity.get("study_horizon") != protocol["primary_horizon"]
        or parity.get("model_sha256") != {side: file_hash(path) for side, path in model_files.items()}
        or not parity.get("symbols")
        or any(
            not variants
            or any(
                not record.get("native_score_parity") or record.get("feature_bitwise_mismatches") or record.get("native_label_bitwise_mismatches") for record in variants.values()
            )
            for variants in parity["symbols"].values()
        )
    ):
        raise ContractError("Baseline admission lacks native parity for these exact fitted models")
    return evidence
