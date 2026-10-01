"""Frozen native-material comparisons; nomination precedes forward scoring."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import pyarrow
from catboost import CatBoostClassifier, CatBoostRegressor, FeaturesData, Pool
from mechanism import describe, permute_native_finite
from study_data import HORIZONS, census, native_classes, nonoverlap, numeric_domain, project, project_nominal, tail_selection

from AstraResearch.contracts import ArtifactRef
from AstraResearch.Experiments.signal.metrics import average_precision, evaluate, line
from AstraResearch.io import ContractError, digest, file_hash, lock, read_yaml, write_yaml
from AstraResearch.kernels.funnel import numeric_features
from AstraResearch.statistics import block_interval
from AstraResearch.store import Store

CONTEXT = {
    "CurrentBook",
    "BookStructure",
    "RealizedVolatility",
    "ReturnVolatility",
    "IntradayTime",
    "TimeInfo",
    "PriceLimitDistance",
    "PricePath",
    "MidPriceEma",
    "MidPriceChange",
    "PriceChange",
    "SessionReturnContext",
    "StickyPriceRealizedVolatility",
    "VolatilityTermStructure",
    "TickRegime",
    "DailyContext",
    "RawDirectionPath",
}
PRESSURE = {
    "PressureResponseState",
    "BarrierWorkState",
    "RefillSurvivalPressure",
    "AbsorptionReleaseLifecycle",
    "QueueResponseLifecycleState",
    "TradeClusterPersistence",
    "SignedTradeClusterState",
    "SupplyResponseState",
}
PARAMS = {
    "iterations": 600,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_seed": 20260930,
    "task_type": "GPU",
    "devices": "0",
    "thread_count": 8,
    "nan_mode": "Max",
    "border_count": 64,
    "gpu_ram_part": 0.85,
    "allow_writing_files": False,
    "one_hot_max_size": 64,
    "verbose": 100,
}


def compact_arms(arms, nominal_arms, nominal_names):
    """An unavailable extension cannot create another identical challenger fit."""
    required = {
        "context",
        "current",
        "current_regression",
        "current_binary",
        "current_equal_day",
        "pmq",
        "pmq_binary",
        "pmq_regression",
        "quality_only",
        "quality_only_binary",
        "quality_only_regression",
    }

    def recipe(name):
        features, kind = arms[name]
        return (
            tuple(sorted(features)),
            kind,
            name == "current_equal_day",
            tuple(nominal_names) if name in nominal_arms else (),
            name if name.endswith("_permuted_control") else None,
        )

    retained = {name: arms[name] for name in arms if name in required}
    seen = {recipe(name): name for name in retained}
    duplicates = {}
    for name in arms:
        if name in retained:
            continue
        key = recipe(name)
        if key in seen:
            duplicates[name] = seen[key]
        else:
            retained[name] = arms[name]
            seen[key] = name
    return {name: arms[name] for name in arms if name in retained}, duplicates


def metrics(rows, scores, horizon):
    y = rows[f"mid_return_ticks[{horizon}s]"].to_numpy(dtype=float)
    day, symbol = rows.day.to_numpy(dtype=str), rows.symbol.to_numpy(dtype=str)
    full = evaluate(y, scores[:, 0], scores[:, 1], day, symbol, 5)
    chosen = nonoverlap(day, symbol, rows.SampleTime.to_numpy(), horizon)
    independent = evaluate(y[chosen], scores[chosen, 0], scores[chosen, 1], day[chosen], symbol[chosen], 5)
    full["nonoverlap"] = independent
    return full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pmq-selection", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preparation = output.parent / f"preparation-{args.material.name.removeprefix('material-')}-launch.json"
    if preparation.exists() and not (output / "preparation.yaml").exists():
        launch = json.loads(preparation.read_text())
        command = launch["command"]
        if Path(command[command.index("--output") + 1]).resolve() != output:
            raise ContractError("Preparation output differs from this study")
        print("Waiting for the existing train/tune preparation; no concurrent projection writes", flush=True)
        started = time.monotonic()
        while not (output / "preparation.yaml").exists():
            try:
                os.kill(launch["pid"], 0)
            except ProcessLookupError as error:
                raise ContractError("Train/tune preparation ended without a receipt; preserve its log and resume explicitly") from error
            if time.monotonic() - started > 3 * 3600:
                raise ContractError("Train/tune preparation exceeded its three-hour wait budget")
            time.sleep(10)
    profile = read_yaml(args.material / "profile.yaml")
    bindings = read_yaml(args.material / "baseline-state.yaml")["bindings"]
    if not all(role in bindings for role in ("train", "tune", "forward", "plan")):
        raise ContractError("Canonical material is not complete")
    store = Store(Path(profile["paths"]["store"]))
    datasets = {role: store.resolve(ArtifactRef(**bindings[role])) for role in ("train", "tune")}
    numeric, excluded = numeric_features(datasets["train"])
    write_yaml(output / "numeric-scope.yaml", {"features": numeric, "nominal_columns_excluded_from_numeric_comparison": excluded, "categorical_followup_required": bool(excluded)})
    numeric = numeric_domain(datasets["train"], numeric, output)
    data = {role: project(dataset, numeric, output / role) for role, dataset in datasets.items()}
    train_x, train_rows, health = data["train"]
    tune_x, tune_rows, _ = data["tune"]
    nominal_names = list(excluded)
    nominal_data = {}
    if nominal_names:
        nominal_data = {role: project_nominal(datasets[role], nominal_names, output / role, data[role][1]) for role in datasets}
        nominal_names = [name for name in nominal_names if nominal_data["train"][name].nunique(dropna=False) > 1]
        write_yaml(output / "nominal-scope.yaml", {"features": nominal_names, "source": "native_nominal_strings_no_ordinal_encoding"})
    for role, (_, rows, _) in data.items():
        census(rows, output, role)
        describe(data[role][0], rows, numeric, output, role)
    usable = set(health.loc[health.nonconstant & health.finite.ge(100), "feature"])
    current = [name for name in numeric if name in usable and not name.startswith(("FlowResponseSurprise.", "CrossReturnContext."))]
    flow = [name for name in numeric if name in usable and name.startswith("FlowResponseSurprise.")]
    cross = [name for name in numeric if name in usable and name.startswith("CrossReturnContext.")]
    context = [name for name in current if name.split(".")[0] in CONTEXT]
    pressure = [name for name in current if name.split(".")[0] in PRESSURE]
    if not context or not current:
        raise ContractError("A preregistered native control is unavailable")
    tail = tail_selection(train_x, train_rows, numeric, usable, output)
    arms = {
        "context": (context, "classifier"),
        "current_regression": (current, "regressor"),
        "current": (current, "classifier"),
        "current_binary": (current, "binary"),
        "current_equal_day": (current, "classifier"),
        "context_pressure": (list(dict.fromkeys(context + pressure)), "classifier"),
        "current_flow": (current + flow, "classifier"),
        "current_flow_binary": (current + flow, "binary"),
        "current_flow_surprise": (current + [name for name in flow if ".flow_surprise" in name], "classifier"),
        "current_flow_innovation": (current + [name for name in flow if ".response_innovation" in name], "classifier"),
        "current_flow_reference": (current + [name for name in flow if ".flow_surprise" in name or ".response_coupling" in name], "classifier"),
        "current_cross": (current + cross, "classifier"),
        "current_cross_target": (current + [name for name in cross if ".target_return_z" in name or ".target_sigma_bps." in name], "classifier"),
        "current_flow_cross": (current + flow + cross, "classifier"),
        "tail_selector": (tail, "classifier"),
    }
    if flow:
        arms["current_flow_permuted_control"] = (current + flow, "classifier")
        arms["current_flow_binary_permuted_control"] = (current + flow, "binary")
    if cross:
        arms["current_cross_permuted_control"] = (current + cross, "classifier")
    if args.pmq_selection:
        selected = read_yaml(args.pmq_selection)
        pmq = selected.get("features", [])
        write_yaml(output / "pmq-input.yaml", selected)
        write_yaml(
            output / "pmq-model-domain.yaml", {"excluded_columns": [name for name in pmq if name not in numeric], "reason": "train_frozen_numeric_domain_or_native_nominal_type"}
        )
        if pmq:
            arms["pmq"] = ([name for name in pmq if name in numeric], "classifier")
            arms["pmq_binary"] = ([name for name in pmq if name in numeric], "binary")
            arms["pmq_regression"] = ([name for name in pmq if name in numeric], "regressor")
        quality_path = args.pmq_selection.with_name("quality-only.yaml")
        if not quality_path.exists():
            raise ContractError("The preregistered exact quality-only PMQ control is missing")
        quality = read_yaml(quality_path)
        write_yaml(output / "quality-only-input.yaml", quality)
        selected_quality = [name for name in quality.get("features", []) if name in numeric]
        write_yaml(
            output / "quality-only-model-domain.yaml",
            {"excluded_columns": [name for name in quality.get("features", []) if name not in numeric], "reason": "train_frozen_numeric_domain_or_native_nominal_type"},
        )
        if selected_quality:
            arms["quality_only"] = (selected_quality, "classifier")
            arms["quality_only_binary"] = (selected_quality, "binary")
            arms["quality_only_regression"] = (selected_quality, "regressor")
    arms = {name: recipe for name, recipe in arms.items() if recipe[0]}
    nominal_arms = {"current_nominal", "current_nominal_binary", "current_nominal_flow_cross"} if nominal_names else set()
    if nominal_names:
        arms["current_nominal"] = (current, "classifier")
        arms["current_nominal_binary"] = (current, "binary")
        arms["current_nominal_flow_cross"] = (current + flow + cross, "classifier")
    arms, duplicate_arms = compact_arms(arms, nominal_arms, nominal_names)
    nominal_arms &= arms.keys()
    protocol = {
        "params": PARAMS,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "catboost": catboost.__version__},
        "primary_horizon": 300,
        "secondary_horizons": list(HORIZONS[:-1]),
        "native_threshold_ticks": 5,
        "arms": {name: {"features": features, "kind": kind} for name, (features, kind) in arms.items()},
        "omitted_identical_candidate_arms": duplicate_arms,
        "material_bindings": bindings,
        "nomination": "calibration_mean_AP_at_least_2_percent_relative_direction_AUC_and_within_SD_AUC_no_more_than_0.01_lower_net_precision_not_lower",
        "feature_rows": "all_origins_native_sentinels_preserved_train_supported_columns_float32_cast_only",
        "native_nominal_arms": sorted(nominal_arms),
        "native_nominal_features": nominal_names,
        "negative_controls": {
            "current_flow_permuted_control": "joint_finite_native_flow_values_within_symbol_day_preserves_all_native_availability_states_never_nominated",
            "current_flow_binary_permuted_control": "same_flow_permutation_with_matched_native_binary_objective_never_nominated",
            "current_cross_permuted_control": "joint_finite_native_cross_values_within_symbol_day_preserves_all_native_availability_states_never_nominated",
        },
        "native_integration_limitations": {
            "CrossReturnContext": "engineering_sampler_invariance_failed_on_2337_normalized_outputs_requires_native_fix_and_rematerialization_before_integration"
        },
        "fixed_objective_comparisons": {
            name + suffix: f"{name}_regression"
            for name in ("current", "pmq", "quality_only")
            for suffix in ("", "_binary")
            if name + suffix in arms and f"{name}_regression" in arms
        },
        "training_weights": {
            "default": "native_origin_day_symbol_balanced_recency",
            "recency_half_life_calendar_days": profile["baseline"]["recency_half_life_days"],
            "current_equal_day": "native_origin_day_symbol_balanced_without_recency",
            "class_reweighting": False,
        },
        "source_files": {path.name: file_hash(path) for path in (Path(__file__), Path(__file__).with_name("study_data.py"), Path(__file__).with_name("mechanism.py"))},
    }
    protocol_path = output / "protocol.yaml"
    if protocol_path.exists() and read_yaml(protocol_path) != protocol:
        raise ContractError("Comparison protocol changed; keep existing models and use a new attempt")
    write_yaml(protocol_path, protocol)
    positions = {name: index for index, name in enumerate(numeric)}
    allocation = read_yaml(args.material / "date-plan.yaml")["reference_dates"]
    early = tune_rows.day.isin(allocation["early_stopping"]).to_numpy()
    calibration = tune_rows.day.isin(allocation["calibration"]).to_numpy()
    if not early.any() or not calibration.any() or (early & calibration).any():
        raise ContractError("Separate native early-stopping/calibration dates are unavailable")
    negative_controls = {}

    def model_values(matrix, role, selected_rows, columns, name):
        result = matrix[np.ix_(selected_rows, columns)]
        controls = {"current_flow_permuted_control": flow, "current_flow_binary_permuted_control": flow, "current_cross_permuted_control": cross}
        if name in controls:
            control_features = controls[name]
            key = (name, role)
            if key not in negative_controls:
                native_values = matrix[:, [positions[feature] for feature in control_features]]
                negative_controls[key] = permute_native_finite(native_values, data[role][1])
            result[:, -len(control_features) :] = negative_controls[key][selected_rows]
        return result

    def fit_model(name, horizon, side=None):
        features, kind = arms[name]
        model_features = features + (nominal_names if name in nominal_arms else [])
        stem = f"{name}-{horizon}" + (f"-{side}" if side else "")
        path = output / f"{stem}.cbm"
        fit_path = output / f"{stem}-fit.yaml"
        recipe = {"protocol": digest(protocol), "horizon": horizon, "arm": name, "native_endpoint_side": side}
        columns = [positions[feature] for feature in features]
        estimator = CatBoostRegressor if kind == "regressor" else CatBoostClassifier
        loss = "Logloss" if kind == "binary" else "MultiClass" if kind == "classifier" else "RMSE"
        model = estimator(**PARAMS, loss_function=loss)
        if path.exists():
            if not fit_path.exists() or read_yaml(fit_path).get("recipe") != recipe:
                raise ContractError("Cached model training recipe differs from the frozen comparison")
            model.load_model(str(path))
            if model.feature_names_ != model_features:
                raise ContractError("Cached model feature recipe differs from the frozen comparison")
        else:
            train_classes, known = native_classes(train_rows, horizon)
            early_classes, early_known = native_classes(tune_rows, horizon)
            valid_early = early & early_known
            if set(train_classes[known]) != {0, 1, 2} or set(early_classes[valid_early]) != {0, 1, 2}:
                raise ContractError("Frozen comparison target lacks native three-class support")
            target = train_classes[known] if kind == "classifier" else train_rows[f"mid_return_ticks[{horizon}s]"].to_numpy()[known]
            early_target = early_classes[valid_early] if kind == "classifier" else tune_rows[f"mid_return_ticks[{horizon}s]"].to_numpy()[valid_early]
            if kind == "binary":
                # Consume the original native indicator, never derive a binary
                # training label by thresholding a return in Python.
                target = train_rows[f"mid_endpoint.{side}.5[{horizon}s]"].to_numpy()[known]
                early_target = tune_rows[f"mid_endpoint.{side}.5[{horizon}s]"].to_numpy()[valid_early]
            tx = model_values(train_x, "train", np.flatnonzero(known), columns, name)
            ex = model_values(tune_x, "tune", np.flatnonzero(valid_early), columns, name)
            origin_groups = train_rows.loc[known, ["day", "symbol"]]
            counts = origin_groups.groupby(["day", "symbol"])["day"].transform("size").to_numpy()
            age = (pd.to_datetime(origin_groups.day.max()) - pd.to_datetime(origin_groups.day)).dt.days.to_numpy()
            weights = np.ones(len(age), dtype=float) if name == "current_equal_day" else np.exp2(-age / profile["baseline"]["recency_half_life_days"])
            weights /= counts
            weights /= weights.mean()
            if name in nominal_arms:
                tx = FeaturesData(
                    num_feature_data=tx,
                    cat_feature_data=nominal_data["train"].loc[known, nominal_names].to_numpy(dtype=object),
                    num_feature_names=features,
                    cat_feature_names=nominal_names,
                )
                ex = FeaturesData(
                    num_feature_data=ex,
                    cat_feature_data=nominal_data["tune"].loc[valid_early, nominal_names].to_numpy(dtype=object),
                    num_feature_names=features,
                    cat_feature_names=nominal_names,
                )
                training, validation = Pool(tx, label=target, weight=weights), Pool(ex, label=early_target)
            else:
                training = Pool(tx, label=target, weight=weights, feature_names=features)
                validation = Pool(ex, label=early_target, feature_names=features)
            start = time.monotonic()
            with lock(store.root / "gpu.lock"):
                model.fit(training, eval_set=validation, early_stopping_rounds=50, use_best_model=True)
            temporary_model = path.with_suffix(".tmp.cbm")
            model.save_model(str(temporary_model))
            write_yaml(
                fit_path,
                {
                    "recipe": recipe,
                    "model_features": model_features,
                    "feature_digest": digest(model_features),
                    "best_iteration": int(model.get_best_iteration()),
                    "seconds": time.monotonic() - start,
                    "train_rows": int(known.sum()),
                    "early_stopping_rows": int(valid_early.sum()),
                    "training_weight_range": [float(weights.min()), float(weights.max())],
                    "loss": loss,
                    "native_label": f"mid_endpoint.{side}.5[{horizon}s]"
                    if side
                    else f"native_endpoint_codebook[{horizon}s]"
                    if kind == "classifier"
                    else f"mid_return_ticks[{horizon}s]",
                    "train_dataset": datasets["train"].ref.document(),
                    "tune_dataset": datasets["tune"].ref.document(),
                },
            )
            temporary_model.replace(path)
            if kind == "binary":
                write_yaml(
                    output / f"{stem}-model-info.yaml",
                    {
                        "model": {"kind": "classifier"},
                        "features": model_features,
                        "categorical_features": nominal_names if name in nominal_arms else [],
                        "target": {
                            "column": f"mid_endpoint.{side}.5[{horizon}s]",
                            "unit": "probability",
                            "horizon_seconds": horizon,
                            "side": side,
                            "contract": {"kind": "endpoint_tail", "large_move_ticks": 5, "costs": "excluded"},
                        },
                    },
                )
            del tx, ex, training, validation
            gc.collect()
        return model

    def fit(name, horizon):
        if arms[name][1] == "binary":
            return [fit_model(name, horizon, side) for side in ("up", "down")]
        return fit_model(name, horizon)

    def predict(model, matrix, name, role):
        features, kind = arms[name]
        columns = [positions[feature] for feature in features]
        pieces = []
        for start in range(0, len(matrix), 16384):
            x = model_values(matrix, role, np.arange(start, min(len(matrix), start + 16384)), columns, name)
            if name in nominal_arms:
                cats = nominal_data[role].iloc[start : start + len(x)][nominal_names].to_numpy(dtype=object)
                x = FeaturesData(num_feature_data=x, cat_feature_data=cats, num_feature_names=features, cat_feature_names=nominal_names)
            if kind == "binary":
                if any(list(head.classes_) != [0, 1] for head in model):
                    raise ContractError("Native binary class codebook changed")
                pieces.append(np.column_stack([head.predict_proba(x, thread_count=8)[:, 1] for head in model]))
            elif kind == "classifier":
                probability = model.predict_proba(x, thread_count=8)
                if list(model.classes_) != [0, 1, 2]:
                    raise ContractError("Native class codebook changed")
                pieces.append(probability[:, [1, 2]])
            else:
                score = model.predict(x, thread_count=8)
                pieces.append(np.column_stack([score, -score]))
        result = np.concatenate(pieces)
        if not np.isfinite(result).all():
            raise ContractError("Model scores are unavailable on native origins")
        return result

    calibration_results = {}
    for name in arms:
        model = fit(name, 300)
        scores = predict(model, tune_x, name, "tune")
        np.save(output / f"{name}-300-tune-scores.npy", scores)
        result = metrics(tune_rows.loc[calibration].reset_index(drop=True), scores[calibration], 300)
        calibration_results[name] = result
        write_yaml(output / "calibration-results.yaml", calibration_results)
        print(line(name, 300, result), flush=True)
        del model
        gc.collect()
    control = calibration_results["current"]

    def qualifies(name):
        candidate = calibration_results[name]
        return (
            candidate["mean_ap"] >= 1.02 * control["mean_ap"]
            and candidate["direction_auc_given_big"] >= control["direction_auc_given_big"] - 0.01
            and candidate["mean_within_sd_auc"] >= control["mean_within_sd_auc"] - 0.01
            and candidate["mean_p@0.01"] - candidate["mean_wrong@0.01"] >= control["mean_p@0.01"] - control["mean_wrong@0.01"]
        )

    nominated = [
        name for name in arms if name not in {"context", "current", "context_pressure", "current_regression"} and not name.endswith("_permuted_control") and qualifies(name)
    ]
    challenger = max(nominated, key=lambda name: calibration_results[name]["mean_ap"]) if nominated else "current"
    write_yaml(
        output / "nomination.yaml",
        {
            "challenger": challenger,
            "control": "current",
            "selection_data": "native_calibration_dates_only",
            "calibration_dates": allocation["calibration"],
            "forward_read_before_nomination": False,
            "qualifying_candidates": nominated,
        },
    )
    # This is the first forward label/model access in this program.
    forward = store.resolve(ArtifactRef(**bindings["forward"]))
    forward_x, forward_rows, _ = project(forward, numeric, output / "forward")
    data["forward"] = (forward_x, forward_rows, None)
    if nominal_names:
        nominal_data["forward"] = project_nominal(forward, list(excluded), output / "forward", forward_rows)
    support = census(forward_rows, output, "forward")
    describe(forward_x, forward_rows, numeric, output, "forward")
    results, forward_scores = {}, {}
    for name in arms:
        model = fit(name, 300)
        scores = predict(model, forward_x, name, "forward")
        forward_scores[name] = scores
        np.save(output / f"{name}-300-forward-scores.npy", scores)
        results[name] = metrics(forward_rows, scores, 300)
        write_yaml(output / "forward-results-300.yaml", results)
        print(line(name, 300, results[name]), flush=True)
        del model
        gc.collect()
    paired = []
    days = sorted(forward_rows.day.unique())
    for day in days:
        indices = np.flatnonzero(forward_rows.day.to_numpy() == day)
        y = forward_rows["mid_return_ticks[300s]"].to_numpy()[indices]
        known = np.isfinite(y)
        record = {"day": str(day)}
        for name in {"current", challenger}:
            scores = forward_scores[name][indices][known]
            record[name] = float(np.nanmean([average_precision(y[known] >= 5, scores[:, 0]), average_precision(y[known] <= -5, scores[:, 1])]))
        record["delta_ap"] = record[challenger] - record["current"]
        paired.append(record)
    pd.DataFrame(paired).to_csv(output / "nominated-daily-paired-ap.csv", index=False)
    values = [record["delta_ap"] for record in paired if np.isfinite(record["delta_ap"])]
    interval = block_interval(values, 3, 20260930, draws=2048, tail_probability=0.05) if values else {"resolution_sufficient": False}
    primary_support = support.loc[support.horizon.eq(300)]
    support_sufficient = all(
        primary_support[f"nonoverlap_{side}"].sum() >= 100 and primary_support.loc[primary_support[f"nonoverlap_{side}"].gt(0), "day"].nunique() >= 15 for side in ("up", "down")
    )
    forward_control, forward_challenger = results["current"], results[challenger]
    checks = {
        "nominated_before_forward": challenger != "current",
        "date_resolution": interval.get("resolution_sufficient", False),
        "positive_daily_AP_bound": interval.get("lower", -1) > 0,
        "nonoverlap_event_support": support_sufficient,
        "direction_not_degraded": forward_challenger["direction_auc_given_big"] >= forward_control["direction_auc_given_big"] - 0.01,
        "within_symbol_day_not_degraded": forward_challenger["mean_within_sd_auc"] >= forward_control["mean_within_sd_auc"] - 0.01,
        "opposite_tail_precision_not_degraded": forward_challenger["mean_p@0.01"] - forward_challenger["mean_wrong@0.01"]
        >= forward_control["mean_p@0.01"] - forward_control["mean_wrong@0.01"],
        "nonoverlap_AP_not_degraded": forward_challenger["nonoverlap"]["mean_ap"] >= forward_control["nonoverlap"]["mean_ap"],
    }
    write_yaml(
        output / "primary-decision.yaml",
        {
            "challenger": challenger,
            "daily_equal_weight_paired_AP_block_interval": interval,
            "evidence": "development_forward_after_calibration_nomination",
            "pristine_OOS": False,
            "predictive_gain_supported": all(checks.values()),
            "checks": checks,
            "all_arm_forward_scores": "descriptive_fixed_family_not_selection",
            "nominal_features": "native_nominal_arms_use_original_strings_no_ordinal_map",
        },
    )
    objective_decisions = {}
    for classification, regression in protocol["fixed_objective_comparisons"].items():
        daily = []
        for day in days:
            indices = np.flatnonzero(forward_rows.day.to_numpy() == day)
            y = forward_rows["mid_return_ticks[300s]"].to_numpy()[indices]
            known = np.isfinite(y)
            ap = {}
            for name in (classification, regression):
                scores = forward_scores[name][indices][known]
                ap[name] = float(np.nanmean([average_precision(y[known] >= 5, scores[:, 0]), average_precision(y[known] <= -5, scores[:, 1])]))
            daily.append({"day": str(day), "classification": ap[classification], "regression": ap[regression], "delta_ap": ap[classification] - ap[regression]})
        pd.DataFrame(daily).to_csv(output / f"objective-{classification}-daily-paired-ap.csv", index=False)
        deltas = [record["delta_ap"] for record in daily if np.isfinite(record["delta_ap"])]
        bound = block_interval(deltas, 3, 20260930, draws=2048, tail_probability=0.05) if deltas else {"resolution_sufficient": False}
        cal_a, cal_b = calibration_results[classification], calibration_results[regression]
        a, b = results[classification], results[regression]
        objective_checks = {
            "calibration_AP_gain": cal_a["mean_ap"] >= 1.02 * cal_b["mean_ap"],
            "calibration_direction_not_degraded": cal_a["direction_auc_given_big"] >= cal_b["direction_auc_given_big"] - 0.01,
            "calibration_within_symbol_day_not_degraded": cal_a["mean_within_sd_auc"] >= cal_b["mean_within_sd_auc"] - 0.01,
            "calibration_net_tail_precision_not_degraded": cal_a["mean_p@0.01"] - cal_a["mean_wrong@0.01"] >= cal_b["mean_p@0.01"] - cal_b["mean_wrong@0.01"],
            "date_resolution": bound.get("resolution_sufficient", False),
            "positive_daily_AP_bound": bound.get("lower", -1) > 0,
            "nonoverlap_event_support": support_sufficient,
            "direction_not_degraded": a["direction_auc_given_big"] >= b["direction_auc_given_big"] - 0.01,
            "within_symbol_day_not_degraded": a["mean_within_sd_auc"] >= b["mean_within_sd_auc"] - 0.01,
            "opposite_tail_precision_not_degraded": a["mean_p@0.01"] - a["mean_wrong@0.01"] >= b["mean_p@0.01"] - b["mean_wrong@0.01"],
            "nonoverlap_AP_not_degraded": a["nonoverlap"]["mean_ap"] >= b["nonoverlap"]["mean_ap"],
        }
        objective_decisions[classification] = {
            "classification": classification,
            "regression": regression,
            "predictive_gain_supported": all(objective_checks.values()),
            "checks": objective_checks,
            "daily_equal_weight_paired_AP_block_interval": bound,
            "evidence": "predeclared_development_objective_comparison_no_forward_selection",
            "pristine_OOS": False,
        }
    write_yaml(output / "objective-decisions.yaml", objective_decisions)
    for horizon in HORIZONS[:-1]:
        secondary = {}
        for name in dict.fromkeys(["context", "current", challenger]):
            model = fit(name, horizon)
            scores = predict(model, forward_x, name, "forward")
            np.save(output / f"{name}-{horizon}-forward-scores.npy", scores)
            secondary[name] = metrics(forward_rows, scores, horizon)
            print(line(name, horizon, secondary[name]), flush=True)
            del model
            gc.collect()
        write_yaml(output / f"forward-results-{horizon}.yaml", secondary)
    write_yaml(
        output / "completed.yaml",
        {
            "completed": True,
            "material": str(args.material.resolve()),
            "challenger": challenger,
            "hypotheses": "H1_existing_pressure_H2_prior_innovation_H3_existing_cross_H4_selection_and_target",
            "execution_or_profitability_claims": False,
        },
    )


if __name__ == "__main__":
    main()
