"""One frozen two-stage objective comparison over the parent's native material."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from factorized_contract import side_scores, targets, weights
from study import PARAMS, metrics
from study_data import nonoverlap
from study_models import binary_heads

from AstraResearch.Experiments.signal.metrics import average_precision, line
from AstraResearch.io import ContractError, digest, file_hash, lock, read_yaml, write_yaml
from AstraResearch.statistics import block_interval


def load_projection(parent, role, binding, numeric):
    directory = parent / role
    document = json.loads((directory / "projection.json").read_text())
    if document["recipe"]["dataset"] != binding or document["recipe"]["features"] != numeric:
        raise ContractError("Factorized projection differs from the frozen parent material")
    if any(file_hash(directory / name) != signature for name, signature in document["files"].items()):
        raise ContractError("Factorized projection changed after its native receipt")
    return np.load(directory / "x.npy", mmap_mode="r"), pd.read_parquet(directory / "rows.parquet")


def qualify(candidate, control):
    return {
        "AP_gain": candidate["mean_ap"] >= 1.02 * control["mean_ap"],
        "direction_gain": candidate["direction_auc_given_big"] >= control["direction_auc_given_big"] + 0.01,
        "within_symbol_day_not_degraded": candidate["mean_within_sd_auc"] >= control["mean_within_sd_auc"] - 0.01,
        "net_tail_precision_not_degraded": candidate["mean_p@0.01"] - candidate["mean_wrong@0.01"] >= control["mean_p@0.01"] - control["mean_wrong@0.01"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    parent, output = args.parent.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = read_yaml(parent / "protocol.yaml")
    profile = read_yaml(args.material / "profile.yaml")
    bindings = read_yaml(args.material / "baseline-state.yaml")["bindings"]
    if original["material_bindings"] != bindings or original["native_threshold_ticks"] != 5:
        raise ContractError("Factorized comparison requires the exact native five-tick parent")
    _, features, nominal, _, control_files, _ = binary_heads(parent, "current_binary", 300)
    if nominal:
        raise ContractError("The fixed factorized treatment uses numeric current features only")
    domain = read_yaml(parent / "numeric-domain.yaml")
    numeric = domain["supported"]
    allocation = read_yaml(args.material / "date-plan.yaml")["reference_dates"]
    here = Path(__file__).resolve().parent
    if original["params"] != PARAMS or any(file_hash(here / name) != signature for name, signature in original["source_files"].items()):
        raise ContractError("The factorized treatment requires the unchanged parent scientific sources")
    protocol = {
        "hypothesis": "H5_factorized_endpoint_magnitude_and_conditional_direction",
        "control": "current_binary",
        "parent_protocol": digest(original),
        "material_bindings": bindings,
        "params": {**PARAMS, "loss_function": "Logloss"},
        "features": features,
        "numeric_domain": numeric,
        "horizon_seconds": 300,
        "threshold_ticks": 5,
        "allocation": allocation,
        "weighting": "all_known_native_origin_symbol_day_balanced_recency_then_condition_and_mean_normalize",
        "recency_half_life_days": profile["baseline"]["recency_half_life_days"],
        "target_codebooks": {"event": "logical_union_of_original_native_up_down_indicators", "direction": "original_native_up_indicator_on_native_big_event_train_and_ES_rows"},
        "score_composition": "up=P_big*P_up_given_big;down=P_big*(1-P_up_given_big)",
        "calibration": "AP_2_percent_relative_direction_AUC_plus_0.01_within_SD_loss_at_most_0.01_net_top1_precision_not_lower",
        "forward": "paired_daily_AP_3day_block_2048_draw_5percent_lower_positive_direction_AUC_plus_0.01_existing_support_and_ranking_guards",
        "control_model_sha256": {side: file_hash(path) for side, path in control_files.items()},
        "parent_projection_sha256": {role: file_hash(parent / role / "projection.json") for role in ["train", "tune"]},
        "source_files": {p.name: file_hash(p) for p in [Path(__file__), here / "factorized_contract.py", here / "FACTORIZED_PLAN.md"]},
        "parent_scientific_sources": original["source_files"],
        "adaptive_motivation": "observed_parent_stock_calibration_before_stock_forward_model_scoring",
        "pristine_OOS": False,
        "native_route": "requires_explicit_factorized_composition_actual_model_parity_before_integration",
    }
    protocol_path = output / "protocol.yaml"
    if protocol_path.exists():
        if read_yaml(protocol_path) != protocol:
            raise ContractError("Factorized protocol changed; preserve the attempt and create a new one")
    else:
        if (parent / "nomination.yaml").exists() or (parent / "forward/projection.json").exists():
            raise ContractError("Freeze this adaptive comparison before the parent stock forward read")
        write_yaml(protocol_path, protocol)
        write_yaml(
            output / "freeze-receipt.yaml",
            {
                "utc": dt.datetime.now(dt.UTC).isoformat(),
                "parent_nomination_exists": False,
                "parent_forward_projection_exists": False,
                "model_fitted": False,
                "protocol_digest": digest(protocol),
            },
        )
    if args.freeze_only:
        print("H5 protocol frozen before parent stock forward model access", flush=True)
        return
    deadline = time.monotonic() + 8 * 3600
    while not (parent / "completed.yaml").exists():
        if time.monotonic() >= deadline:
            raise ContractError("Parent study did not complete within the H5 memory-safe wait budget")
        time.sleep(20)
    data = {role: load_projection(parent, role, bindings[role], numeric) for role in ("train", "tune")}
    tx, train_rows = data["train"]
    ex, tune_rows = data["tune"]
    train, early = targets(train_rows), targets(tune_rows)
    early_dates = tune_rows.day.isin(allocation["early_stopping"]).to_numpy()
    cal = tune_rows.day.isin(allocation["calibration"]).to_numpy()
    columns = [numeric.index(name) for name in features]
    all_weights = weights(train_rows, train["known"], protocol["recency_half_life_days"])
    original_weight_rows = np.full(len(train_rows), np.nan)
    original_weight_rows[train["known"]] = all_weights
    models = {}
    for stage in ["event", "direction"]:
        path = output / f"{stage}.cbm"
        fit_path = output / f"{stage}-fit.yaml"
        if path.exists():
            fit = read_yaml(fit_path)
            if fit["protocol"] != digest(protocol) or file_hash(path) != fit["model_sha256"]:
                raise ContractError("Factorized fitted model does not match the frozen protocol")
            models[stage] = CatBoostClassifier().load_model(str(path))
            continue
        mask = train["known"] if stage == "event" else train["big"]
        valid = early_dates & (early["known"] if stage == "event" else early["big"])
        target = train["event_label"] if stage == "event" else train["direction_label"]
        validation_target = (early["big"][valid]).astype(np.int8) if stage == "event" else tune_rows["mid_endpoint.up.5[300s]"].to_numpy()[valid]
        if set(target) != {0, 1} or set(validation_target) != {0, 1}:
            raise ContractError("Factorized stage lacks both native training/ES classes")
        stage_weights = original_weight_rows[mask]
        stage_weights /= stage_weights.mean()
        values = np.asarray(tx[np.ix_(np.flatnonzero(mask), columns)], dtype=np.float32, order="C")
        validation = np.asarray(ex[np.ix_(np.flatnonzero(valid), columns)], dtype=np.float32, order="C")
        training = Pool(values, label=target, weight=stage_weights, feature_names=features)
        reference = Pool(validation, label=validation_target, feature_names=features)
        started = time.monotonic()
        model = CatBoostClassifier(**protocol["params"])
        with lock(Path(profile["paths"]["store"]) / "gpu.lock"):
            model.fit(training, eval_set=reference, early_stopping_rounds=50, use_best_model=True)
        if list(model.classes_) != [0, 1]:
            raise ContractError("Factorized model changed its binary positive class")
        model.save_model(str(path))
        write_yaml(
            fit_path,
            {
                "protocol": digest(protocol),
                "model_sha256": file_hash(path),
                "stage": stage,
                "best_iteration": int(model.get_best_iteration()),
                "seconds": time.monotonic() - started,
                "train_rows": int(mask.sum()),
                "early_stopping_rows": int(valid.sum()),
                "positive_train_rows": int(target.sum()),
                "weight_range": [float(stage_weights.min()), float(stage_weights.max())],
                "features": features,
            },
        )
        models[stage] = model
        del values, validation, training, reference
        gc.collect()

    def score(matrix):
        result = []
        for start in range(0, len(matrix), 16384):
            selected = matrix[start : start + 16384, columns]
            result.append(side_scores(models["event"].predict_proba(selected, thread_count=8)[:, 1], models["direction"].predict_proba(selected, thread_count=8)[:, 1]))
        return np.concatenate(result)

    tune_scores = score(ex)
    np.save(output / "factorized-300-tune-scores.npy", tune_scores)
    control_tune = np.load(parent / "current_binary-300-tune-scores.npy")
    if control_tune.shape != tune_scores.shape:
        raise ContractError("Factorized calibration origins differ from control")
    cal_rows = tune_rows.loc[cal].reset_index(drop=True)
    a, b = metrics(cal_rows, control_tune[cal], 300), metrics(cal_rows, tune_scores[cal], 300)
    cal_checks = qualify(b, a)
    write_yaml(output / "calibration-results.yaml", {"control": a, "factorized": b})
    write_yaml(
        output / "nomination.yaml",
        {"protocol": digest(protocol), "challenger": "factorized" if all(cal_checks.values()) else "current_binary", "checks": cal_checks, "forward_read_before_nomination": False},
    )
    print(line("factorized_calibration", 300, b), flush=True)
    fx, rows = load_projection(parent, "forward", bindings["forward"], numeric)
    scores = score(fx)
    control = np.load(parent / "current_binary-300-forward-scores.npy")
    if control.shape != scores.shape:
        raise ContractError("Factorized forward origins differ from control")
    np.save(output / "factorized-300-forward-scores.npy", scores)
    result_a, result_b = metrics(rows, control, 300), metrics(rows, scores, 300)
    write_yaml(output / "forward-results.yaml", {"control": result_a, "factorized": result_b})
    known = targets(rows)
    independent = nonoverlap(rows.day.to_numpy(), rows.symbol.to_numpy(), rows.SampleTime.to_numpy(), 300)
    supports = {}
    for side in ("up", "down"):
        positive = independent & known["known"] & rows[f"mid_endpoint.{side}.5[300s]"].eq(1).to_numpy()
        supports[side] = {"events": int(positive.sum()), "days": int(rows.loc[positive, "day"].nunique())}
    daily = []
    for day, indices in rows.groupby("day", sort=True).indices.items():
        y = rows["mid_return_ticks[300s]"].to_numpy()[indices]
        valid = np.isfinite(y)
        ap = []
        for value in (control, scores):
            s = value[indices][valid]
            ap.append(float(np.nanmean([average_precision(y[valid] >= 5, s[:, 0]), average_precision(y[valid] <= -5, s[:, 1])])))
        daily.append({"day": day, "control_ap": ap[0], "factorized_ap": ap[1], "delta_ap": ap[1] - ap[0]})
    pd.DataFrame(daily).to_csv(output / "daily-paired-ap.csv", index=False)
    bound = block_interval([row["delta_ap"] for row in daily if np.isfinite(row["delta_ap"])], 3, 20260930, draws=2048, tail_probability=0.05)
    checks = {
        "calibration_nomination": all(cal_checks.values()),
        "date_resolution": bound["resolution_sufficient"],
        "positive_daily_AP_bound": bound["lower"] > 0,
        "direction_gain": result_b["direction_auc_given_big"] >= result_a["direction_auc_given_big"] + 0.01,
        "within_symbol_day_not_degraded": result_b["mean_within_sd_auc"] >= result_a["mean_within_sd_auc"] - 0.01,
        "net_tail_precision_not_degraded": result_b["mean_p@0.01"] - result_b["mean_wrong@0.01"] >= result_a["mean_p@0.01"] - result_a["mean_wrong@0.01"],
        "nonoverlap_AP_not_degraded": result_b["nonoverlap"]["mean_ap"] >= result_a["nonoverlap"]["mean_ap"],
        "nonoverlap_event_support": all(v["events"] >= 100 and v["days"] >= 15 for v in supports.values()),
    }
    write_yaml(
        output / "decision.yaml",
        {
            "predictive_gain_supported": all(checks.values()),
            "checks": checks,
            "daily_equal_weight_paired_AP_block_interval": bound,
            "nonoverlap_support": supports,
            "pristine_OOS": False,
            "evidence": "adaptive_hypothesis_frozen_before_parent_stock_forward_scoring_separate_nomination",
            "native_baseline_integrated": False,
        },
    )
    print(line("factorized_forward", 300, result_b), flush=True)
    write_yaml(output / "completed.yaml", {"completed": True, "protocol": digest(protocol), "execution_or_profitability_claims": False})


if __name__ == "__main__":
    main()
