"""Bounded native H10 comparison; V1/V2 scientific modules remain immutable."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import pyarrow
import scipy
from catboost import CatBoostClassifier, FeaturesData, Pool

from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from ..representation_research import contract, data, evaluation
from ..representation_research import runner as inherited
from ..representation_research_v2.runner import JUDGE_FIELDS, assert_same_cohort, hardlink
from . import data as h10_data

output = inherited.output
signature = inherited.signature
predict = inherited.predict
load_role = h10_data.load_role
ARMS = {
    "baseline": {"base": "current_nominal", "families": []},
    "h10_ordered": {"base": "current_nominal", "families": ["h10_ordered"]},
    "h10_reset": {"base": "current_nominal", "families": ["h10_reset"]},
}


def parent_contract(profile):
    parent = contract.verify(Path(profile["inheritance"]["frozen_contract"]))
    if parent["identity"] != profile["inheritance"]["identity"]:
        raise ContractError("V3 parent scientific identity changed")
    if Path(profile["paths"]["parent_study"]).resolve() != Path(parent["profile"]["paths"]["parent_study"]).resolve():
        raise ContractError("V3 must retain the exact historical baseline projection input")
    for field in JUDGE_FIELDS:
        if profile[field] != parent["profile"][field]:
            raise ContractError(f"V3 changes the inherited primary300 judge: {field}")
    if profile["arms"] != ARMS or list(profile["arms"]) != list(ARMS) or profile["baseline"] != "baseline":
        raise ContractError("V3 requires exactly its three predeclared native H10 arms")
    if profile["compute"]["primary_fits"] != 3 or profile["compute"]["secondary_fits_maximum"] != 0 or not profile["compute"]["no_hpo"]:
        raise ContractError("V3 permits exactly three primary300 fits and no secondary fits or HPO")
    cfg = profile["h10"]
    if cfg["numeric_fields"] != list(h10_data.NUMERIC) or cfg["categorical_fields"] != list(h10_data.CATEGORICAL) or cfg["diagnostics_predictors"] is not False:
        raise ContractError("V3 predictors must be the seven native numeric fields and one phase category")
    if cfg["producers"] != {"h10_ordered": "DemandRepairRenewal.0", "h10_reset": "DemandRepairRenewal.1"} or cfg["native_keys"] != h10_data.KEYS[2:]:
        raise ContractError("V3 chronology variants and native observation keys changed")
    return parent


def prepare_features(profile):
    if (output(profile) / "frozen-contract.yaml").exists():
        raise ContractError("Frozen V3 feature artifacts cannot be prepared or overwritten")
    parent = parent_contract(profile)
    h10_data.native_validation(profile)
    original = Path(parent["profile"]["paths"]["output"]) / "shared"
    shared = output(profile) / "shared"
    # Bind existing light context caches without physically duplicating any dataset.
    for path in sorted(original.iterdir()):
        if path.is_file():
            hardlink(path, shared / path.name)
    return h10_data.prepare(profile)


def fit_one(profile, arm, horizon):
    root = output(profile)
    stem = f"{arm}-{horizon}"
    fit_path = root / "fits" / f"{stem}.yaml"
    model_path = root / "fits" / f"{stem}.cbm"
    frozen = read_yaml(root / "frozen-contract.yaml")
    if fit_path.exists():
        receipt = read_yaml(fit_path)
        if receipt["contract_digest"] != digest(frozen) or receipt["model_sha256"] != file_hash(model_path):
            raise ContractError("Resumed model belongs to another frozen comparison")
        model = CatBoostClassifier().load_model(str(model_path))
        expected_features = receipt["numeric_features"] + receipt["nominal_features"]
        if (
            list(model.classes_) != [0, 1, 2]
            or model.feature_names_ != expected_features
            or model.get_cat_feature_indices() != list(range(len(receipt["numeric_features"]), len(expected_features)))
        ):
            raise ContractError("Resumed model changed its endpoint classes, feature order or nominal schema")
        return model
    tr, tx, tc, names, cats = load_role(profile, "train", arm)
    es, ex, ec, enames, ecats = load_role(profile, "es", arm)
    if names != enames or cats != ecats:
        raise ContractError("Feature order changed across roles")
    train_signature = signature(profile, tr, horizon, "train")
    es_signature = signature(profile, es, horizon, "es")
    target, known = data.native_classes(tr, horizon)
    early, valid = data.native_classes(es, horizon)
    if set(target[known]) != {0, 1, 2} or set(early[valid]) != {0, 1, 2}:
        raise ContractError("Three-class endpoint support is insufficient")
    groups = tr.loc[known, ["day", "symbol"]]
    counts = groups.groupby(["day", "symbol"])["day"].transform("size").to_numpy()
    age = (pd.to_datetime(groups.day.max()) - pd.to_datetime(groups.day)).dt.days.to_numpy()
    weights = np.exp2(-age / profile["training"]["recency_half_life_days"]) / counts
    weights /= weights.mean()
    train_pool = Pool(FeaturesData(num_feature_data=tx[known], cat_feature_data=tc[known], num_feature_names=names, cat_feature_names=cats), label=target[known], weight=weights)
    early_pool = Pool(FeaturesData(num_feature_data=ex[valid], cat_feature_data=ec[valid], num_feature_names=names, cat_feature_names=cats), label=early[valid])
    params = profile["training"]["params"]
    model = CatBoostClassifier(**params)
    started = time.monotonic()
    with contract.gpu_lock(root / "frozen-contract.yaml", check_device=True):
        print(f"FIT {arm} {horizon}s rows={known.sum()} inputs={len(names) + len(cats)}", flush=True)
        model.fit(train_pool, eval_set=early_pool, early_stopping_rounds=profile["training"]["early_stopping_rounds"], use_best_model=True)
    if list(model.classes_) != [0, 1, 2]:
        raise ContractError("Fitted model changed the native endpoint codebook")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.cbm")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    write_yaml(
        fit_path,
        {
            "contract_digest": digest(frozen),
            "model_sha256": file_hash(model_path),
            "arm": arm,
            "horizon": horizon,
            "numeric_features": names,
            "nominal_features": cats,
            "train_comparison": train_signature,
            "es_comparison": es_signature,
            "trees": model.tree_count_,
            "seconds": time.monotonic() - started,
            "weights": "equal_symbol_day_recency",
            "rows": int(known.sum()),
        },
    )
    return model


def score(profile, arm, horizon, role, model):
    rows, x, cat, names, cats = load_role(profile, role, arm)
    scores = predict(model, x, cat, names, cats)
    root = output(profile)
    np.save(root / "fits" / f"{arm}-{horizon}-{role}-scores.npy", scores)
    result = evaluation.evaluate(rows, scores, horizon)
    held = rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).to_numpy()
    groups = {"seen": ~held, "label_held_out": held}
    result["symbol_groups"] = {group: evaluation.evaluate(rows.loc[mask].reset_index(drop=True), scores[mask], horizon) for group, mask in groups.items() if mask.any()}
    days = evaluation.daily(rows, scores, horizon)
    result["comparison_signature"] = signature(profile, rows, horizon, role)
    days.to_csv(root / "fits" / f"{arm}-{horizon}-{role}-daily.csv", index=False)
    write_yaml(root / "fits" / f"{arm}-{horizon}-{role}-metrics.yaml", result)
    print(f"SCORE {arm} {horizon}s {role}: jointAP={result['mean_ap']:.6f} bigAP={result['big_ap']:.6f} dirAUC={result['conditional_direction_auc']:.6f}", flush=True)
    return result, days


def preflight(profile):
    root = output(profile)
    if (root / "frozen-contract.yaml").exists():
        raise ContractError("Frozen V3 preflight evidence cannot be overwritten")
    (root / "preflight.yaml").unlink(missing_ok=True)
    parent = parent_contract(profile)
    h10_data.native_validation(profile)
    previous = read_yaml(Path(parent["profile"]["paths"]["output"]) / "preflight.yaml")
    control = {entry["role"]: entry["comparison"] for entry in previous["entries"] if entry["arm"] == "baseline"}
    entries = []
    # One baseline load per role; no parallel fits or duplicate persisted matrices.
    for role in contract.ROLES:
        baseline = load_role(profile, role, "baseline")
        reference = signature(profile, baseline[0], 300, role)
        assert_same_cohort(control[role], reference)
        for arm in ARMS:
            rows, x, cat, names, cats = load_role(profile, role, arm, baseline=baseline)
            comparison = signature(profile, rows, 300, role)
            contract.assert_same_comparison(reference, comparison)
            if not len(rows) or len(set(names + cats)) != len(names) + len(cats) or x.shape != (len(rows), len(names)) or cat.shape != (len(rows), len(cats)):
                raise ContractError("V3 preflight found invalid predictor schema or common population")
            expected_names = baseline[3] + (h10_data.CANONICAL[:-1] if arm != "baseline" else [])
            expected_cats = baseline[4] + (h10_data.CANONICAL[-1:] if arm != "baseline" else [])
            if names != expected_names or cats != expected_cats:
                raise ContractError("V3 candidate must retain the baseline schema and add exactly seven numeric fields and one phase category")
            Pool(FeaturesData(num_feature_data=x[:8], cat_feature_data=cat[:8], num_feature_names=names, cat_feature_names=cats), label=np.zeros(min(8, len(rows)), dtype=int))
            entries.append({"arm": arm, "role": role, "comparison": comparison, "rows": len(rows), "numeric_columns": len(names), "nominal_columns": len(cats)})
            del rows, x, cat
        del baseline
        gc.collect()
    receipt = {
        "passed": True,
        "catboost_fits": 0,
        "all_arms_identical_keys_roles_labels": True,
        "inherited_judge_identical": True,
        "inherited_native_cohort_identical": True,
        "entries": entries,
    }
    write_yaml(root / "preflight.yaml", receipt)
    return receipt


def source_files(profile, parent, native):
    experiment = Path(profile["_profile_path"]).parent
    return (
        [Path(record["path"]) for record in parent["sources"] + native["sources"]]
        + sorted(Path(__file__).parent.glob("*.py"))
        + [
            experiment / "V3_PLAN.md",
            experiment / "H10_NOTES.md",
            experiment / "H10_NATIVE_SCREEN.md",
            experiment / "H10_COMPONENT_GATE_V2.md",
            Path(__file__).parents[2] / "tests" / "test_representation_v3.py",
        ]
    )


def freeze(profile):
    root = output(profile)
    parent = parent_contract(profile)
    native = h10_data.native_validation(profile)
    receipt = read_yaml(root / "preflight.yaml")
    required = ("passed", "all_arms_identical_keys_roles_labels", "inherited_judge_identical", "inherited_native_cohort_identical")
    if not all(receipt.get(key) is True for key in required) or receipt.get("catboost_fits") != 0:
        raise ContractError("V3 requires its completed no-fit inherited-judge/cohort/schema preflight")
    prepared = read_yaml(root / "shared" / "h10-preparation.yaml")
    if (
        prepared.get("passed") is not True
        or prepared.get("catboost_fits") != 0
        or prepared.get("labels_read") is not False
        or prepared.get("all_original_cells_mid_clock_phase_validated") is not True
    ):
        raise ContractError("V3 native feature preparation did not preserve its predictor-only contract")
    expected_caches = {str(root / "shared" / f"{role}-{family}.parquet") for role in ("train", "tune", "forward") for family in ("h10_ordered", "h10_reset")}
    if {record["path"] for record in prepared.get("caches", [])} != expected_caches or len(prepared["caches"]) != len(expected_caches):
        raise ContractError("V3 preparation must bind exactly its six aligned native role/variant caches")
    for record in prepared["caches"]:
        if record.get("numeric") != h10_data.CANONICAL[:-1] or record.get("categorical") != h10_data.CANONICAL[-1:] or file_hash(Path(record["path"])) != record.get("sha256"):
            raise ContractError("Aligned H10 cache changed its values or predictor schema since preparation")
    for path, expected in prepared["native_files"].items():
        if file_hash(Path(path)) != expected:
            raise ContractError("Native H10 parquet changed since its exact observation join")
    inputs = {record["path"]: Path(record["path"]) for record in parent["inputs"] + native["inputs"]}
    parent_path = Path(profile["inheritance"]["frozen_contract"])
    for path in (
        parent_path,
        parent_path.with_name(parent_path.name + ".identity"),
        Path(parent["profile_source"]["path"]),
        root / "preflight.yaml",
        Path(profile["h10"]["preparation_receipt"]),
        Path(profile["h10"]["validation_receipt"]),
        Path(profile["h10"]["producer_lineage_receipt"]),
        Path(profile["h10"]["prefix_validation_receipt"]),
        Path(profile["h10"]["temporal_method_contract"]),
        Path(profile["h10"]["temporal_method_contract"] + ".identity"),
        Path(profile["h10"]["temporal_validation_receipt"]),
        Path(profile["h10"]["binary"]),
    ):
        inputs[str(path)] = path
    for path in [*map(Path, prepared["native_files"]), *(root / "shared").iterdir(), *Path(profile["h10"]["native_root"]).glob("*/config.yaml")]:
        if path.is_file():
            inputs[str(path)] = path
    versions = {m.__name__: m.__version__ for m in (catboost, np, pd, pyarrow, scipy)}
    write_yaml(root / "environment.yaml", versions)
    inputs[str(root / "environment.yaml")] = root / "environment.yaml"
    return contract.freeze(profile, root / "frozen-contract.yaml", source_files(profile, parent, native), inputs)


def run(profile):
    root = output(profile)
    for name in ("completed.yaml", "nomination.yaml", "development-decisions.yaml", "ordered-versus-reset.yaml"):
        (root / name).unlink(missing_ok=True)
    frozen = contract.verify(root / "frozen-contract.yaml")
    if contract.scientific_profile(profile) != frozen["profile"]:
        raise ContractError("V3 in-memory profile differs from its immutable scientific freeze")
    parent_contract(profile)
    versions = {m.__name__: m.__version__ for m in (catboost, np, pd, pyarrow, scipy)}
    if versions != read_yaml(root / "environment.yaml"):
        raise ContractError("V3 scientific package environment changed")
    calibrated, models = {}, {}
    for arm in ARMS:
        model = fit_one(profile, arm, 300)
        calibrated[arm], _ = score(profile, arm, 300, "calibration", model)
        models[arm] = model
        contract.assert_same_comparison(calibrated["baseline"]["comparison_signature"], calibrated[arm]["comparison_signature"])
        gc.collect()
    decisions = {arm: evaluation.decision(profile, calibrated["baseline"], calibrated[arm]) for arm in ARMS if arm != "baseline"}
    qualifying = [arm for arm in decisions if decisions[arm]["supported"]]
    nominated = max(qualifying, key=lambda arm: calibrated[arm]["mean_ap"]) if qualifying else None
    write_yaml(
        root / "nomination.yaml",
        {"baseline": "baseline", "challenger": nominated, "qualifying": qualifying, "decisions": decisions, "selection_role": "calibration", "held_symbol_labels_used": False},
    )
    developed, daily = {}, {}
    for arm in ARMS:
        developed[arm], daily[arm] = score(profile, arm, 300, "development", models[arm])
        contract.assert_same_comparison(developed["baseline"]["comparison_signature"], developed[arm]["comparison_signature"])
    final = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        final[arm] = evaluation.decision(profile, developed["baseline"], developed[arm], daily["baseline"], daily[arm], stage="development")
        final[arm]["checks"]["nominated_before_development_comparison"] = arm == nominated
        for group in ("seen", "label_held_out"):
            if any(group not in report.get("symbol_groups", {}) for report in (developed["baseline"], developed[arm])):
                raise ContractError("V3 development lacks a frozen symbol-group guard")
            a, b = developed["baseline"]["symbol_groups"][group], developed[arm]["symbol_groups"][group]
            final[arm]["checks"][f"{group}_magnitude_generalization"] = b["big_ap"] >= a["big_ap"]
            final[arm]["checks"][f"{group}_direction_generalization"] = b["conditional_direction_auc"] >= max(
                a["conditional_direction_auc"] - 0.01, profile["acceptance"]["minimum_group_direction_auc"]
            )
        final[arm]["supported"] = all(final[arm]["checks"].values())
    write_yaml(root / "development-decisions.yaml", final)
    write_yaml(
        root / "ordered-versus-reset.yaml",
        {
            "calibration": evaluation.decision(profile, calibrated["h10_reset"], calibrated["h10_ordered"]),
            "development": evaluation.decision(profile, developed["h10_reset"], developed["h10_ordered"], daily["h10_reset"], daily["h10_ordered"], stage="development"),
            "qualification": "Predeclared same-judge chronology ablation; no extra nomination or pristine OOS claim.",
        },
    )
    write_yaml(
        root / "completed.yaml",
        {
            "primary_completed": True,
            "primary_fits": 3,
            "secondary_fits": 0,
            "pristine_oos": False,
            "sealed_scored": False,
            "nominated": nominated,
            "supported": bool(nominated and final[nominated]["supported"]),
            "contract": digest(frozen),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("stage", choices=("features", "preflight", "freeze", "verify", "run"))
    args = parser.parse_args()
    profile = contract.load_profile(args.profile)
    if args.stage == "verify":
        contract.verify(output(profile) / "frozen-contract.yaml")
    else:
        {"features": prepare_features, "preflight": preflight, "freeze": freeze, "run": run}[args.stage](profile)


if __name__ == "__main__":
    main()
