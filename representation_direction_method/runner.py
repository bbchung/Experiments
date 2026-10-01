"""Frozen baseline-only direction formulation study; root schedules all fits."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, FeaturesData, Pool

from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from ..representation_methodology import judgement
from ..representation_methodology.null_study import check_cohort, check_receipts, validate_model
from ..representation_research import contract, data
from ..representation_research import runner as inherited
from ..representation_research_v2.runner import JUDGE_FIELDS
from . import evaluation

STUDY = {
    "kind": "fixed_magnitude_conditional_direction_v1",
    "claim": "direction",
    "maximum_new_fits": 3,
    "replicate_kind": "same_seed_GPU_nondeterminism",
    "first_calibration_early_reject": True,
    "all_calibration_before_development": True,
    "train_weights": "original_full_train_equal_symbol_day_recency_then_big_subset_no_renormalization",
    "es_use": "big_events_early_stopping_only_no_calibration",
    "score_contract": "direct_fixed_b_q_v1",
    "ensemble": "three_per_row_q_means_plus_descriptive_leave_one_out_score_means",
    "noninferiority_margin": 0,
    "adoption": "none_method_study_only",
}
COHORT_ROLES = ("train", "es", "calibration", "development")
ACCEPTANCE_DESCRIPTION = {
    "nomination": "direction_claim_joint_gain_fixed_magnitude_absolute_power",
    "development": "typed_gates_paired_joint_direction_gain_exact_magnitude_zero_delta",
    "final_oos": "not_scored_or_claimed_by_method_study",
    "magnitude_noninferiority_margin": 0,
    "direction_noninferiority_margin": 0,
}


def output(profile):
    return inherited.output(profile)


def load_profile(path):
    profile = contract.load_profile(path)
    validate_study(profile)
    return profile


def validate_study(profile):
    if profile.get("direction_method") != STUDY or profile["training"]["params"]["loss_function"] != "Logloss":
        raise ContractError("Direction study formulation, claim, budget and loss must be predeclared")
    if (
        profile["arms"] != {"baseline": {"base": "current_nominal", "families": []}}
        or profile["compute"]["primary_fits"] != 3
        or profile["compute"]["secondary_fits_maximum"] != 0
        or not profile["compute"]["no_hpo"]
    ):
        raise ContractError("Direction study permits one unchanged representation and at most three conditional fits")
    controls = profile.get("direction_controls")
    if not isinstance(controls, list) or len(controls) != 3 or any(set(control) != {"contract", "identity", "fits"} for control in controls):
        raise ContractError("Direction study requires exactly three fixed immutable MultiClass controls")
    if len({control["contract"] for control in controls}) != 3 or profile.get("fixed_magnitude_control") != 0:
        raise ContractError("Magnitude anchor is fixed control zero; no outcome-based control selection")


def metadata(profile, *, inspect_models=False):
    validate_study(profile)
    documents, receipts = [], []
    reference = None
    for control in profile["direction_controls"]:
        document = contract.verify(Path(control["contract"]))
        if document["identity"] != control["identity"]:
            raise ContractError("Direction control changed its pinned immutable identity")
        for field in JUDGE_FIELDS:
            expected = copy.deepcopy(document["profile"][field])
            if field == "training":
                expected["params"]["loss_function"] = "Logloss"
            elif field == "acceptance":
                expected.update(ACCEPTANCE_DESCRIPTION)
            if expected != profile[field]:
                raise ContractError(f"Direction formulation changes more than its declared conditional loss: {field}")
        if document["profile"]["training"]["params"]["loss_function"] != "MultiClass":
            raise ContractError("Direction controls must be the original MultiClass baselines")
        folder = Path(control["fits"])
        receipt = read_yaml(folder / "baseline-300.yaml")
        if (
            receipt.get("arm") != "baseline"
            or receipt.get("horizon") != 300
            or receipt.get("contract_digest") != digest(document)
            or receipt.get("model_sha256") != file_hash(folder / "baseline-300.cbm")
        ):
            raise ContractError("Direction control is not its frozen baseline-300 model")
        if reference is not None:
            check_receipts(reference, receipt)
        if inspect_models:
            mc_profile = copy.deepcopy(profile)
            mc_profile["training"]["params"]["loss_function"] = "MultiClass"
            validate_model(folder / "baseline-300.cbm", receipt, mc_profile)
        reference = receipt
        documents.append(document)
        receipts.append(receipt)
    return documents, receipts


def dependency_paths(profile, documents):
    package = Path(__file__).parent
    experiment = Path(profile["_profile_path"]).parent
    sources = {Path(node["path"]) for document in documents for node in document["sources"]}
    sources.update(package.glob("*.py"))
    sources.update(
        [
            Path(judgement.__file__),
            package.parent / "representation_methodology" / "null_study.py",
            experiment / "DIRECTION_METHOD_PLAN.md",
            experiment / "DIRECTION_METHOD_OUTLINE.md",
            package.parents[1] / "tests" / "test_direction_method.py",
        ]
    )
    inputs = {Path(node["path"]) for document in documents for node in document["inputs"]}
    for control, document in zip(profile["direction_controls"], documents, strict=True):
        manifest = Path(control["contract"])
        inputs.update([manifest, Path(str(manifest) + ".identity"), Path(document["profile_source"]["path"])])
        folder = Path(control["fits"])
        inputs.update([folder / "baseline-300.yaml", folder / "baseline-300.cbm"])
        for role in ("calibration", "development"):
            inputs.update(folder / f"baseline-300-{role}-{suffix}" for suffix in ("scores.npy", "metrics.yaml"))
    return sorted(sources), sorted(inputs)


def preflight(profile):
    root = output(profile)
    if (root / "frozen-contract.yaml").exists():
        raise ContractError("Frozen direction method cannot overwrite its preflight")
    path = root / "preflight.yaml"
    path.unlink(missing_ok=True)
    documents, receipts = metadata(profile, inspect_models=True)
    sources, inputs = dependency_paths(profile, documents)
    if any(not path.is_file() for path in sources + inputs):
        raise ContractError("Direction study lacks an immutable model/source/score binding")
    result = {
        "passed": True,
        "new_fits": 0,
        "datasets_or_scores_loaded": False,
        "metadata_digest": digest(receipts),
        "control_identities": [document["identity"] for document in documents],
    }
    write_yaml(path, result)
    return result


def freeze(profile):
    root = output(profile)
    documents, receipts = metadata(profile)
    checked = read_yaml(root / "preflight.yaml")
    if checked != {
        "passed": True,
        "new_fits": 0,
        "datasets_or_scores_loaded": False,
        "metadata_digest": digest(receipts),
        "control_identities": [document["identity"] for document in documents],
    }:
        raise ContractError("Direction study requires its exact no-fit metadata preflight")
    sources, inputs = dependency_paths(profile, documents)
    write_yaml(root / "environment.yaml", versions())
    return contract.freeze(profile, root / "frozen-contract.yaml", sources, inputs + [root / "preflight.yaml", root / "environment.yaml"])


def verify(profile):
    root = output(profile)
    frozen = contract.verify(root / "frozen-contract.yaml")
    if frozen["profile"] != contract.scientific_profile(profile):
        raise ContractError("Direction in-memory profile changed its frozen methodology")
    documents, receipts = metadata(profile)
    sources, inputs = dependency_paths(profile, documents)
    inputs += [root / "preflight.yaml", root / "environment.yaml"]
    for scope, paths in (("sources", sources), ("inputs", inputs)):
        bound = {node["path"]: node["sha256"] for node in frozen[scope]}
        if any(bound.get(str(path.resolve())) != file_hash(path) for path in paths):
            raise ContractError("Direction freeze omitted mandatory methodology/control source/input closure")
    if read_yaml(root / "environment.yaml") != versions():
        raise ContractError("Direction method package environment changed")
    checked = read_yaml(root / "preflight.yaml")
    if checked != {
        "passed": True,
        "new_fits": 0,
        "datasets_or_scores_loaded": False,
        "metadata_digest": digest(receipts),
        "control_identities": [document["identity"] for document in documents],
    }:
        raise ContractError("Direction method preflight changed its metadata validation")
    return frozen


def versions():
    return {module.__name__: module.__version__ for module in (inherited.catboost, np, pd, inherited.pyarrow, inherited.scipy)}


def load_role(profile, role, reference):
    if role not in COHORT_ROLES:
        raise ContractError("Direction study role is outside its frozen chronology")
    rows, x, cat, names, cats = inherited.load_role(profile, role, "baseline")
    if role != "development" and rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).any():
        raise ContractError("Held-symbol endpoint labels cannot enter fitting, stopping or nomination")
    if (
        names != reference["numeric_features"]
        or cats != reference["nominal_features"]
        or any(name.startswith(("mid_return_ticks[", "mid_endpoint.", "sticky_return_ticks[")) for name in names + cats)
    ):
        raise ContractError("Direction inputs changed original feature order or contain endpoint-label predictors")
    comparison = inherited.signature(profile, rows, 300, role)
    if role in ("train", "es"):
        check_cohort(reference[role + "_comparison"], comparison)
    return rows, x, cat, names, cats


def original_weights(rows, known, half_life):
    groups = rows.loc[known, ["day", "symbol"]]
    counts = groups.groupby(["day", "symbol"])["day"].transform("size").to_numpy()
    age = (pd.to_datetime(groups.day.max()) - pd.to_datetime(groups.day)).dt.days.to_numpy()
    weights = np.exp2(-age / half_life) / counts
    return weights / weights.mean()


def training_pools(profile, reference):
    pieces, bindings = [], {}
    for role in ("train", "es"):
        rows, x, cat, names, cats = load_role(profile, role, reference)
        labels, known = data.native_classes(rows, 300)
        selected = known & (labels != 0)
        targets = (labels[selected] == 1).astype(np.int64)
        if set(targets) != {0, 1}:
            raise ContractError("Conditional direction fitting/stopping requires both native big-event sides")
        weights = original_weights(rows, known, profile["training"]["recency_half_life_days"])[labels[known] != 0] if role == "train" else None
        pieces.append(
            Pool(FeaturesData(num_feature_data=x[selected], cat_feature_data=cat[selected], num_feature_names=names, cat_feature_names=cats), label=targets, weight=weights)
        )
        bindings[role] = {
            "comparison": inherited.signature(profile, rows, 300, role),
            "big_mask_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
            "event_rows": int(selected.sum()),
        }
        if weights is not None:
            bindings[role]["original_subset_weights_sha256"] = hashlib.sha256(weights.tobytes()).hexdigest()
        del rows, x, cat
        gc.collect()
    bindings["numeric_features"], bindings["nominal_features"] = names, cats
    return pieces, bindings


def validate_direction_model(model, profile, bindings):
    names, cats = bindings["numeric_features"], bindings["nominal_features"]
    if list(model.classes_) != [0, 1] or model.feature_names_ != names + cats or model.get_cat_feature_indices() != list(range(len(names), len(names + cats))):
        raise ContractError("Conditional direction model changed its classes or original feature schema")
    actual, expected = model.get_all_params(), profile["training"]["params"]
    for key in ("loss_function", "iterations", "depth", "random_seed", "border_count", "nan_mode", "one_hot_max_size", "task_type", "learning_rate", "l2_leaf_reg"):
        same = np.float32(actual[key]) == np.float32(expected[key]) if key in ("learning_rate", "l2_leaf_reg") else actual[key] == expected[key]
        if not same:
            raise ContractError(f"Conditional direction model changed frozen parameter {key}")


def fit_one(profile, repeat, pools, bindings, frozen):
    if repeat not in (1, 2, 3):
        raise ContractError("Direction study exceeds its three-fit budget")
    root = output(profile)
    model_path, receipt_path = root / "fits" / f"q{repeat}.cbm", root / "fits" / f"q{repeat}.yaml"
    if receipt_path.exists():
        receipt = read_yaml(receipt_path)
        ledger = read_yaml(root / "fit-attempts.yaml")
        if (
            receipt.get("repeat") != repeat
            or receipt.get("contract_digest") != digest(frozen)
            or receipt.get("bindings") != bindings
            or receipt.get("model_sha256") != file_hash(model_path)
            or ledger.get("contract_digest") != digest(frozen)
            or repeat not in ledger.get("attempted", [])
            or len(ledger["attempted"]) > 3
        ):
            raise ContractError("Resumed direction model differs from frozen inputs/weights/procedure")
        model = CatBoostClassifier().load_model(str(model_path))
        validate_direction_model(model, profile, bindings)
        return model
    model = CatBoostClassifier(**profile["training"]["params"])
    ledger_path = root / "fit-attempts.yaml"
    with contract.gpu_lock(root / "frozen-contract.yaml", check_device=True):
        if repeat > 1:
            gate = read_yaml(root / "first-calibration-gate.yaml")
            if gate.get("passed") is not True or gate.get("contract_digest") != digest(frozen):
                raise ContractError("Direction fit cannot continue after its first calibration failed")
        ledger = read_yaml(ledger_path) if ledger_path.exists() else {"contract_digest": digest(frozen), "attempted": []}
        if ledger.get("contract_digest") != digest(frozen) or repeat in ledger["attempted"] or len(ledger["attempted"]) != repeat - 1 or model_path.exists():
            raise ContractError("Incomplete/orphaned direction attempt cannot consume an extra GPU fit")
        ledger["attempted"].append(repeat)
        write_yaml(ledger_path, ledger)
        model.fit(pools[0], eval_set=pools[1], early_stopping_rounds=profile["training"]["early_stopping_rounds"], use_best_model=True)
    validate_direction_model(model, profile, bindings)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.cbm")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    write_yaml(receipt_path, {"contract_digest": digest(frozen), "bindings": bindings, "repeat": repeat, "model_sha256": file_hash(model_path), "trees": model.tree_count_})
    return model


def control_q(profile, rows, role):
    if role not in ("calibration", "development"):
        raise ContractError("Existing score read requires an explicit evaluation role")
    result, fixed_b = [], None
    current = inherited.signature(profile, rows, 300, role)
    for index, control in enumerate(profile["direction_controls"]):
        folder = Path(control["fits"])
        # The source metrics are used only to bind rows/labels, never select a
        # control, threshold or hypothesis by its original scalar results.
        source = read_yaml(folder / f"baseline-300-{role}-metrics.yaml")
        check_cohort(source["comparison_signature"], current)
        scores = np.load(folder / f"baseline-300-{role}-scores.npy", mmap_mode="r")
        if len(scores) != len(rows):
            raise ContractError("Original MultiClass score row count differs from the fixed cohort")
        b, q = evaluation.original_heads(scores)
        if index == profile["fixed_magnitude_control"]:
            fixed_b = b.copy()
        result.append(q)
    root = output(profile)
    saved = root / "scores" / f"fixed-b-{role}.npy"
    saved.parent.mkdir(parents=True, exist_ok=True)
    if saved.exists():
        evaluation.assert_fixed_b(fixed_b, np.load(saved))
    else:
        np.save(saved, fixed_b)
    saved_bits = root / "scores" / f"fixed-b-{role}-bits.npy"
    if saved_bits.exists():
        bits = np.load(saved_bits)
        if bits.dtype != np.uint64 or not np.array_equal(bits, fixed_b.view(np.uint64)):
            raise ContractError("Saved magnitude uint64 bits differ from their immutable original source")
    else:
        np.save(saved_bits, fixed_b.view(np.uint64))
    write_yaml(
        root / "scores" / f"fixed-b-{role}.yaml",
        {
            "comparison": current,
            "control_identity": profile["direction_controls"][0]["identity"],
            "bits_sha256": hashlib.sha256(fixed_b.tobytes()).hexdigest(),
            "artifact_sha256": file_hash(saved),
            "uint64_artifact_sha256": file_hash(saved_bits),
        },
    )
    return fixed_b, result


def predict_q(model, material):
    _, x, cat, names, cats = material
    pieces = []
    for start in range(0, len(x), 16384):
        pool = FeaturesData(num_feature_data=x[start : start + 16384], cat_feature_data=cat[start : start + 16384], num_feature_names=names, cat_feature_names=cats)
        pieces.append(model.predict_proba(pool, thread_count=8)[:, 1])
    return evaluation.head(np.concatenate(pieces), len(x))


def scored(profile, rows, b, q, role, name):
    report, daily = evaluation.report(profile, rows, b, q, role)
    root = output(profile) / "scores"
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{name}-{role}-q.npy", q)
    np.save(root / f"{name}-{role}-joint.npy", np.column_stack([b * q, b * (1 - q)]))
    write_yaml(root / f"{name}-{role}-metrics.yaml", report)
    daily.to_csv(root / f"{name}-{role}-daily.csv", index=False)
    return report, daily


def run(profile):
    root = output(profile)
    for name in ("completed.yaml", "nomination.yaml", "null-pairs.yaml", "development-decisions.yaml", "stability.yaml", "first-calibration-gate.yaml"):
        (root / name).unlink(missing_ok=True)
    frozen = verify(profile)
    _, receipts = metadata(profile)
    cal = load_role(profile, "calibration", receipts[0])
    b, mc_q = control_q(profile, cal[0], "calibration")
    control = [scored(profile, cal[0], b, q, "calibration", f"mc{index + 1}") for index, q in enumerate(mc_q)]
    pools, bindings = training_pools(profile, receipts[0])
    models, q_cal, calibrated = [], [], []
    for repeat in (1, 2, 3):
        model = fit_one(profile, repeat, pools, bindings, frozen)
        models.append(model)
        q = predict_q(model, cal)
        q_cal.append(q)
        calibrated.append(scored(profile, cal[0], b, q, "calibration", f"q{repeat}"))
        if repeat == 1:
            first = evaluation.decision(profile, control[0][0], calibrated[0][0])
            write_yaml(root / "first-calibration-gate.yaml", {"contract_digest": digest(frozen), "passed": first["supported"], "decision": first})
            if not first["supported"]:
                write_yaml(
                    root / "completed.yaml",
                    {
                        "contract_identity": frozen["identity"],
                        "supported": False,
                        "early_rejected": True,
                        "new_fits_maximum": 1,
                        "development_read": False,
                        "method_adopted": False,
                        "feature_admission": False,
                        "pristine_oos": False,
                    },
                )
                return
    mc_mean = scored(profile, cal[0], b, evaluation.mean_q(mc_q), "calibration", "mc-mean")
    q_mean = scored(profile, cal[0], b, evaluation.mean_q(q_cal), "calibration", "q-mean")
    calibration_deltas = [calibrated[index][0]["conditional_direction_auc"] - control[index][0]["conditional_direction_auc"] for index in range(3)]
    nominated = evaluation.decision(profile, mc_mean[0], q_mean[0])["supported"]
    write_yaml(root / "nomination.yaml", {"nominated": nominated, "role": "calibration", "all_three_calibration_completed": True, "conditional_repeats": 3})
    del cal, pools
    gc.collect()
    # First development material/score/label access occurs after all calibration
    # scores and the immutable nomination have been written.
    dev = load_role(profile, "development", receipts[0])
    db, dmc_q = control_q(profile, dev[0], "development")
    developed_control = [scored(profile, dev[0], db, q, "development", f"mc{index + 1}") for index, q in enumerate(dmc_q)]
    q_dev = [predict_q(model, dev) for model in models]
    developed = [scored(profile, dev[0], db, q, "development", f"q{index + 1}") for index, q in enumerate(q_dev)]
    dmc_mean = scored(profile, dev[0], db, evaluation.mean_q(dmc_q), "development", "mc-mean")
    dq_mean = scored(profile, dev[0], db, evaluation.mean_q(q_dev), "development", "q-mean")
    final = evaluation.decision(profile, dmc_mean[0], dq_mean[0], dmc_mean[1], dq_mean[1], stage="development")
    repeat_decisions = [
        evaluation.decision(profile, developed_control[index][0], developed[index][0], developed_control[index][1], developed[index][1], stage="development") for index in range(3)
    ]
    nulls = []
    for family, reports in (("mc", developed_control), ("q", developed)):
        for first, second in itertools.permutations(range(3), 2):
            decision = evaluation.decision(profile, reports[first][0], reports[second][0], reports[first][1], reports[second][1], stage="development")
            nulls.append({"family": family, "baseline": first + 1, "candidate": second + 1, "supported": decision["supported"], "decision": decision})
    write_yaml(root / "null-pairs.yaml", nulls)
    stability = {"calibration": {"replica_direction_auc_deltas": calibration_deltas}}
    for role, rows, fixed_b, families in (("development", dev[0], db, {"mc": dmc_q, "q": q_dev}),):
        for family, vectors in families.items():
            for omitted in range(3):
                scored(profile, rows, fixed_b, evaluation.mean_q([vector for index, vector in enumerate(vectors) if index != omitted]), role, f"{family}-leave-out-{omitted + 1}")
        stability[role] = {
            "replica_direction_auc_deltas": [developed[index][0]["conditional_direction_auc"] - developed_control[index][0]["conditional_direction_auc"] for index in range(3)],
            "null_supported_pairs": sum(record["supported"] for record in nulls),
            "same_seed_repeats_not_seed_variance_or_FPR_estimate": True,
        }
    write_yaml(root / "stability.yaml", stability)
    final["checks"].update(
        nominated_before_development=nominated,
        all_three_repeat_direction_gains=all(result["supported"] for result in repeat_decisions),
        no_supported_null_pairs=not any(record["supported"] for record in nulls),
    )
    final["supported"] = all(final["checks"].values())
    write_yaml(root / "development-decisions.yaml", {"ensemble": final, "replicas": repeat_decisions})
    write_yaml(
        root / "completed.yaml",
        {
            "contract_identity": frozen["identity"],
            "supported": final["supported"],
            "early_rejected": False,
            "new_fits_maximum": 3,
            "development_read": True,
            "method_adopted": False,
            "pristine_oos": False,
            "feature_admission": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("stage", choices=("preflight", "freeze", "verify", "run"))
    args = parser.parse_args()
    profile = load_profile(args.profile)
    globals()[args.stage](profile)


if __name__ == "__main__":
    main()
