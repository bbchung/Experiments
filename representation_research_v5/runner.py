"""Three original-native arms under the unchanged frozen V3 primary300 judge."""

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
from ..representation_research_v2.runner import JUDGE_FIELDS, assert_same_cohort
from ..representation_research_v4 import data as audited
from . import data as selection

output = inherited.output
signature = inherited.signature
predict = inherited.predict
load_role = selection.load_role
ARMS = selection.ARMS


def require(condition, message):
    if not bool(condition):
        raise ContractError(message)


def parent_contract(profile):
    parent = contract.verify(Path(profile["inheritance"]["frozen_contract"]))
    require(parent["identity"] == profile["inheritance"]["identity"], "V5 frozen V3 parent identity changed")
    require(Path(profile["paths"]["parent_study"]).resolve() == Path(parent["profile"]["paths"]["parent_study"]).resolve(), "V5 original native projection changed")
    for name in JUDGE_FIELDS:
        require(digest(profile.get(name)) == digest(parent["profile"].get(name)), f"V5 changes frozen V3 judge field {name}")
    require(profile["arms"] == ARMS and list(profile["arms"]) == list(ARMS) and profile["baseline"] == "baseline", "V5 requires exactly fresh B, C15 and C18")
    budget = profile["compute"]
    require(
        type(budget["primary_fits"]) is int
        and budget["primary_fits"] == 3
        and type(budget["secondary_fits_maximum"]) is int
        and budget["secondary_fits_maximum"] == 0
        and budget["no_hpo"] is True,
        "V5 permits exactly three serial primary300 fits and no HPO/secondary fits",
    )
    require(
        profile["representations"]
        == {
            "c15": {"numeric": list(selection.C15_NUMERIC), "categorical": list(selection.C_CATEGORICAL)},
            "c18": {"numeric": list(selection.C18_NUMERIC), "categorical": list(selection.C_CATEGORICAL)},
        },
        "V5 original-native selector definitions changed",
    )
    require(profile["analysis"] == {"development_all_arms": True, "ablation": ["c15", "c18"], "ablation_has_nomination": False}, "V5 prospective diagnostics/ablation rule changed")
    require("h15" not in profile and "h10" not in profile, "V5 contains no H10/H15 representation/material prerequisite")
    return parent


def source_files(profile, parent):
    experiment = Path(profile["_profile_path"]).parent
    paths = [Path(node["path"]) for node in parent["sources"]]
    paths += [
        *Path(__file__).parent.glob("*.py"),
        Path(audited.__file__),
        Path(audited.__file__).parent / "__init__.py",
        Path(data.__file__),
        Path(__file__).parents[2] / "feature_values.py",
        experiment / "V5_PLAN.md",
        experiment / "H15_COMPACT_C_SOURCE_AUDIT.md",
        Path(__file__).parents[2] / "tests/test_representation_v5.py",
        Path(__file__).parents[2] / "tests/test_representation_v4_data.py",
    ]
    return sorted({path.resolve() for path in paths})


def record(path):
    path = Path(path).resolve()
    require(path.is_file(), f"Missing V5 bound source/input: {path}")
    return {"path": str(path), "sha256": file_hash(path)}


def binding(profile, parent):
    return {
        "profile_identity": digest(contract.scientific_profile(profile)),
        "profile_source": record(profile["_profile_path"]),
        "sources": [record(path) for path in source_files(profile, parent)],
        "inputs": parent["inputs"],
        "parent_identity": parent["identity"],
    }


def validate_preflight(profile, receipt, current_binding):
    require(
        receipt.get("schema") == "native-compact-response-preflight-v1"
        and all(
            receipt.get(key) is True
            for key in ("passed", "all_arms_identical_keys_roles_labels", "inherited_judge_identical", "inherited_native_cohort_identical", "c15_shared_bits_identical")
        ),
        "V5 preflight lacks complete typed equality proofs",
    )
    require(
        type(receipt.get("catboost_fits")) is int and receipt["catboost_fits"] == 0 and digest(receipt.get("binding")) == digest(current_binding),
        "V5 preflight/source/profile drift or fit exposure",
    )
    entries = receipt.get("entries")
    require(
        isinstance(entries, list) and [(entry.get("role"), entry.get("arm")) for entry in entries] == [(role, arm) for role in contract.ROLES for arm in ARMS],
        "V5 preflight must cover exactly three arms by four roles",
    )
    for entry in entries:
        names, cats = selection.arm_names(profile, entry["arm"])
        require(
            entry.get("numeric_features") == names and entry.get("nominal_features") == cats and type(entry.get("rows")) is int and entry["rows"] > 0,
            "V5 preflight feature-role/schema support changed",
        )
    for role in contract.ROLES:
        group = [entry for entry in entries if entry["role"] == role]
        for entry in group:
            require(entry["rows"] == group[0]["rows"], "V5 shared cohort row count changed")
            contract.assert_same_comparison(group[0]["comparison"], entry["comparison"])


def preflight(profile):
    root = output(profile)
    require(not (root / "frozen-contract.yaml").exists(), "Frozen V5 preflight cannot be overwritten")
    (root / "preflight.yaml").unlink(missing_ok=True)
    parent = parent_contract(profile)
    original_binding = binding(profile, parent)
    previous = read_yaml(Path(parent["profile"]["paths"]["output"]) / "preflight.yaml")
    control = {entry["role"]: entry["comparison"] for entry in previous["entries"] if entry["arm"] == "baseline"}
    entries = []
    positions = [selection.C18_NUMERIC.index(name) for name in selection.C15_NUMERIC]
    for role in contract.ROLES:
        baseline = load_role(profile, role, "baseline")
        reference = signature(profile, baseline[0], 300, role)
        assert_same_cohort(control[role], reference)
        c15 = None
        for arm in ARMS:
            rows, x, cat, names, cats = baseline if arm == "baseline" else load_role(profile, role, arm)
            comparison = signature(profile, rows, 300, role)
            contract.assert_same_comparison(reference, comparison)
            require(names == selection.arm_names(profile, arm)[0] and cats == selection.arm_names(profile, arm)[1], "V5 exact feature order changed")
            require(
                len(rows) > 0
                and x.dtype == np.float32
                and cat.dtype == object
                and x.shape == (len(rows), len(names))
                and cat.shape == (len(rows), len(cats))
                and len(set(names + cats)) == len(names) + len(cats),
                "V5 predictor storage/schema/cohort invalid",
            )
            if arm == "c15":
                c15 = (x.copy(), cat.copy())
            elif arm == "c18":
                require(np.array_equal(x[:, positions].view(np.uint32), c15[0].view(np.uint32)) and np.array_equal(cat, c15[1]), "V5 C18 changed shared C15 bits or categories")
            Pool(FeaturesData(num_feature_data=x[:8], cat_feature_data=cat[:8], num_feature_names=names, cat_feature_names=cats), label=np.zeros(min(8, len(rows)), dtype=int))
            entries.append({"role": role, "arm": arm, "comparison": comparison, "rows": len(rows), "numeric_features": names, "nominal_features": cats})
        del baseline, c15, rows, x, cat
        gc.collect()
    receipt = {
        "schema": "native-compact-response-preflight-v1",
        "passed": True,
        "catboost_fits": 0,
        "all_arms_identical_keys_roles_labels": True,
        "inherited_judge_identical": True,
        "inherited_native_cohort_identical": True,
        "c15_shared_bits_identical": True,
        "binding": original_binding,
        "entries": entries,
    }
    require(digest(original_binding) == digest(binding(profile, parent_contract(profile))), "V5 sources/inputs changed during preflight")
    validate_preflight(profile, receipt, original_binding)
    write_yaml(root / "preflight.yaml", receipt)
    return receipt


def freeze(profile):
    root = output(profile)
    require(not (root / "frozen-contract.yaml").exists(), "Frozen V5 cannot be overwritten")
    parent = parent_contract(profile)
    validate_preflight(profile, read_yaml(root / "preflight.yaml"), binding(profile, parent))
    inputs = {node["path"]: Path(node["path"]) for node in parent["inputs"]}
    parent_path = Path(profile["inheritance"]["frozen_contract"])
    for path in (parent_path, parent_path.with_suffix(parent_path.suffix + ".identity"), Path(parent["profile_source"]["path"]), root / "preflight.yaml"):
        inputs[str(path)] = path
    # All full original projection columns, including the three omitted from
    # current_nominal, remain immutable inputs. No C15/C18 matrix is persisted.
    original = Path(profile["paths"]["parent_study"])
    for role in ("train", "tune", "forward"):
        for name in ("projection.json", "x.npy", "rows.parquet", "nominal.parquet", "nominal-projection.yaml"):
            path = original / role / name
            inputs[str(path)] = path
    versions = {module.__name__: module.__version__ for module in (catboost, np, pd, pyarrow, scipy)}
    write_yaml(root / "environment.yaml", versions)
    inputs[str(root / "environment.yaml")] = root / "environment.yaml"
    return contract.freeze(profile, root / "frozen-contract.yaml", source_files(profile, parent), inputs)


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


def development_decision(profile, baseline, candidate, baseline_daily, candidate_daily):
    """The same V3 decision and seen/held guards for both full and ablation tests."""
    verdict = evaluation.decision(profile, baseline, candidate, baseline_daily, candidate_daily, stage="development")
    for group in ("seen", "label_held_out"):
        require(group in baseline.get("symbol_groups", {}) and group in candidate.get("symbol_groups", {}), "V5 development lacks a frozen symbol-group guard")
        a, b = baseline["symbol_groups"][group], candidate["symbol_groups"][group]
        verdict["checks"][f"{group}_magnitude_generalization"] = b["big_ap"] >= a["big_ap"]
        verdict["checks"][f"{group}_direction_generalization"] = b["conditional_direction_auc"] >= max(
            a["conditional_direction_auc"] - 0.01, profile["acceptance"]["minimum_group_direction_auc"]
        )
    verdict["supported"] = all(verdict["checks"].values())
    return verdict


def run(profile):
    root = output(profile)
    frozen = contract.verify(root / "frozen-contract.yaml")
    require(digest(contract.scientific_profile(profile)) == digest(frozen["profile"]), "V5 in-memory profile differs from its immutable freeze")
    parent_contract(profile)
    versions = {module.__name__: module.__version__ for module in (catboost, np, pd, pyarrow, scipy)}
    require(versions == read_yaml(root / "environment.yaml"), "V5 scientific package environment changed")
    for name in ("completed.yaml", "nomination.yaml", "development-decisions.yaml", "c18-versus-c15.yaml"):
        (root / name).unlink(missing_ok=True)
    calibrated, models = {}, {}
    for arm in ARMS:
        model = fit_one(profile, arm, 300)
        calibrated[arm], _ = score(profile, arm, 300, "calibration", model)
        models[arm] = model
        contract.assert_same_comparison(calibrated["baseline"]["comparison_signature"], calibrated[arm]["comparison_signature"])
        gc.collect()
    decisions = {arm: evaluation.decision(profile, calibrated["baseline"], calibrated[arm]) for arm in ARMS if arm != "baseline"}
    qualifying = [arm for arm, verdict in decisions.items() if verdict["supported"]]
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
        verdict = development_decision(profile, developed["baseline"], developed[arm], daily["baseline"], daily[arm])
        verdict["checks"]["nominated_before_development_comparison"] = arm == nominated
        verdict["supported"] = all(verdict["checks"].values())
        final[arm] = verdict
    write_yaml(root / "development-decisions.yaml", final)
    write_yaml(
        root / "c18-versus-c15.yaml",
        {
            "calibration": evaluation.decision(profile, calibrated["c15"], calibrated["c18"]),
            "development": development_decision(profile, developed["c15"], developed["c18"], daily["c15"], daily["c18"]),
            "qualification": "Predeclared same-judge conditional response ablation; no extra nomination or complete representation/OOS admission.",
        },
    )
    contract.verify(root / "frozen-contract.yaml")
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
    parser.add_argument("stage", choices=("preflight", "freeze", "verify", "run"))
    args = parser.parse_args()
    profile = contract.load_profile(args.profile)
    if args.stage == "verify":
        print(contract.verify(output(profile) / "frozen-contract.yaml")["identity"])
    else:
        print({"preflight": preflight, "freeze": freeze, "run": run}[args.stage](profile))


if __name__ == "__main__":
    main()
