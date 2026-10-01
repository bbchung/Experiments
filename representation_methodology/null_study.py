"""A frozen, one-new-fit study of identical-representation GPU variability.

Existing V1/V2 baseline artifacts are exposed inputs. A prospective third fit
is scheduled only after its study contract is frozen. No feature candidates or
hyperparameter choices are compared or selected by this study.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from ..representation_research import contract, evaluation
from ..representation_research import runner as inherited
from ..representation_research_v2.runner import COHORT_FIELDS, JUDGE_FIELDS
from . import judgement


def validate_study(profile, parent):
    for field in JUDGE_FIELDS:
        if profile[field] != parent["profile"][field]:
            raise ContractError(f"Null repeat changes inherited {field}")
    if profile["arms"] != {"baseline": {"base": "current_nominal", "families": []}}:
        raise ContractError("The null study may fit only one fresh full baseline")
    study = profile["methodology_study"]
    if study["prospective_fits"] != 1 or study["existing_repeats"] != 2 or study["replicate_kind"] != "same_seed_GPU_nondeterminism":
        raise ContractError("Method study requires two exposed same-seed inputs and one prospective fit")
    if study["claims"] != ["direction", "magnitude", "both"] or len(study["existing_fit_folders"]) != 2:
        raise ContractError("Null pairs require all three preregistered claims and two exposed controls")
    if profile["compute"]["primary_fits"] != 1 or profile["compute"]["secondary_fits_maximum"] != 0:
        raise ContractError("The null-study compute budget is exactly one primary fit and no secondary fits")
    if study["noninferiority_margins"] != {"direction": 0, "magnitude": 0}:
        raise ContractError("The null-study noninferiority margins are exactly zero")
    if study["stability_gate"] != "no_supported_development_in_six_directed_identical_representation_pairs":
        raise ContractError("The null-study gate rejects every supported development null comparison")


def validate_existing(profile):
    reference = None
    for folder, manifest in zip(profile["methodology_study"]["existing_fit_folders"], profile["methodology_study"]["existing_contracts"], strict=True):
        document = contract.verify(Path(manifest))
        for field in JUDGE_FIELDS:
            if document["profile"][field] != profile[field]:
                raise ContractError(f"Exposed baseline repeat changes {field}")
        folder = Path(folder)
        receipt = read_yaml(folder / "baseline-300.yaml")
        if receipt["contract_digest"] != digest(document) or receipt["arm"] != "baseline" or receipt["horizon"] != 300:
            raise ContractError("Exposed null repeat does not belong to its frozen baseline-300 contract")
        if receipt["model_sha256"] != file_hash(folder / "baseline-300.cbm"):
            raise ContractError("Exposed null repeat model changed from its fit receipt")
        validate_model(folder / "baseline-300.cbm", receipt, profile)
        if reference is not None:
            check_receipts(reference, receipt)
        reference = receipt
    return reference


def validate_model(path, receipt, profile):
    model = inherited.CatBoostClassifier().load_model(str(path))
    features = receipt["numeric_features"] + receipt["nominal_features"]
    if list(model.classes_) != [0, 1, 2] or model.feature_names_ != features or model.get_cat_feature_indices() != list(range(len(receipt["numeric_features"]), len(features))):
        raise ContractError("Exposed repeat changed native classes, feature order or nominal schema")
    params = model.get_all_params()
    for name in ("loss_function", "iterations", "depth", "random_seed", "border_count", "nan_mode", "one_hot_max_size", "task_type"):
        if params[name] != profile["training"]["params"][name]:
            raise ContractError(f"Exposed repeat changed model training parameter: {name}")


def check_receipts(first, second):
    for name in ("numeric_features", "nominal_features", "weights", "rows", "horizon"):
        if first[name] != second[name]:
            raise ContractError(f"Null repeat changes its native representation or procedure: {name}")
    for name in ("train_comparison", "es_comparison"):
        check_cohort(first[name], second[name])


def freeze(profile):
    root = inherited.output(profile)
    parent_path = Path(profile["methodology_study"]["parent_contract"])
    parent = contract.verify(parent_path)
    validate_study(profile, parent)
    validate_existing(profile)
    inputs = {item["path"]: Path(item["path"]) for item in parent["inputs"]}
    for path in (parent_path, parent_path.with_name(parent_path.name + ".identity"), Path(parent["profile_source"]["path"])):
        inputs[str(path)] = path
    for manifest in profile["methodology_study"]["existing_contracts"]:
        manifest = Path(manifest)
        document = contract.verify(manifest)
        for path in (manifest, manifest.with_name(manifest.name + ".identity"), Path(document["profile_source"]["path"])):
            inputs[str(path)] = path
    sources = [Path(item["path"]) for item in parent["sources"]] + sorted(Path(__file__).parent.glob("*.py"))
    sources.append(Path(profile["_profile_path"]).parent / "METHOD_STUDY_PLAN.md")
    sources += [Path(__file__).parents[2] / "tests" / name for name in ("test_representation_methodology.py", "test_representation_null_study.py")]
    for folder in profile["methodology_study"]["existing_fit_folders"]:
        folder = Path(folder)
        for suffix in (
            "yaml",
            "cbm",
            "calibration-metrics.yaml",
            "calibration-daily.csv",
            "calibration-scores.npy",
            "development-metrics.yaml",
            "development-daily.csv",
            "development-scores.npy",
        ):
            path = folder / f"baseline-300.{suffix}" if suffix in ("yaml", "cbm") else folder / f"baseline-300-{suffix}"
            inputs[str(path)] = path
    write_yaml(root / "environment.yaml", {m.__name__: m.__version__ for m in (inherited.catboost, np, pd, inherited.pyarrow, inherited.scipy)})
    inputs[str(root / "environment.yaml")] = root / "environment.yaml"
    return contract.freeze(profile, root / "frozen-contract.yaml", sources, inputs)


def check_cohort(first, second):
    if any(first[key] != second[key] for key in COHORT_FIELDS):
        raise ContractError("Identical-representation repeat changed cohort or native labels")


def run(profile):
    root = inherited.output(profile)
    for name in ("completed.yaml", "null-pairs.yaml"):
        (root / name).unlink(missing_ok=True)
    frozen = contract.verify(root / "frozen-contract.yaml")
    if contract.scientific_profile(profile) != frozen["profile"]:
        raise ContractError("Null-study in-memory profile differs from its frozen scientific procedure")
    validate_study(profile, contract.verify(Path(profile["methodology_study"]["parent_contract"])))
    versions = {m.__name__: m.__version__ for m in (inherited.catboost, np, pd, inherited.pyarrow, inherited.scipy)}
    if versions != read_yaml(root / "environment.yaml"):
        raise ContractError("Null-study scientific environment changed")
    reference_receipt = validate_existing(profile)
    model = inherited.fit_one(profile, "baseline", 300)
    check_receipts(reference_receipt, read_yaml(root / "fits" / "baseline-300.yaml"))
    validate_model(root / "fits" / "baseline-300.cbm", read_yaml(root / "fits" / "baseline-300.yaml"), profile)
    for role in ("calibration", "development"):
        inherited.score(profile, "baseline", 300, role, model)
    folders = [Path(p) for p in profile["methodology_study"]["existing_fit_folders"]] + [root / "fits"]
    reports, daily = {}, {}
    for role in ("calibration", "development"):
        reports[role] = [read_yaml(folder / f"baseline-300-{role}-metrics.yaml") for folder in folders]
        daily[role] = [pd.read_csv(folder / f"baseline-300-{role}-daily.csv", dtype={"day": str}) for folder in folders]
        for report in reports[role][1:]:
            check_cohort(reports[role][0]["comparison_signature"], report["comparison_signature"])
    pairs = []
    for base, candidate in itertools.permutations(range(3), 2):
        record = {"baseline_repeat": base + 1, "candidate_repeat": candidate + 1, "legacy": {}, "typed": {}}
        for role in ("calibration", "development"):
            kwargs = {} if role == "calibration" else {"baseline_daily": daily[role][base], "candidate_daily": daily[role][candidate], "stage": "development"}
            record["legacy"][role] = evaluation.decision(profile, reports[role][base], reports[role][candidate], **kwargs)
            record["typed"][role] = {
                claim: judgement.claim_decision(profile, reports[role][base], reports[role][candidate], claim=claim, **kwargs) for claim in ("direction", "magnitude", "both")
            }
        record["legacy_complete_admission"] = all(record["legacy"][role]["supported"] for role in ("calibration", "development"))
        record["typed_complete_admission"] = {
            claim: all(record["typed"][role][claim]["supported"] for role in ("calibration", "development")) for claim in ("direction", "magnitude", "both")
        }
        pairs.append(record)
    ranges = {
        role: {
            name: {"values": [float(r[name]) for r in reports[role]], "range": float(np.ptp([r[name] for r in reports[role]]))}
            for name in ("mean_ap", "big_ap", "conditional_direction_auc")
        }
        for role in ("calibration", "development")
    }
    old_admitted = sum(p["legacy_complete_admission"] for p in pairs)
    typed_admitted = sum(any(p["typed_complete_admission"].values()) for p in pairs)
    old_development = sum(p["legacy"]["development"]["supported"] for p in pairs)
    typed_development = sum(any(result["supported"] for result in p["typed"]["development"].values()) for p in pairs)
    write_yaml(root / "null-pairs.yaml", pairs)
    metric_means = {
        role: {
            name: [float(np.mean([report[name] for index, report in enumerate(reports[role]) if index != excluded])) for excluded in range(3)]
            for name in ("mean_ap", "big_ap", "conditional_direction_auc")
        }
        for role in ("calibration", "development")
    }
    write_yaml(
        root / "completed.yaml",
        {
            "contract_identity": frozen["identity"],
            "new_fits": 1,
            "same_seed_repeats": 3,
            "ranges": ranges,
            "leave_one_out_scalar_means_not_ensemble_predictions": metric_means,
            "legacy_null_admissions": old_admitted,
            "typed_null_admissions": typed_admitted,
            "legacy_supported_development_null_pairs": old_development,
            "typed_supported_development_null_pairs": typed_development,
            "bounded_no_admission_check_passed": typed_development == 0,
            "bounded_stability_regression_observed": typed_development > old_development,
            "qualifications": [
                "Three same-seed repeats cannot estimate seed variance or certify false-admission rates.",
                "All original score dates were already exposed; no predictive FE admission or pristine OOS proof is made.",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    profile = contract.load_profile(args.profile)
    {"freeze": freeze, "run": run}[args.stage](profile)


if __name__ == "__main__":
    main()
