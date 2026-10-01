"""Engineering-only input/model checks over one completed native day."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, FeaturesData, Pool
from study_data import HORIZONS, native_classes, numeric_domain, project, project_nominal

from AstraResearch.contracts import ArtifactRef
from AstraResearch.io import read_yaml, write_yaml
from AstraResearch.kernels.funnel import numeric_features
from AstraResearch.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/fe_origin_20260930/study-smoke"))
    args = parser.parse_args()
    store = Store(Path("runs/fe-origin-store").resolve())
    ref = ArtifactRef("native_material_day", "bc56e95608d67a79a28bad7f52762a8e11a25d5958ca94b8aef64fe70131d751")
    dataset = store.resolve(ref)
    dataset.metadata["role"] = "engineering_smoke"
    args.output.mkdir(parents=True, exist_ok=True)
    numeric, nominal = numeric_features(dataset)
    numeric = numeric_domain(dataset, numeric, args.output)
    x, rows, _ = project(dataset, numeric, args.output)
    categorical = project_nominal(dataset, list(nominal), args.output, rows)
    support = {}
    for horizon in HORIZONS:
        target, known = native_classes(rows, horizon)
        support[horizon] = {"known": int(known.sum()), "up": int((target[known] == 1).sum()), "down": int((target[known] == 2).sum())}
    target, known = native_classes(rows, 300)
    if set(target[known]) != {0, 1, 2}:
        raise AssertionError("Native smoke day needs both five-tick directions")
    # Feature count and sentinels are real native inputs. The fit is deliberately
    # not a chronological experiment and produces no predictive acceptance.
    values = FeaturesData(
        num_feature_data=np.asarray(x[known], dtype=np.float32),
        cat_feature_data=categorical.loc[known].to_numpy(dtype=object),
        num_feature_names=numeric,
        cat_feature_names=list(nominal),
    )
    pool = Pool(values, label=target[known])
    model = CatBoostClassifier(iterations=10, depth=3, task_type="GPU", devices="0", nan_mode="Max", one_hot_max_size=64, allow_writing_files=False, verbose=False)
    model.fit(pool)
    prediction = model.predict_proba(pool)
    if list(model.classes_) != [0, 1, 2] or not np.isfinite(prediction).all() or not np.allclose(prediction.sum(axis=1), 1):
        raise AssertionError("Native three-class/nominal model input is invalid")
    for side in ("up", "down"):
        binary_target = rows.loc[known, f"mid_endpoint.{side}.5[300s]"].to_numpy()
        binary = CatBoostClassifier(
            iterations=10, depth=3, loss_function="Logloss", task_type="GPU", devices="0", nan_mode="Max", one_hot_max_size=64, allow_writing_files=False, verbose=False
        )
        binary.fit(Pool(values, label=binary_target))
        probabilities = binary.predict_proba(values)
        if list(binary.classes_) != [0, 1] or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
            raise AssertionError("Native binary/nominal input is invalid")
        binary.save_model(str(args.output / f"binary-{side}.cbm"))
    write_yaml(
        args.output / "results.yaml",
        {
            "native_artifact": ref.document(),
            "day": dataset.metadata["days"],
            "numeric_features": len(numeric),
            "native_nominal_features": len(nominal),
            "native_endpoint_support": support,
            "GPU_model_fit": "passed",
            "native_binary_GPU_model_fit": "passed_up_and_down_original_native_indicators",
            "predictive_evidence": False,
            "row_count": len(rows),
        },
    )
    print(read_yaml(args.output / "results.yaml"), flush=True)


if __name__ == "__main__":
    main()
