"""LightGBM MultiClass on numeric columns: a diagnostic model comparison.

Everything except the learner matches `AstraResearch.Experiments.fe.train`: cohort rows, roles,
training stride, down / none / up classes at the threshold, the early-stopping
role, seeds and the persisted output format, so `AstraResearch.Experiments.fe.compare` scores both
models identically. Native categorical columns are excluded. Parameters mirror
the CatBoost budget (depth-6 trees with 63 leaves, learning rate, L2, 128
bins, round cap and early-stopping patience); row and column subsampling are
the only stochastic elements, so seeds give distinct realizations.
"""

from __future__ import annotations

import contextlib
import copy
import json
import time

import numpy as np

from ...io import write_yaml
from . import dataset
from . import profile as P
from .train import STAGES, classes, gpu_lock, materialize, role_rows


def numeric_arm(arm: dict) -> dict:
    out = copy.deepcopy(arm)
    out["sets"] = [{"set": s, "numeric_only": True} if isinstance(s, str) else {**s, "numeric_only": True} for s in arm["sets"]]
    return out


def params(profile, seed: int) -> dict:
    cfg, lcfg = profile["training"], profile.get("lightgbm", {})
    return {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": cfg["learning_rate"],
        "max_depth": cfg["depth"],
        "num_leaves": 2 ** cfg["depth"] - 1,
        "lambda_l2": cfg["l2_leaf_reg"],
        "max_bin": cfg["border_count"],
        "min_data_in_leaf": lcfg.get("min_data_in_leaf", 20),
        "feature_fraction": lcfg.get("feature_fraction", 0.8),
        "bagging_fraction": lcfg.get("bagging_fraction", 0.8),
        "bagging_freq": 1,
        "seed": seed,
        "num_threads": lcfg.get("num_threads", 24),
        # "cuda" needs a CUDA build of LightGBM; GPU fits share the CatBoost GPU lock.
        "device_type": lcfg.get("device_type", "cpu"),
        "verbosity": -1,
    }


def fit_arm(profile, round_name, arm_name, arm, stage, horizons=None, seeds=None, stride=None):
    import lightgbm as lgb

    cfg = profile["training"]
    threshold = float(profile["labels"]["threshold_ticks"])
    horizons = [int(h) for h in (horizons or profile["labels"]["horizons"])]
    seeds = list(seeds or cfg["seeds"])
    stride = int(stride or cfg.get("train_stride", 1))
    cohort = dataset.load_cohort(profile)
    all_h = [int(h) for h in cohort["horizons"]]
    spec = STAGES[stage]
    rows = {k: role_rows(profile, cohort, *spec[k]) for k in spec}
    rows["train"] = rows["train"][::stride]
    out = P.work(profile) / "fits" / round_name / arm_name
    out.mkdir(parents=True, exist_ok=True)
    narm = numeric_arm(arm)
    t0 = time.time()
    data = {k: materialize(profile, narm, r) for k, r in rows.items()}
    names = data["train"][2]
    receipt = {"arm": arm, "model": "lightgbm", "stage": stage, "columns": len(names), "load_seconds": round(time.time() - t0, 1), "horizons": {}}
    for h in horizons:
        k = all_h.index(h)
        target = out / f"{stage}-h{h}.npz"
        if target.exists():
            continue
        m = {part: cohort["valid"][rows[part], k] for part in rows}
        y = {part: classes(cohort["y"][rows[part], k], threshold) for part in rows}
        train_set = lgb.Dataset(data["train"][0][m["train"]], label=y["train"][m["train"]], feature_name=[f"f{i}" for i in range(len(names))], free_raw_data=False)
        es_set = lgb.Dataset(data["es"][0][m["es"]], label=y["es"][m["es"]], reference=train_set, free_raw_data=False)
        X_eval = data["eval"][0][m["eval"]]
        probs, info = [], []
        for seed in seeds:
            f0 = time.time()
            settings = params(profile, seed)
            with gpu_lock() if settings["device_type"] != "cpu" else contextlib.nullcontext():
                booster = lgb.train(
                    settings,
                    train_set,
                    num_boost_round=cfg["iterations"],
                    valid_sets=[es_set],
                    callbacks=[lgb.early_stopping(cfg["od_wait"], verbose=False)],
                )
            fit_seconds = time.time() - f0
            p = booster.predict(X_eval, num_iteration=booster.best_iteration)
            probs.append(np.asarray(p, dtype=np.float32))
            booster.save_model(str(out / f"{stage}-h{h}-s{seed}.txt"), num_iteration=booster.best_iteration)
            gain = booster.feature_importance(importance_type="gain", iteration=booster.best_iteration)
            (out / f"{stage}-h{h}-s{seed}-importance.json").write_text(json.dumps(sorted(zip(names, map(float, gain)), key=lambda t: -t[1])))
            info.append({"seed": seed, "best_iteration": int(booster.best_iteration), "fit_seconds": round(fit_seconds, 1)})
            print(
                f"{round_name}/{arm_name} lightgbm {stage} h={h} seed={seed} rows={int(m['train'].sum())} cols={len(names)} best={booster.best_iteration} {fit_seconds:.0f}s",
                flush=True,
            )
        np.savez_compressed(target, index=rows["eval"][m["eval"]], p=np.mean(probs, axis=0), p_seeds=np.stack(probs))
        receipt["horizons"][str(h)] = {"train_rows": int(m["train"].sum()), "es_rows": int(m["es"].sum()), "eval_rows": int(m["eval"].sum()), "seeds": info}
        write_yaml(out / f"{stage}-receipt-h{h}.yaml", receipt)
    return receipt
