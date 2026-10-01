"""Fixed CatBoost formulation per arm, stage and horizon; one GPU fit at a time.

An arm is a list of cached feature sets (optionally filtered by column prefix).
A stage fixes the fitting, early-stopping and evaluation roles and symbol
groups. Every fit uses the profile's training section unchanged: MultiClass on
down / no large move / up at the threshold, identical parameters, and the seed
ensemble (mean of per-seed probabilities). A process-wide file lock serializes
all GPU training on this machine.
"""

from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ...io import write_yaml
from . import dataset
from . import profile as P

STAGES = {
    "dev": {"train": ("dev_train", "development"), "es": ("dev_es", "development"), "eval": ("dev_val", "development")},
    # Cross-symbol development check: fit on half the development symbols,
    # score the other half; holdout symbols stay unread.
    "devx": {"train": ("dev_train", "dev_a"), "es": ("dev_es", "dev_a"), "eval": ("dev_val", "dev_b")},
    "final": {"train": ("final_train", "development"), "es": ("final_es", "development"), "eval": ("oos", "all")},
}
GPU_LOCK = Path(__file__).resolve().parents[2] / "runs" / ".fe-gpu.lock"


@contextmanager
def gpu_lock():
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def classes(y, threshold):
    c = np.ones(len(y), dtype=np.int32)
    c[y >= threshold] = 2
    c[y <= -threshold] = 0
    return c


def role_rows(profile, cohort, role, group):
    days = set(P.roles(profile)[role])
    symbols = set(P.symbols(profile, group))
    return np.flatnonzero(np.isin(cohort["day"], list(days)) & np.isin(cohort["symbol"], list(symbols)))


def _slow_columns(profile, set_name, threshold):
    import csv

    path = P.work(profile) / "census" / f"{set_name}-constancy.csv"
    with path.open() as stream:
        return [r["column"] for r in csv.DictReader(stream) if r["constancy"] not in ("", "nan") and float(r["constancy"]) >= threshold]


def arm_columns(profile, arm: dict):
    """(set, numeric indices, categorical indices, names) per set of the arm."""
    out = []
    for item in arm["sets"]:
        name = item if isinstance(item, str) else item["set"]
        cache = dataset.open_cache(profile, name)
        include = None if isinstance(item, str) else item.get("include")
        exclude = [] if isinstance(item, str) else item.get("exclude", [])

        def keep(col, include, exclude):
            family = col.split(".")[0]
            if include is not None and family not in include and col not in include:
                return False
            return family not in exclude and col not in exclude

        if isinstance(item, dict) and item.get("exclude_slow") is not None:
            # Day-constancy measured on dev_train only (AstraResearch.Experiments.fe.constancy); columns at or
            # above the threshold describe slow state and are kept out of this stage.
            slow = _slow_columns(profile, name, float(item["exclude_slow"]))
            exclude = [*exclude, *slow]
        num = [j for j, c in enumerate(cache["numeric"]) if keep(c, include, exclude)]
        cat = [j for j, c in enumerate(cache["categorical"]) if keep(c, include, exclude)] if not (isinstance(item, dict) and item.get("numeric_only")) else []
        out.append((name, cache, num, cat))
    return out


def materialize(profile, arm, rows):
    num_blocks, cat_blocks, num_names, cat_names = [], [], [], []
    for name, cache, num, cat in arm_columns(profile, arm):
        if num:
            block = cache["X"][rows]
            num_blocks.append(np.asarray(block if len(num) == block.shape[1] else block[:, num], dtype=np.float32))
            num_names += [f"{name}:{cache['numeric'][j]}" for j in num]
        if cat:
            codes = np.asarray(cache["C"][rows][:, cat])
            block = np.empty(codes.shape, dtype=object)
            for k, j in enumerate(cat):
                words = np.asarray([str(w).encode() for w in cache["vocab"][j]] + [b"<missing>"], dtype=object)
                block[:, k] = words[np.where(codes[:, k] >= 0, codes[:, k], len(words) - 1)]
            cat_blocks.append(block)
            cat_names += [f"{name}:{cache['categorical'][j]}" for j in cat]
    X = np.concatenate(num_blocks, axis=1) if num_blocks else np.zeros((len(rows), 0), np.float32)
    # CatBoost's quantizer cannot order infinities; keep the sentinel's side.
    X[np.isneginf(X)] = -1e30
    X[np.isposinf(X)] = 1e30
    C = np.concatenate(cat_blocks, axis=1) if cat_blocks else None
    return X, C, num_names, cat_names


def pool(X, C, y=None):
    from catboost import FeaturesData, Pool

    if C is None:
        return Pool(X, y)
    return Pool(FeaturesData(num_feature_data=X, cat_feature_data=C), y)


def _params(cfg, seed, loss):
    params = {
        "loss_function": loss,
        "iterations": cfg["iterations"],
        "learning_rate": cfg["learning_rate"],
        "depth": cfg["depth"],
        "l2_leaf_reg": cfg["l2_leaf_reg"],
        "border_count": cfg["border_count"],
        "task_type": cfg["task_type"],
        "od_type": "Iter",
        "od_wait": cfg["od_wait"],
        "random_seed": seed,
        "verbose": 0,
        "use_best_model": True,
        "allow_writing_files": False,
    }
    if cfg["task_type"] == "GPU":
        params.update(devices="0", gpu_ram_part=float(cfg.get("gpu_ram_part", 0.85)))
    return params


def combine(big, up):
    """Joint (down, none, up) from P(big) and P(up | big)."""
    big, up = np.asarray(big, np.float64), np.asarray(up, np.float64)
    return np.column_stack([big * (1.0 - up), 1.0 - big, big * up])


def fit_arm_factorized(profile, round_name, arm_name, arm, stage, horizons=None, seeds=None, stride=None):
    """Magnitude stage on every row, direction stage on large-move rows only.

    Each stage may use its own sets (`magnitude_sets`, `direction_sets`); slow
    state can then inform whether a move happens without deciding its sign.
    """
    from catboost import CatBoostClassifier

    cfg = profile["training"]
    threshold = float(profile["labels"]["threshold_ticks"])
    horizons = [int(h) for h in (horizons or profile["labels"]["horizons"])]
    seeds = list(seeds or cfg["seeds"])
    stride = int(stride or cfg.get("train_stride", 1))
    cohort = dataset.load_cohort(profile)
    all_h = [int(h) for h in cohort["horizons"]]
    spec = STAGES[stage]
    rows = {k: role_rows(profile, cohort, *spec[k]) for k in spec}
    # The direction stage sees large-move rows only; it may sample training origins
    # more densely than the magnitude stage (direction_stride, default train_stride).
    direction_stride = int(cfg.get("direction_stride", stride))
    direction_train = rows["train"][::direction_stride]
    rows["train"] = rows["train"][::stride]
    out = P.work(profile) / "fits" / round_name / arm_name
    out.mkdir(parents=True, exist_ok=True)
    roles = {"magnitude": {"sets": arm.get("magnitude_sets", arm["sets"])}, "direction": {"sets": arm.get("direction_sets", arm["sets"])}}
    t0 = time.time()
    data = {"magnitude": {k: materialize(profile, roles["magnitude"], r) for k, r in rows.items()}}
    same = roles["direction"] == roles["magnitude"]
    data["direction"] = {k: data["magnitude"][k] if same else materialize(profile, roles["direction"], rows[k]) for k in ("es", "eval")}
    names = {"magnitude": data["magnitude"]["train"][2] + data["magnitude"]["train"][3], "direction": data["direction"]["eval"][2] + data["direction"]["eval"][3]}
    receipt = {
        "arm": arm,
        "stage": stage,
        "formulation": "factorized",
        "direction_stride": direction_stride,
        "columns": {r: len(n) for r, n in names.items()},
        "load_seconds": round(time.time() - t0, 1),
        "horizons": {},
    }
    for h in horizons:
        k = all_h.index(h)
        target = out / f"{stage}-h{h}.npz"
        if target.exists():
            continue
        valid = {part: cohort["valid"][rows[part], k] for part in rows}
        y = {part: cohort["y"][rows[part], k] for part in rows}
        big = {part: valid[part] & (np.abs(y[part]) >= threshold) for part in rows}
        dense_y = cohort["y"][direction_train, k]
        dense_big = cohort["valid"][direction_train, k] & (np.abs(dense_y) >= threshold)
        data["direction"]["train"] = materialize(profile, roles["direction"], direction_train[dense_big])

        def subset(role, part, mask):
            X, C = data[role][part][0], data[role][part][1]
            return X[mask], None if C is None else C[mask]

        mag_pools = {
            "train": pool(*subset("magnitude", "train", valid["train"]), big["train"][valid["train"]]),
            "es": pool(*subset("magnitude", "es", valid["es"]), big["es"][valid["es"]]),
            "eval": pool(*subset("magnitude", "eval", valid["eval"])),
        }
        dir_pools = {
            "train": pool(data["direction"]["train"][0], data["direction"]["train"][1], dense_y[dense_big] > 0),
            "es": pool(*subset("direction", "es", big["es"]), y["es"][big["es"]] > 0),
            "eval": pool(*subset("direction", "eval", valid["eval"])),
        }
        probs, info = [], []
        for seed in seeds:
            stage_out = {}
            for role, pools in (("magnitude", mag_pools), ("direction", dir_pools)):
                model = CatBoostClassifier(**_params(cfg, seed, "Logloss"))
                f0 = time.time()
                with gpu_lock():
                    model.fit(pools["train"], eval_set=pools["es"])
                stage_out[role] = model.predict_proba(pools["eval"])[:, 1]
                model.save_model(str(out / f"{stage}-h{h}-s{seed}-{role}.cbm"))
                importance = model.get_feature_importance(type="PredictionValuesChange")
                (out / f"{stage}-h{h}-s{seed}-{role}-importance.json").write_text(json.dumps(sorted(zip(names[role], map(float, importance)), key=lambda t: -t[1])))
                info.append({"seed": seed, "role": role, "best_iteration": int(model.get_best_iteration()), "fit_seconds": round(time.time() - f0, 1)})
                print(f"{round_name}/{arm_name} {stage} h={h} seed={seed} {role} best={model.get_best_iteration()} {time.time() - f0:.0f}s", flush=True)
            probs.append(combine(stage_out["magnitude"], stage_out["direction"]).astype(np.float32))
        eval_index = rows["eval"][valid["eval"]]
        np.savez_compressed(target, index=eval_index, p=np.mean(probs, axis=0), p_seeds=np.stack(probs))
        receipt["horizons"][str(h)] = {
            "train_rows": int(valid["train"].sum()),
            "train_big_rows": int(big["train"].sum()),
            "direction_train_rows": int(dense_big.sum()),
            "eval_rows": int(valid["eval"].sum()),
            "seeds": info,
        }
        write_yaml(out / f"{stage}-receipt-h{h}.yaml", receipt)
    return receipt


def fit_arm(profile, round_name, arm_name, arm, stage, horizons=None, seeds=None, stride=None):
    if profile["training"].get("formulation", "multiclass") == "factorized":
        return fit_arm_factorized(profile, round_name, arm_name, arm, stage, horizons, seeds, stride)
    return fit_arm_multiclass(profile, round_name, arm_name, arm, stage, horizons, seeds, stride)


def fit_arm_multiclass(profile, round_name, arm_name, arm, stage, horizons=None, seeds=None, stride=None):
    """Fit every horizon of one arm and stage; persist eval probabilities and receipts."""
    from catboost import CatBoostClassifier

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
    t0 = time.time()
    data = {k: materialize(profile, arm, r) for k, r in rows.items()}
    names = data["train"][2] + data["train"][3]
    load_seconds = time.time() - t0
    receipt = {
        "arm": arm,
        "stage": stage,
        "columns": len(names),
        "numeric": len(data["train"][2]),
        "categorical": len(data["train"][3]),
        "load_seconds": round(load_seconds, 1),
        "horizons": {},
    }
    for h in horizons:
        k = all_h.index(h)
        target = out / f"{stage}-h{h}.npz"
        if target.exists():
            continue
        m = {part: cohort["valid"][rows[part], k] for part in rows}
        y = {part: classes(cohort["y"][rows[part], k], threshold) for part in rows}
        sub = {part: (data[part][0][m[part]], None if data[part][1] is None else data[part][1][m[part]]) for part in rows}
        probs, info = [], []
        # Pools are quantized once per horizon and shared by every seed.
        train_pool = pool(*sub["train"], y["train"][m["train"]])
        es_pool = pool(*sub["es"], y["es"][m["es"]])
        eval_pool = pool(*sub["eval"])
        for seed in seeds:
            params = {
                "loss_function": cfg["loss"],
                "iterations": cfg["iterations"],
                "learning_rate": cfg["learning_rate"],
                "depth": cfg["depth"],
                "l2_leaf_reg": cfg["l2_leaf_reg"],
                "border_count": cfg["border_count"],
                "task_type": cfg["task_type"],
                "od_type": "Iter",
                "od_wait": cfg["od_wait"],
                "random_seed": seed,
                "verbose": 0,
                "use_best_model": True,
                "allow_writing_files": False,
            }
            if cfg["task_type"] == "GPU":
                params.update(devices="0", gpu_ram_part=float(cfg.get("gpu_ram_part", 0.85)))
            model = CatBoostClassifier(**params)
            f0 = time.time()
            with gpu_lock():
                model.fit(train_pool, eval_set=es_pool)
            fit_seconds = time.time() - f0
            p = model.predict_proba(eval_pool)
            probs.append(p.astype(np.float32))
            model.save_model(str(out / f"{stage}-h{h}-s{seed}.cbm"))
            importance = model.get_feature_importance(type="PredictionValuesChange")
            top = sorted(zip(names, map(float, importance)), key=lambda t: -t[1])
            (out / f"{stage}-h{h}-s{seed}-importance.json").write_text(json.dumps(top))
            info.append({"seed": seed, "best_iteration": int(model.get_best_iteration()), "fit_seconds": round(fit_seconds, 1)})
            print(
                f"{round_name}/{arm_name} {stage} h={h} seed={seed} rows={int(m['train'].sum())} cols={len(names)} best={model.get_best_iteration()} {fit_seconds:.0f}s", flush=True
            )
        eval_index = rows["eval"][m["eval"]]
        np.savez_compressed(target, index=eval_index, p=np.mean(probs, axis=0), p_seeds=np.stack(probs))
        receipt["horizons"][str(h)] = {"train_rows": int(m["train"].sum()), "es_rows": int(m["es"].sum()), "eval_rows": int(m["eval"].sum()), "seeds": info}
        write_yaml(out / f"{stage}-receipt-h{h}.yaml", receipt)
    return receipt
