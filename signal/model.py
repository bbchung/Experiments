"""CatBoost fitting per horizon and held-out scoring."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import metrics


def classes(y, threshold):
    c = np.ones(len(y), dtype=np.int32)
    c[y >= threshold] = 2
    c[y <= -threshold] = 0
    return c


def fit(cfg: dict, Xtr, ytr, Xes, yes, loss="MultiClass", weights=None):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    params = {
        "iterations": cfg["iterations"],
        "learning_rate": cfg["learning_rate"],
        "depth": cfg["depth"],
        "l2_leaf_reg": cfg["l2_leaf_reg"],
        "border_count": cfg.get("border_count", 128),
        "task_type": cfg.get("task_type", "GPU"),
        "od_type": "Iter",
        "od_wait": cfg["od_wait"],
        "random_seed": cfg.get("seed", 17),
        "verbose": 0,
        "use_best_model": True,
    }
    if params["task_type"] == "GPU":
        params["gpu_ram_part"] = 0.9
    cls = CatBoostClassifier if loss in ("MultiClass", "Logloss") else CatBoostRegressor
    model = cls(loss_function=loss, **params)
    model.fit(Pool(Xtr, ytr, weight=weights), eval_set=Pool(Xes, yes))
    return model


def fit_and_score(cfg, threshold, train, es, test, h):
    """Test (up, down) scores for one horizon and the MultiClass model.

    multiclass: P(up), P(down) of a down / none / up classifier.
    hybrid:     P(big) = P(up) + P(down) from that classifier times P(up | big), a
                logistic calibration fitted on early-stopping big moves of a
                regression on the clipped move and the classifier's own up/down odds.
    """
    k = train["horizons"].index(h)
    mtr = train["ok"][h] & np.isfinite(train["Y"][:, k])
    mes = es["ok"][h] & np.isfinite(es["Y"][:, k])
    t0 = time.time()
    model = fit(cfg, train["X"][mtr], classes(train["Y"][mtr, k], threshold), es["X"][mes], classes(es["Y"][mes, k], threshold))
    p = model.predict_proba(test["X"])
    print(f"fit h={h} rows={int(mtr.sum())} cols={train['X'].shape[1]} best_iter={model.get_best_iteration()} {time.time() - t0:.0f}s", flush=True)
    if cfg.get("head", "multiclass") == "multiclass":
        return (p[:, 2].astype(np.float32), p[:, 0].astype(np.float32)), model
    from sklearn.linear_model import LogisticRegression

    clip = float(cfg.get("regression_clip_ticks", 10.0))
    reg = fit(cfg, train["X"][mtr], np.clip(train["Y"][mtr, k], -clip, clip), es["X"][mes], np.clip(es["Y"][mes, k], -clip, clip), loss="RMSE")

    def parts(X, probabilities):
        big = probabilities[:, 2] + probabilities[:, 0]
        ratio = np.clip(probabilities[:, 2] / np.maximum(big, 1e-12), 1e-6, 1 - 1e-6)
        return big, np.column_stack([reg.predict(X), np.log(ratio / (1 - ratio))])

    _, z_es = parts(es["X"][mes], model.predict_proba(es["X"][mes]))
    y_es = es["Y"][mes, k]
    moved = np.abs(y_es) >= threshold
    calibrator = LogisticRegression(C=1.0).fit(z_es[moved], (y_es[moved] > 0).astype(int))
    big, z = parts(test["X"], p)
    p_up = calibrator.predict_proba(z)[:, 1]
    print(f"hybrid h={h} direction coef={calibrator.coef_.round(3).tolist()} es big moves={int(moved.sum())}", flush=True)
    return ((big * p_up).astype(np.float32), (big * (1 - p_up)).astype(np.float32)), model


def run(profile, tag, train, es, test, out_dir: Path):
    """Fit every horizon, persist metrics, scores and importances under out_dir/tag.*"""
    cfg, threshold = profile["model"], profile["target"]["threshold_ticks"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result, scores = {"tag": tag, "columns": len(train["columns"]), "horizons": {}}, {}
    for h in profile["target"]["horizons"]:
        (up, down), model = fit_and_score(cfg, threshold, train, es, test, h)
        k = test["horizons"].index(h)
        m = test["ok"][h] & np.isfinite(test["Y"][:, k])
        r = metrics.evaluate(test["Y"][m, k], up[m], down[m], test["day"][m], test["symbol"][m], threshold)
        r["best_iteration"] = model.get_best_iteration()
        r["importance"] = sorted(zip(train["columns"], map(float, model.get_feature_importance())), key=lambda t: -t[1])
        result["horizons"][str(h)] = r
        scores[f"up{h}"], scores[f"dn{h}"] = up, down
        print(metrics.line(tag, h, r), flush=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=float))
    np.savez_compressed(out_dir / f"{tag}.scores.npz", day=test["day"], symbol=test["symbol"], time=test["time"], **scores)
    return result


def compare(profile, test, out_dir: Path, a_tag, b_tag, metric_names=("mean_ap", "mean_auc", "direction_auc", "p@0.01", "p@0.001"), n=200):
    a, b = np.load(out_dir / f"{a_tag}.scores.npz"), np.load(out_dir / f"{b_tag}.scores.npz")
    if not (np.array_equal(a["time"], test["time"]) and np.array_equal(b["time"], test["time"])):
        raise RuntimeError("compared scores were not produced on the same test rows")
    out = {}
    for h in profile["target"]["horizons"]:
        if f"up{h}" not in a or f"up{h}" not in b:
            continue
        k = test["horizons"].index(h)
        y = np.where(test["ok"][h], test["Y"][:, k], np.nan)
        for name in metric_names:
            r = metrics.paired_bootstrap(y, (a[f"up{h}"], a[f"dn{h}"]), (b[f"up{h}"], b[f"dn{h}"]), test["day"], profile["target"]["threshold_ticks"], name, n=n)
            out[f"{h}:{name}"] = r
            print(f"  {b_tag} - {a_tag} h={h:3d} {name:14s} {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] P(<=0)={r['p_le0']:.2f}", flush=True)
    (out_dir / f"cmp_{b_tag}__{a_tag}.json").write_text(json.dumps(out, indent=1))
    return out
