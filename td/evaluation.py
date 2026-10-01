"""Fixed CatBoost budget and one calibration/holding policy for every target."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from ...feature_values import model_values
from ...io import ContractError, digest, write_yaml
from .targets import column, project


def binary_loss(actual, prediction):
    p = np.clip(prediction, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return -np.mean(actual * np.log(p) + (1 - actual) * np.log1p(-p))


def average_precision(actual, prediction):
    order = np.argsort(-prediction, kind="stable")
    y, p = actual[order], prediction[order]
    endpoints = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]
    tp = np.cumsum(y)[endpoints]
    return float(np.sum(np.diff(np.r_[0, tp]) * tp / (endpoints + 1)) / y.sum()) if y.sum() else 0.0


def metrics(y, prediction, training_y, task, *, minimum_rows=1, minimum_class_count=1):
    heads = []
    for i in range(y.shape[1]):
        actual, pred, train = y[:, i], prediction[:, i], training_y[:, i]
        observed = np.isfinite(actual) & np.isfinite(pred)
        actual, pred, train = actual[observed], pred[observed], train[np.isfinite(train)]
        support = {"rows": len(y), "observed_rows": int(observed.sum()), "unknown_rows": int((~observed).sum()), "coverage": float(observed.mean())}
        enough = len(actual) >= minimum_rows and len(train) > 0
        if task == "binary_pair":
            counts = [int((actual == value).sum()) for value in [0, 1]]
            support["class_counts"] = counts
            enough = enough and min(counts) >= minimum_class_count
        else:
            enough = enough and np.var(actual) > 0
        if not enough:
            heads.append({**support, "status": "insufficient_observed_support", "skill": None})
            continue
        if task == "regression":
            mse = np.mean((actual - pred) ** 2)
            baseline = np.mean((actual - train.mean()) ** 2)
            rank = pd.Series(actual).corr(pd.Series(pred), method="spearman") if np.std(actual) and np.std(pred) else 0.0
            heads.append(
                {
                    **support,
                    "status": "measured",
                    "r2": float(1 - mse / np.var(actual)) if np.var(actual) > 0 else 0.0,
                    "skill": float(1 - mse / baseline) if baseline > 0 else 0.0,
                    "rank_correlation": float(rank),
                    "prediction_std": float(np.std(pred)),
                    "calibration_bias": float(np.mean(pred - actual)),
                }
            )
        else:
            prevalence = float(train.mean())
            loss = float(binary_loss(actual, pred))
            baseline = float(binary_loss(actual, np.full(len(actual), prevalence)))
            k = max(1, int(np.ceil(0.1 * len(actual))))
            order = np.argsort(-pred, kind="stable")[:k]
            calibration = sum(
                float(np.mean(mask)) * abs(float(pred[mask].mean() - actual[mask].mean()))
                for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True)
                if np.any(mask := (pred >= lo) & (pred < hi if hi < 1 else pred <= hi))
            )
            heads.append(
                {
                    **support,
                    "status": "measured",
                    "log_loss": loss,
                    "baseline_log_loss": baseline,
                    "skill": (baseline - loss) / baseline if baseline > 0 else 0.0,
                    "pr_auc": float(average_precision(actual, pred)),
                    "precision_at_10pct": float(actual[order].mean()),
                    "prevalence": float(actual.mean()),
                    "calibration_error": calibration,
                    "prediction_std": float(np.std(pred)),
                }
            )
    measured = all(head["status"] == "measured" for head in heads)
    return {
        "task": task,
        "basis": "conditional_on_observed_target_labels_and_finite_predictions",
        "status": "measured" if measured else "insufficient_observed_support",
        "heads": heads,
        "skill": float(np.mean([h["skill"] for h in heads])) if measured else None,
    }


def policy_fit(prediction, returns, cfg):
    # A fixed scalar projection for two independently fitted directional heads.
    score = prediction[:, 0] if prediction.shape[1] == 1 else prediction[:, 0] - prediction[:, 1]
    if not len(score) or not np.isfinite(score).all():
        raise ContractError("TD calibration requires finite predictions on every captured origin")
    boundaries = np.unique(np.quantile(score, np.linspace(0, 1, cfg["bins"] + 1)[1:-1]))
    bins = np.searchsorted(boundaries, score, side="right")
    returns = np.asarray(returns, dtype=float)
    observed = np.isfinite(returns)
    counts = [int((bins == i).sum()) for i in range(len(boundaries) + 1)]
    known = [int(((bins == i) & observed).sum()) for i in range(len(boundaries) + 1)]
    means = [float(returns[(bins == i) & observed].mean()) if known[i] else None for i in range(len(boundaries) + 1)]
    cut = cfg["round_trip_ticks"] + cfg["margin_ticks"]
    return {
        "boundaries": boundaries.tolist(),
        "boundary_population": "all_captured_calibration_origins",
        "return_estimate": "conditional_on_observed_own_economic_horizon",
        "return_ticks": means,
        "actions": [None if mean is None else 1 if mean > cut else -1 if mean < -cut else 0 for mean in means],
        "bin_rows": counts,
        "bin_observed_returns": known,
        "bin_unknown_returns": [n - k for n, k in zip(counts, known, strict=True)],
        "unavailable_bins": [i for i, n in enumerate(known) if not n],
        "calibration_rows": len(score),
    }


def policy_apply(prediction, policy):
    score = prediction[:, 0] if prediction.shape[1] == 1 else prediction[:, 0] - prediction[:, 1]
    actions = np.asarray(policy["actions"], dtype=float)[np.searchsorted(policy["boundaries"], score, side="right")]
    return np.where(np.isfinite(score), actions, np.nan)


def economics(frame, actions, cfg):
    h, cost = cfg["horizon_seconds"], cfg["round_trip_ticks"]
    actions = np.asarray(actions, dtype=float)
    policy_available = np.isfinite(actions)
    if len(actions) != len(frame) or not np.isin(actions[policy_available], [-1, 0, 1]).all():
        raise ContractError("TD policy actions must be directional, neutral or unavailable")
    accepted = np.zeros(len(frame), dtype=bool)
    # One constant-notional position per symbol; no overlapping returns counted
    # as independent trades. Different symbols have separate fixed capital.
    times = frame.SampleTime.to_numpy()
    for indices in frame.groupby(["day", "symbol"], sort=True).indices.values():
        available = -1
        for i in indices:
            time = int(times[i])
            if policy_available[i] and actions[i] and time >= available:
                accepted[i] = True
                available = time + h * 1_000_000
    # The complete holding schedule is fixed before reading any return.
    returns = frame[column("return", h)].to_numpy(dtype=float)
    return_observed = np.isfinite(returns)
    resolved = accepted & return_observed
    unresolved = accepted & ~return_observed
    net = np.zeros(len(frame))
    net[resolved] = actions[resolved] * returns[resolved] - cost
    net[unresolved | ~policy_available] = np.nan
    rows = frame[["sample_id", "day", "symbol", "SampleTime"]].copy()
    rows["action"], rows["trigger"], rows["net_ticks"] = actions, accepted, net
    rows["policy_available"], rows["return_observed"], rows["unresolved_trigger"] = policy_available, return_observed, unresolved
    rows["economic_status"] = np.select([~policy_available, unresolved, resolved], ["policy_unavailable", "trigger_return_unknown", "trigger_resolved"], default="no_trigger")
    total = int(accepted.sum())
    complete = bool(np.isfinite(net).all())
    groups = []
    for axis in ["day", "symbol"]:
        for name, group in rows.groupby(axis, sort=True):
            count = int(group.trigger.sum())
            group_complete = bool(group.net_ticks.notna().all())
            groups.append(
                {
                    "axis": axis,
                    "name": str(name),
                    "rows": len(group),
                    "triggers": count,
                    "net_ticks": float(group.net_ticks.sum()) if group_complete else None,
                    "net_ticks_per_opportunity": float(group.net_ticks.mean()) if group_complete else None,
                    "observed_net_ticks_sum": float(group.loc[group.trigger & group.return_observed, "net_ticks"].sum()),
                    "unresolved_triggers": int(group.unresolved_trigger.sum()),
                    "policy_unavailable_rows": int((~group.policy_available).sum()),
                    "coverage": count / len(group),
                }
            )
    daily = [g for g in groups if g["axis"] == "day"]
    symbols = [g for g in groups if g["axis"] == "symbol"]
    return {
        "basis": "sticky_fixed_holding_cost_proxy_not_fills_or_native_pnl",
        "rows": len(frame),
        "triggers": total,
        "resolved_triggers": int(resolved.sum()),
        "unresolved_triggers": int(unresolved.sum()),
        "policy_unavailable_rows": int((~policy_available).sum()),
        "evaluation_complete": complete,
        "incomplete_reasons": (["policy_unavailable"] if not policy_available.all() else []) + (["trigger_return_unknown"] if unresolved.any() else []),
        "observed_net_ticks_sum": float(net[resolved].sum()),
        "observed_net_ticks_per_resolved_trade": float(net[resolved].mean()) if resolved.any() else None,
        "coverage": total / len(frame),
        "raw_trigger_coverage": float(np.mean(policy_available & (actions != 0))),
        "net_ticks_per_opportunity": float(net.mean()) if complete else None,
        "net_ticks_per_trade": (float(net.sum() / total) if total else 0.0) if complete else None,
        "positive_day_fraction": sum(g["net_ticks"] > 0 for g in daily) / len(daily) if complete else None,
        "positive_symbol_fraction": sum(g["net_ticks"] > 0 for g in symbols) / len(symbols) if complete else None,
        "max_symbol_share": max((g["triggers"] / total for g in symbols), default=0.0) if total else 1.0,
        "active_days": sum(g["triggers"] > 0 for g in daily),
        "active_symbols": sum(g["triggers"] > 0 for g in symbols),
        "groups": groups,
        "fixed_policy_across_days_and_symbols": True,
        "leave_one_day_out_min_net_ticks": min((float(net.sum()) - g["net_ticks"] for g in daily), default=0.0) if complete else None,
        "leave_one_symbol_out_min_net_ticks": min((float(net.sum()) - g["net_ticks"] for g in symbols), default=0.0) if complete else None,
    }, rows


def date_view(frame, days):
    """Read-only contiguous walk-forward roles share their numeric feature data."""
    mask = frame.day.isin(days)
    indices = np.flatnonzero(mask.to_numpy())
    if not len(indices) or indices[-1] - indices[0] + 1 != len(indices):
        return frame.loc[mask].reset_index(drop=True)
    result = frame.iloc[indices[0] : indices[-1] + 1].copy(deep=False)
    result.index = pd.RangeIndex(len(result))
    return result


def fold_evaluate(frame, features, target, split, cfg, *, save=None):
    roles = {role: date_view(frame, days) for role, days in split.items()}
    train, calibration, validation = [roles[k] for k in ["train", "calibration", "validation"]]
    g = cfg["gates"]
    audit = {role: {"days": split[role], "rows": len(data), "sample_digest": digest(data.sample_id.tolist())} for role, data in roles.items()}
    if any(len(data) < g["min_rows"] or sorted(data.day.unique()) != split[role] for role, data in roles.items()):
        return {"status": "insufficient_samples", "roles": audit, "fits": 0}, None
    max_h = max(cfg["truth"]["horizons_seconds"]) * 1_000_000
    if train.SampleTime.max() + max_h >= calibration.SampleTime.min() or calibration.SampleTime.max() + max_h >= validation.SampleTime.min():
        raise ContractError("Future truth overlaps a later split role")
    labels = {role: project(data, target, cfg["truth"]) for role, data in roles.items()}
    observed = {role: np.isfinite(y) for role, y in labels.items()}
    for role, mask in observed.items():
        audit[role]["target_heads"] = [
            {
                "observed_rows": int(mask[:, i].sum()),
                "unknown_rows": int((~mask[:, i]).sum()),
                "observed_sample_digest": digest(roles[role].sample_id[mask[:, i]].tolist()),
                "observed_days": sorted(roles[role].day[mask[:, i]].unique().tolist()),
            }
            for i in range(mask.shape[1])
        ]
    if any(mask.sum() < g["min_rows"] for mask in observed["train"].T):
        return {"status": "insufficient_observed_training_samples", "roles": audit, "fits": 0}, None
    if target["task"] == "binary_pair":
        support = {role: [[int((y[:, i] == c).sum()) for c in [0, 1]] for i in range(y.shape[1])] for role, y in labels.items()}
        if any(min(counts) < g["min_class_count"] for counts in support["train"]):
            return {"status": "insufficient_class_support", "roles": audit, "class_support": support, "fits": 0}, None
    elif any(np.var(y[mask]) == 0 for y, mask in zip(labels["train"].T, observed["train"].T, strict=True)):
        return {"status": "constant_target", "roles": audit, "fits": 0}, None
    p = cfg["proxy"]
    params = {
        "iterations": p["iterations"] // labels["train"].shape[1],
        "depth": p["depth"],
        "learning_rate": p["learning_rate"],
        "l2_leaf_reg": p["l2_leaf_reg"],
        "random_seed": p["seed"],
        "nan_mode": "Max",
        "thread_count": 1,
        "allow_writing_files": False,
        "verbose": False,
    }
    models = []
    training_x, categorical = model_values(train, features)
    params["cat_features"] = categorical
    for i in range(labels["train"].shape[1]):
        model = CatBoostRegressor(loss_function="RMSE", **params) if target["task"] == "regression" else CatBoostClassifier(loss_function="Logloss", **params)
        mask = observed["train"][:, i]
        model.fit(training_x.loc[mask], labels["train"][mask, i])
        models.append(model)

    def predict(data):
        x, _ = model_values(data, features)
        return np.column_stack([m.predict(x) if target["task"] == "regression" else m.predict_proba(x)[:, 1] for m in models])

    policy = policy_fit(predict(calibration), calibration[column("return", cfg["economics"]["horizon_seconds"])].to_numpy(), cfg["economics"])
    if save:
        save.mkdir(parents=True, exist_ok=True)
        write_yaml(
            save / "frozen-policy.yaml",
            {"target": target, "policy": policy, "fit_days": split["train"], "fit_population": audit["train"], "calibration_days": split["calibration"]},
        )
        for i, model in enumerate(models):
            model.save_model(str(save / f"head-{i}.cbm"))
    predictions = {"validation": predict(validation)}
    actions = policy_apply(predictions["validation"], policy)
    econ, rows = economics(validation, actions, cfg["economics"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prediction = metrics(labels["validation"], predictions["validation"], labels["train"], target["task"], minimum_rows=g["min_rows"], minimum_class_count=g["min_class_count"])
    for i in range(predictions["validation"].shape[1]):
        rows[f"prediction_{i}"] = predictions["validation"][:, i]
        rows[f"label_{i}"] = labels["validation"][:, i]
        rows[f"metric_observed_{i}"] = observed["validation"][:, i] & np.isfinite(predictions["validation"][:, i])
    result = {
        "status": "measured",
        "roles": audit,
        "fits": len(models),
        "trees": len(models) * params["iterations"],
        "model_parameters": params,
        "predictability": prediction,
        "economics": econ,
        "policy": policy,
        "policy_digest": digest(policy),
        "fit_population": "per_head_finite_target_labels_within_causal_train_origins",
        "decision_population": "all_causal_validation_origins",
    }
    return result, rows


def summarize(target, folds, cfg):
    measured = [f for f in folds if f["status"] == "measured"]
    result = {
        "target": target,
        "status": "inconclusive",
        "qualified": False,
        "folds": len(folds),
        "measured_folds": len(measured),
        "fits": sum(f["fits"] for f in folds),
        "trees": sum(f.get("trees", 0) for f in folds),
    }
    if len(measured) != len(folds):
        return result
    g = cfg["gates"]
    e = [f["economics"] for f in measured]
    skills = [f["predictability"]["skill"] for f in measured]
    values = [f["net_ticks_per_opportunity"] for f in e]
    result["support"] = {
        "predictive_metrics_observed": all(s is not None for s in skills),
        "economic_evaluation_complete": all(f["evaluation_complete"] for f in e),
        "unresolved_triggers": sum(f["unresolved_triggers"] for f in e),
        "policy_unavailable_rows": sum(f["policy_unavailable_rows"] for f in e),
    }
    if not result["support"]["predictive_metrics_observed"] or not result["support"]["economic_evaluation_complete"]:
        return result
    checks = {
        "predictability": float(np.mean(skills)) > 0,
        "fold_reproducibility": float(np.mean([(s > 0 and v > 0) for s, v in zip(skills, values, strict=True)])) >= g["positive_fold_fraction"],
        "economics": float(np.mean(values)) > 0,
        "not_one_day": all(f["leave_one_day_out_min_net_ticks"] > 0 for f in e),
        "not_one_symbol": all(f["leave_one_symbol_out_min_net_ticks"] > 0 for f in e),
        "coverage": all(f["triggers"] >= g["min_triggers"] and f["coverage"] >= g["min_coverage"] for f in e),
        "day_stability": all(f["active_days"] >= g["min_days"] and f["positive_day_fraction"] >= g["positive_day_fraction"] for f in e),
        "symbol_stability": all(
            f["active_symbols"] >= g["min_symbols"] and f["positive_symbol_fraction"] >= g["positive_symbol_fraction"] and f["max_symbol_share"] <= g["max_symbol_share"] for f in e
        ),
    }
    result.update(
        status="measured",
        qualified=all(checks.values()),
        checks=checks,
        skill=float(np.mean(skills)),
        predictive_stability=float(np.mean([s > 0 for s in skills])),
        economic_value=float(np.mean(values)),
        economic_std=float(np.std(values)),
        coverage=min(f["coverage"] for f in e),
        stability=float(np.mean([(s > 0 and v > 0) for s, v in zip(skills, values, strict=True)])),
    )
    return result


def null_features(frame, features, seed, horizon):
    """Joint circular feature shift within day/symbol, with temporal separation.

    Labels, future truth, class balance and economic paths remain untouched.
    No permutation crosses split roles (splits contain whole days).
    """
    order = np.arange(len(frame))
    rng = np.random.default_rng(seed)
    offsets = []
    for (day, symbol), indices in frame.groupby(["day", "symbol"], sort=True).indices.items():
        times = frame.iloc[indices].SampleTime.to_numpy()
        choices = rng.permutation(np.arange(1, len(indices)))
        shift = next((int(k) for k in choices if np.all(np.abs(times - np.roll(times, int(k))) > horizon * 1_000_000)), None)
        if shift is None:
            raise ContractError(f"Insufficient temporal span for separated null shift: {day}/{symbol}")
        order[indices] = np.roll(indices, shift)
        offsets.append({"day": str(day), "symbol": str(symbol), "shift_rows": shift})
    feature_names = set(features)
    result = pd.DataFrame({name: frame[name].array.take(order) if name in feature_names else frame[name].array for name in frame}, index=frame.index, copy=False)
    return result, offsets
