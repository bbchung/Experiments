"""Claim-conforming prototype, pending an independently frozen method study.

Meaningful gain thresholds retain the frozen legacy controls; every
noninferiority margin is exactly zero. Neither is tuned to candidate results.
This prototype evaluates large-event probability claims, not conditional
absolute-amplitude regression claims. It does not train or select features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ...io import ContractError
from ...statistics import block_interval
from ..representation_research.data import native_classes
from ..representation_research.evaluation import nonoverlap
from ..signal.metrics import auc, average_precision
from ..signal.metrics import evaluate as joint_evaluate

CLAIMS = {"direction", "magnitude", "both"}
PAIRED_METRICS = ("mean_ap", "big_ap", "conditional_direction_auc")


def factorized_metrics(rows, p_big, q, horizon):
    """Use explicit P(big), P(up|big); never reconstruct the magnitude head.

    The joint up/down components can differ by an arithmetic ULP in their sum.
    Their products define joint ranking; the original heads define component
    metrics, preserving magnitude ties and zero/nonzero state boundaries.
    """
    p_big, q = np.asarray(p_big, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p_big.shape != (len(rows),) or q.shape != p_big.shape:
        raise ContractError("Factorized heads must match every native row")
    if not np.isfinite(p_big).all() or not np.isfinite(q).all() or (p_big <= 0).any() or (p_big > 1).any() or (q < 0).any() or (q > 1).any():
        raise ContractError("Factorized heads require finite 0 < P(big) <= 1 and 0 <= P(up|big) <= 1; no epsilon repair")
    scores = np.column_stack([p_big * q, p_big * (1 - q)])
    if (scores.sum(axis=1) <= 0).any():
        raise ContractError("Joint probability arithmetic crossed the positive P(big) state boundary")
    labels, known = native_classes(rows, horizon)
    rows = rows.loc[known].reset_index(drop=True)
    labels, p_big, q, scores = labels[known], p_big[known], q[known], scores[known]
    if not len(rows):
        raise ContractError("No mature native endpoints")
    big, up = labels != 0, labels == 1
    move = rows[f"mid_return_ticks[{horizon}s]"].to_numpy(dtype=float)
    result = joint_evaluate(move, scores[:, 0], scores[:, 1], rows.day.to_numpy(), rows.symbol.to_numpy(), 5)
    result["big_ap"], result["big_base"] = average_precision(big, p_big), float(big.mean())
    result["conditional_direction_auc"] = auc(up[big], q[big])
    result["absolute_move_spearman"] = float(spearmanr(np.abs(move), p_big).statistic) if np.unique(move).size > 1 and np.unique(p_big).size > 1 else np.nan
    magnitude, direction = [], []
    for indices in rows.groupby(["day", "symbol"], sort=True).indices.values():
        if 0 < big[indices].sum() < len(indices):
            magnitude.append(auc(big[indices], p_big[indices]))
        events = indices[big[indices]]
        if 0 < up[events].sum() < len(events):
            direction.append(auc(up[events], q[events]))
    result["equal_symbol_day_big_auc"] = float(np.mean(magnitude)) if magnitude else np.nan
    result["equal_symbol_day_direction_auc"] = float(np.mean(direction)) if direction else np.nan
    sparse = nonoverlap(rows, horizon)
    result["nonoverlap"] = {
        "rows": int(sparse.sum()),
        "up": int((up & sparse).sum()),
        "down": int(((labels == 2) & sparse).sum()),
        "up_days": int(rows.loc[up & sparse, "day"].nunique()),
        "down_days": int(rows.loc[(labels == 2) & sparse, "day"].nunique()),
        "big_ap": average_precision(big[sparse], p_big[sparse]),
        "side_ap": float(np.nanmean([average_precision(up[sparse], scores[sparse, 0]), average_precision(labels[sparse] == 2, scores[sparse, 1])])),
        "conditional_direction_auc": auc(up[sparse & big], q[sparse & big]),
    }
    result["head_contract"] = "explicit_p_big_and_q_v1"
    return result


def factorized_daily(rows, p_big, q, horizon):
    records = []
    p_big, q = np.asarray(p_big), np.asarray(q)
    for day, indices in rows.groupby("day", sort=True).indices.items():
        values = factorized_metrics(rows.iloc[indices].reset_index(drop=True), p_big[indices], q[indices], horizon)
        records.append({"day": str(day), **{key: values[key] for key in PAIRED_METRICS}})
    return pd.DataFrame(records)


def _at_least(value, reference, *, strict=False):
    return bool(np.isfinite(value) and np.isfinite(reference) and (value > reference if strict else value >= reference))


def claim_decision(profile, baseline, candidate, baseline_daily=None, candidate_daily=None, *, claim, stage="calibration"):
    """Research prototype; inputs use existing native V1 metric dictionaries.

    Stage calibration nominates without future labels. A development decision
    additionally requires paired chronological bounds, nonoverlap support and
    both frozen symbol groups. Development evidence is never pristine OOS.
    """
    if claim not in CLAIMS or stage not in {"calibration", "development", "sealed_oos"}:
        raise ContractError("Claim and stage must be explicit before evaluation")
    cfg = profile["acceptance"]
    if cfg.get("direction_noninferiority_margin", 0) != 0 or cfg.get("magnitude_noninferiority_margin", 0) != 0:
        raise ContractError("This method prototype has fixed zero noninferiority margins")
    direction_claim = claim in {"direction", "both"}
    magnitude_claim = claim in {"magnitude", "both"}
    checks = {
        "joint_gain": _at_least(candidate["mean_ap"], baseline["mean_ap"] * (1 + cfg["relative_ap_gain"])),
        "magnitude_claim_gain" if magnitude_claim else "magnitude_noninferiority": _at_least(
            candidate["big_ap"], baseline["big_ap"] * (1 + cfg["relative_ap_gain"]) if magnitude_claim else baseline["big_ap"]
        ),
        "direction_claim_gain" if direction_claim else "direction_noninferiority": _at_least(
            candidate["conditional_direction_auc"], baseline["conditional_direction_auc"] + cfg["direction_auc_gain"] if direction_claim else baseline["conditional_direction_auc"]
        ),
        "magnitude_power": _at_least(candidate["big_ap"], candidate["big_base"], strict=True),
        "direction_power": _at_least(candidate["conditional_direction_auc"], cfg["minimum_direction_auc"]),
        "equal_symbol_day_magnitude_power": _at_least(candidate["equal_symbol_day_big_auc"], 0.5, strict=True),
        "equal_symbol_day_direction_power": _at_least(candidate["equal_symbol_day_direction_auc"], 0.5, strict=True),
        "equal_symbol_day_magnitude_noninferiority": _at_least(candidate["equal_symbol_day_big_auc"], baseline["equal_symbol_day_big_auc"]),
        "equal_symbol_day_direction_noninferiority": _at_least(candidate["equal_symbol_day_direction_auc"], baseline["equal_symbol_day_direction_auc"]),
        "net_directional_tail_noninferiority": _at_least(candidate["mean_p@0.01"] - candidate["mean_wrong@0.01"], baseline["mean_p@0.01"] - baseline["mean_wrong@0.01"]),
        "nonoverlap_joint_noninferiority": _at_least(candidate["nonoverlap"]["side_ap"], baseline["nonoverlap"]["side_ap"]),
    }
    intervals = {}
    if stage != "calibration":
        if baseline_daily is None or candidate_daily is None or not baseline_daily.day.equals(candidate_daily.day):
            raise ContractError("Claim evidence requires the complete identical chronological day axis")
        for metric in PAIRED_METRICS:
            delta = candidate_daily[metric].to_numpy() - baseline_daily[metric].to_numpy()
            required_gain = metric == "mean_ap" or (metric == "big_ap" and magnitude_claim) or (metric == "conditional_direction_auc" and direction_claim)
            label = "gain" if required_gain else "noninferiority"
            if not np.isfinite(delta).all():
                checks[f"paired_{metric}_{label}"] = False
                intervals[metric] = {"resolution_sufficient": False, "reason": "undefined_daily_metric_not_dropped"}
            else:
                interval = block_interval(delta, cfg["block_days"], cfg["bootstrap_seed"], draws=cfg["bootstrap_draws"], tail_probability=cfg["tail_probability"])
                intervals[metric] = interval
                checks[f"paired_{metric}_{label}"] = bool(interval["resolution_sufficient"] and (interval["lower"] > 0 if required_gain else interval["lower"] >= 0))
        support = candidate["nonoverlap"]
        checks["nonoverlap_support"] = all(support[side] >= cfg["minimum_side_events"] and support[side + "_days"] >= cfg["minimum_event_days"] for side in ("up", "down"))
        sparse_base = (support["up"] + support["down"]) / support["rows"] if support["rows"] else np.nan
        checks["nonoverlap_magnitude_power"] = _at_least(support["big_ap"], sparse_base, strict=True)
        checks["nonoverlap_direction_power"] = _at_least(support["conditional_direction_auc"], 0.5, strict=True)
        for group in ("seen", "label_held_out"):
            if group not in baseline.get("symbol_groups", {}) or group not in candidate.get("symbol_groups", {}):
                checks[f"{group}_support"] = False
                continue
            a, b = baseline["symbol_groups"][group], candidate["symbol_groups"][group]
            checks[f"{group}_magnitude_power"] = _at_least(b["big_ap"], b["big_base"], strict=True)
            checks[f"{group}_direction_power"] = _at_least(b["conditional_direction_auc"], 0.5, strict=True)
            checks[f"{group}_magnitude_noninferiority"] = _at_least(b["big_ap"], a["big_ap"])
            checks[f"{group}_direction_noninferiority"] = _at_least(b["conditional_direction_auc"], a["conditional_direction_auc"])
    return {
        "method": "claim-conforming-zero-margin-prototype-v1",
        "claim": claim,
        "stage": stage,
        "supported": all(checks.values()),
        "checks": checks,
        "paired_daily_intervals": intervals,
        "pristine_oos": stage == "sealed_oos",
        "qualification": "Method study prototype; not an adopted FE kernel or evidence of methodology superiority.",
    }
