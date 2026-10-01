"""Fixed magnitude, conditional direction and joint endpoint evidence."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ...io import ContractError
from ...statistics import block_interval
from ..signal.metrics import auc, average_precision
from ..signal.metrics import evaluate as joint_evaluate
from .data import native_classes


def nonoverlap(rows, horizon):
    chosen = np.zeros(len(rows), dtype=bool)
    for indices in rows.groupby(["day", "symbol"], sort=True).indices.values():
        order = indices[np.argsort(rows.SampleTime.to_numpy()[indices], kind="stable")]
        last = None
        for i in order:
            now = int(rows.SampleTime.iat[i])
            if last is None or now >= last + horizon * 1_000_000:
                chosen[i], last = True, now
    return chosen


def evaluate(rows, scores, horizon):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(rows), 2) or not np.isfinite(scores).all() or (scores < 0).any() or (scores.sum(axis=1) > 1 + 1e-12).any():
        raise ContractError("Scoring requires finite joint native up/down probabilities")
    labels, known = native_classes(rows, horizon)
    rows = rows.loc[known].reset_index(drop=True)
    scores, labels = scores[known], labels[known]
    if not len(rows):
        raise ContractError("No mature native endpoints")
    move = rows[f"mid_return_ticks[{horizon}s]"].to_numpy(dtype=float)
    big, up = labels != 0, labels == 1
    pbig = scores.sum(axis=1)
    if (pbig <= 0).any():
        raise ContractError("Conditional direction is undefined at zero large-move probability; no epsilon repair")
    direction = scores[:, 0] / pbig
    out = joint_evaluate(move, scores[:, 0], scores[:, 1], rows.day.to_numpy(), rows.symbol.to_numpy(), 5)
    out["big_ap"] = average_precision(big, pbig)
    out["big_base"] = float(big.mean())
    out["conditional_direction_auc"] = auc(up[big], direction[big])
    out["absolute_move_spearman"] = float(spearmanr(np.abs(move), pbig).statistic) if np.unique(move).size > 1 and np.unique(pbig).size > 1 else np.nan
    sd_big, sd_direction = [], []
    for indices in rows.groupby(["day", "symbol"], sort=True).indices.values():
        if 0 < big[indices].sum() < len(indices):
            sd_big.append(auc(big[indices], pbig[indices]))
        events = indices[big[indices]]
        if 0 < up[events].sum() < len(events):
            sd_direction.append(auc(up[events], direction[events]))
    out["equal_symbol_day_big_auc"] = float(np.mean(sd_big)) if sd_big else np.nan
    out["equal_symbol_day_direction_auc"] = float(np.mean(sd_direction)) if sd_direction else np.nan
    sparse = nonoverlap(rows, horizon)
    out["nonoverlap"] = {
        "rows": int(sparse.sum()),
        "up": int((up & sparse).sum()),
        "down": int(((labels == 2) & sparse).sum()),
        "up_days": int(rows.loc[up & sparse, "day"].nunique()),
        "down_days": int(rows.loc[(labels == 2) & sparse, "day"].nunique()),
        "big_ap": average_precision(big[sparse], pbig[sparse]),
        "side_ap": float(np.nanmean([average_precision(up[sparse], scores[sparse, 0]), average_precision(labels[sparse] == 2, scores[sparse, 1])])),
        "conditional_direction_auc": auc(up[sparse & big], direction[sparse & big]),
    }
    return out


def daily(rows, scores, horizon):
    records = []
    for day, indices in rows.groupby("day", sort=True).indices.items():
        result = evaluate(rows.iloc[indices].reset_index(drop=True), scores[indices], horizon)
        records.append({"day": str(day), **{k: result[k] for k in ("big_ap", "mean_ap", "conditional_direction_auc")}})
    return pd.DataFrame(records)


def decision(profile, baseline, candidate, baseline_daily=None, candidate_daily=None, *, stage="calibration"):
    cfg = profile["acceptance"]
    checks = {
        "magnitude_gain": candidate["big_ap"] >= baseline["big_ap"] * (1 + cfg["relative_ap_gain"]),
        "joint_gain": candidate["mean_ap"] >= baseline["mean_ap"] * (1 + cfg["relative_ap_gain"]),
        "direction_gain": candidate["conditional_direction_auc"] >= baseline["conditional_direction_auc"] + cfg["direction_auc_gain"],
        "direction_power": candidate["conditional_direction_auc"] >= cfg["minimum_direction_auc"],
        "within_symbol_day_magnitude": candidate["equal_symbol_day_big_auc"] >= baseline["equal_symbol_day_big_auc"],
        "net_directional_tail": candidate["mean_p@0.01"] - candidate["mean_wrong@0.01"] >= baseline["mean_p@0.01"] - baseline["mean_wrong@0.01"],
        "nonoverlap_ranking": candidate["nonoverlap"]["side_ap"] >= baseline["nonoverlap"]["side_ap"],
    }
    intervals = {}
    if stage != "calibration":
        if baseline_daily is None or candidate_daily is None or not baseline_daily.day.equals(candidate_daily.day):
            raise ContractError("Paired evidence requires the identical full chronological day axis")
        for metric in ("big_ap", "mean_ap", "conditional_direction_auc"):
            delta = candidate_daily[metric].to_numpy() - baseline_daily[metric].to_numpy()
            if not np.isfinite(delta).all():
                checks[f"positive_daily_{metric}_bound"] = False
                intervals[metric] = {"resolution_sufficient": False, "reason": "undefined_daily_metric_not_dropped"}
            else:
                intervals[metric] = block_interval(delta, cfg["block_days"], cfg["bootstrap_seed"], draws=cfg["bootstrap_draws"], tail_probability=cfg["tail_probability"])
                checks[f"positive_daily_{metric}_bound"] = intervals[metric]["resolution_sufficient"] and intervals[metric]["lower"] > 0
        support = candidate["nonoverlap"]
        checks["nonoverlap_support"] = all(support[side] >= cfg["minimum_side_events"] and support[side + "_days"] >= cfg["minimum_event_days"] for side in ("up", "down"))
    return {"stage": stage, "supported": all(checks.values()), "checks": checks, "paired_daily_intervals": intervals, "pristine_oos": stage == "sealed_oos"}
