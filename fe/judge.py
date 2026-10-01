"""Separate magnitude, direction and joint evidence; paired day-block bootstrap.

Scores are the seed-ensemble MultiClass probabilities (down, none, up).
* magnitude: b = P(up) + P(down) ranks |move| >= threshold (big).
  - big_ap (pooled), big_auc_within (mean AUC within symbol-day, timing only)
* direction: q = P(up) / b ranks up versus down among big moves only.
  - dir_auc (pooled over big events), dir_auc_symbol (mean over symbols)
* joint: side_ap = mean of AP(up event, P(up)) and AP(down event, P(down)).

A candidate is compared with the baseline on identical rows. Paired inference
resamples whole days in moving blocks and recomputes pooled metrics for both
arms on the same resample; bounds are one-sided at the frozen tail.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import rankdata

from ..signal.metrics import average_precision


def auc(event, score):
    event = np.asarray(event, bool)
    n1 = int(event.sum())
    n0 = len(event) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _parts(y, p, threshold):
    up, down = y >= threshold, y <= -threshold
    big = up | down
    b = p[:, 2] + p[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(b > 0, p[:, 2] / b, 0.5)
    return up, down, big, b, q


def metrics(y, p, day, symbol, threshold) -> dict:
    up, down, big, b, q = _parts(y, p, threshold)
    out = {
        "rows": len(y),
        "big_rate": float(big.mean()),
        "big_events": int(big.sum()),
        "big_ap": average_precision(big, b),
        "big_auc": auc(big, b),
        "dir_auc": auc(up[big], q[big]),
        "side_ap": float(np.nanmean([average_precision(up, p[:, 2]), average_precision(down, p[:, 0])])),
    }
    within, weights = [], []
    group = np.char.add(day.astype(str), symbol.astype(str))
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.r_[True, group[order][1:] != group[order][:-1], True])
    for a, z in itertools.pairwise(bounds):
        idx = order[a:z]
        e = big[idx]
        if 0 < e.sum() < len(e):
            within.append(auc(e, b[idx]))
            weights.append(e.sum())
    out["big_auc_within"] = float(np.average(within, weights=weights)) if within else np.nan
    per_symbol = []
    for s in np.unique(symbol):
        m = (symbol == s) & big
        if 0 < up[m].sum() < m.sum():
            per_symbol.append(auc(up[m], q[m]))
    out["dir_auc_symbol"] = float(np.mean(per_symbol)) if per_symbol else np.nan
    return out


POOLED = ("side_ap", "big_ap", "dir_auc")


def _pooled(y, p, threshold, name):
    up, down, big, b, q = _parts(y, p, threshold)
    if name == "side_ap":
        return float(np.nanmean([average_precision(up, p[:, 2]), average_precision(down, p[:, 0])]))
    if name == "big_ap":
        return average_precision(big, b)
    if name == "dir_auc":
        return auc(up[big], q[big])
    raise ValueError(name)


class _SortedAP:
    """AP of a fixed score/event pair under row weights, sorting once.

    A day resample with multiplicities equals weighting each row by its day's
    multiplicity: duplicated rows share a score, so they join existing tie
    groups and the tie-group-end cumulative counts are the weighted sums.
    """

    def __init__(self, event, score, row_day):
        order = np.argsort(-np.asarray(score), kind="stable")
        s = np.asarray(score)[order]
        self.event = np.asarray(event, bool)[order].astype(np.float64)
        self.day = row_day[order]
        self.ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]

    def __call__(self, day_weight):
        w = day_weight[self.day]
        tp = np.cumsum(w * self.event)[self.ends]
        if tp[-1] <= 0:
            return np.nan
        seen = np.cumsum(w)[self.ends]
        # Groups with no weighted row yet add no true positive; skip their 0/0.
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        return float(np.dot(np.diff(np.r_[0.0, tp]), precision) / tp[-1])


class _SortedAUC:
    """Mann-Whitney AUC (average ranks for ties) under row weights, sorting once."""

    def __init__(self, event, score, row_day):
        order = np.argsort(np.asarray(score), kind="stable")
        s = np.asarray(score)[order]
        self.event = np.asarray(event, bool)[order]
        self.day = row_day[order]
        starts = np.r_[0, np.flatnonzero(s[:-1] != s[1:]) + 1]
        self.group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s)]))
        self.groups = len(starts)

    def __call__(self, day_weight):
        w = day_weight[self.day]
        pos = np.bincount(self.group, weights=w * self.event, minlength=self.groups)
        neg = np.bincount(self.group, weights=w * ~self.event, minlength=self.groups)
        total_pos, total_neg = pos.sum(), neg.sum()
        if total_pos <= 0 or total_neg <= 0:
            return np.nan
        below = np.cumsum(neg) - neg
        return float(np.dot(pos, below + 0.5 * neg) / (total_pos * total_neg))


def _evaluators(y, p, threshold, row_day):
    up, down, big, b, q = _parts(y, p, threshold)
    ap_up, ap_down = _SortedAP(up, p[:, 2], row_day), _SortedAP(down, p[:, 0], row_day)
    big_ap = _SortedAP(big, b, row_day)
    direction = _SortedAUC(up[big], q[big], row_day[big])
    return {
        "side_ap": lambda w: float(np.nanmean([ap_up(w), ap_down(w)])),
        "big_ap": big_ap,
        "dir_auc": direction,
    }


def paired(y, pa, pb, day, threshold, *, draws, block_days, seed, tail) -> dict:
    """Moving-block day bootstrap of metric(b) - metric(a) for the pooled metrics.

    Each resample draws whole days in moving blocks; both arms are scored on the
    same resample, expressed as day multiplicities weighting the rows.
    """
    days, row_day = np.unique(day, return_inverse=True)
    n = len(days)
    size = min(block_days, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - size + 1, (draws, (n + size - 1) // size))
    picks = (starts[..., None] + np.arange(size)).reshape(draws, -1)[:, :n]
    ea, eb = _evaluators(y, pa, threshold, row_day), _evaluators(y, pb, threshold, row_day)
    ones = np.ones(n)
    point = {name: eb[name](ones) - ea[name](ones) for name in POOLED}
    diffs = {name: np.empty(draws) for name in POOLED}
    for i, pick in enumerate(picks):
        weight = np.bincount(pick, minlength=n).astype(np.float64)
        for name in POOLED:
            diffs[name][i] = eb[name](weight) - ea[name](weight)
    out = {}
    for name in POOLED:
        d = diffs[name]
        out[name] = {"diff": float(point[name]), "lower": float(np.quantile(d, tail)), "upper": float(np.quantile(d, 1 - tail)), "p_le_0": float((d <= 0).mean())}
    return out


def paired_reference(y, pa, pb, day, threshold, *, draws, block_days, seed, tail) -> dict:
    """Direct concatenation of resampled days; the definition ``paired`` must reproduce."""
    days = np.unique(day)
    by_day = [np.flatnonzero(day == d) for d in days]
    n = len(days)
    size = min(block_days, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - size + 1, (draws, (n + size - 1) // size))
    picks = (starts[..., None] + np.arange(size)).reshape(draws, -1)[:, :n]
    point = {name: _pooled(y, pb, threshold, name) - _pooled(y, pa, threshold, name) for name in POOLED}
    diffs = {name: np.empty(draws) for name in POOLED}
    for i, pick in enumerate(picks):
        idx = np.concatenate([by_day[j] for j in pick])
        for name in POOLED:
            diffs[name][i] = _pooled(y[idx], pb[idx], threshold, name) - _pooled(y[idx], pa[idx], threshold, name)
    out = {}
    for name in POOLED:
        d = diffs[name]
        out[name] = {"diff": float(point[name]), "lower": float(np.quantile(d, tail)), "upper": float(np.quantile(d, 1 - tail)), "p_le_0": float((d <= 0).mean())}
    return out


def decide(rule: dict, base: dict, cand: dict, boot: dict, groups: dict | None = None, claim="joint") -> dict:
    """Frozen acceptance: claim in {joint, magnitude, direction}."""
    checks = {}
    # Training-realization noise the day bootstrap cannot see: a claimed gain's
    # point estimate must exceed the frozen seed-ensemble noise floor.
    floor = rule.get("noise_floor", {})
    target = {"joint": "side_ap", "magnitude": "big_ap", "direction": "dir_auc"}[claim]
    checks["above_noise_floor"] = boot[target]["diff"] > floor.get(target, 0.0)
    if claim == "joint":
        checks["joint_gain"] = boot["side_ap"]["lower"] > 0
    if claim in ("joint", "magnitude"):
        checks["magnitude_noninferior"] = boot["big_ap"]["lower"] > -rule["magnitude_margin"] * base["big_ap"]
    if claim == "magnitude":
        checks["magnitude_gain"] = boot["big_ap"]["lower"] > 0
        checks["timing_noninferior"] = cand["big_auc_within"] >= base["big_auc_within"] - rule["timing_margin"]
    if claim in ("joint", "direction"):
        checks["direction_noninferior"] = boot["dir_auc"]["lower"] > -rule["direction_margin"]
    if claim == "direction":
        checks["direction_gain"] = boot["dir_auc"]["lower"] > 0
    if groups:
        for g, (gb, gc) in groups.items():
            key = {"joint": "side_ap", "magnitude": "big_ap", "direction": "dir_auc"}[claim]
            checks[f"group_{g}_not_worse"] = gc[key] >= gb[key]
    return {"claim": claim, "supported": all(checks.values()), "checks": checks}


def daily_wins(y, pa, pb, day, threshold) -> float:
    """Share of days whose side-mean AP is higher for b than for a (days with events on both sides)."""
    wins, total = 0, 0
    for d in np.unique(day):
        m = day == d
        up, down = y[m] >= threshold, y[m] <= -threshold
        if not up.any() or not down.any():
            continue
        a = np.mean([average_precision(up, pa[m, 2]), average_precision(down, pa[m, 0])])
        b = np.mean([average_precision(up, pb[m, 2]), average_precision(down, pb[m, 0])])
        wins += b > a
        total += 1
    return wins / total if total else float("nan")


def overall(rule: dict, per_horizon: dict, primary: int) -> dict:
    """A contrast is supported when its primary horizon is supported and no other
    scored horizon shows a significant joint loss (side_ap lower bound below
    -horizon_margin of the baseline's side_ap)."""
    if str(primary) not in per_horizon:
        return {"supported": False, "reason": "primary horizon not scored"}
    checks = {"primary": per_horizon[str(primary)]["supported"]}
    for h, c in per_horizon.items():
        if h != str(primary):
            checks[f"h{h}_joint_not_worse"] = c["bootstrap"]["side_ap"]["lower"] > -rule["horizon_margin"] * c["baseline_side_ap"]
    return {"supported": all(checks.values()), "checks": checks}
