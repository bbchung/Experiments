"""Big-move detection metrics and day-block paired bootstrap.

For horizon h and threshold T: big up = y >= T, big down = y <= -T. Each side has a
score (e.g. class probability). Reported per side:
  auc / ap        pooled over all held-out rows (includes between-symbol scale)
  within_sd_auc   AUC inside each symbol-day, event-weighted (timing skill only)
  p@q / wrong@q   per day, the top q fraction of rows across symbols: share that
                  became a big move in the scored direction / the opposite one
Plus direction_auc_given_big: AUC of (up - down) for up vs down among big moves.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

QS = (0.001, 0.005, 0.01, 0.05)


def auc(event, score):
    event = np.asarray(event, bool)
    n1 = int(event.sum())
    n0 = len(event) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(event, score):
    event = np.asarray(event, bool)
    if not event.any():
        return np.nan
    order = np.argsort(-score, kind="stable")
    hits, sorted_score = event[order], np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(hits) - 1]
    true_positive = np.cumsum(hits)[ends]
    precision = true_positive / (ends + 1)
    return float(np.dot(np.diff(np.r_[0, true_positive]), precision) / true_positive[-1])


def top_mean(values, score, q):
    """Fixed quota, with equal fractional weight for every boundary-score tie."""
    k = max(1, round(q * len(score)))
    cutoff = -np.partition(-score, k - 1)[k - 1]
    above, tied = score > cutoff, score == cutoff
    fraction = (k - int(above.sum())) / int(tied.sum())
    return float((np.asarray(values)[above].sum() + fraction * np.asarray(values)[tied].sum()) / k)


def side(y, score, event, wrong, day, groups):
    out = {"base": float(event.mean()), "auc": auc(event, score), "ap": average_precision(event, score)}
    out["ap_lift"] = out["ap"] / out["base"] if out["base"] > 0 else np.nan
    for q in QS:
        prec, wr, mv = [], [], []
        for d in np.unique(day):
            m = day == d
            prec.append(top_mean(event[m], score[m], q))
            wr.append(top_mean(wrong[m], score[m], q))
            mv.append(top_mean(y[m], score[m], q))
        out[f"p@{q}"], out[f"wrong@{q}"], out[f"move@{q}"] = float(np.mean(prec)), float(np.mean(wr)), float(np.mean(mv))
    vals, weights = [], []
    for g in np.unique(groups):
        m = groups == g
        e = event[m]
        if 0 < e.sum() < len(e):
            vals.append(auc(e, score[m]))
            weights.append(e.sum())
    out["within_sd_auc"] = float(np.average(vals, weights=weights)) if vals else np.nan
    return out


def evaluate(y, up, down, day, symbol, threshold):
    ok = np.isfinite(y)
    y, up, down, day, symbol = y[ok], up[ok], down[ok], day[ok], symbol[ok]
    ev_up, ev_dn = y >= threshold, y <= -threshold
    groups = np.char.add(day.astype(str), symbol.astype(str))
    res = {"rows": len(y), "up": side(y, up, ev_up, ev_dn, day, groups), "down": side(-y, down, ev_dn, ev_up, day, groups)}
    big = ev_up | ev_dn
    res["direction_auc_given_big"] = auc(ev_up[big], (up - down)[big]) if big.any() else np.nan
    for key in ("ap", "ap_lift", "auc", "within_sd_auc", "p@0.01", "wrong@0.01", "p@0.001"):
        res[f"mean_{key}"] = float(np.nanmean([res["up"][key], res["down"][key]]))
    return res


def line(tag, h, r):
    u, d = r["up"], r["down"]
    return (
        f"{tag:28s} h={h:3d} base={u['base']:.4f}/{d['base']:.4f} AUC={u['auc']:.3f}/{d['auc']:.3f} wAUC={u['within_sd_auc']:.3f}/{d['within_sd_auc']:.3f} "
        f"AP={u['ap']:.4f}/{d['ap']:.4f} lift={r['mean_ap_lift']:.2f} P@1%={u['p@0.01']:.3f}/{d['p@0.01']:.3f} "
        f"wrong@1%={u['wrong@0.01']:.3f}/{d['wrong@0.01']:.3f} P@.1%={u['p@0.001']:.3f}/{d['p@0.001']:.3f} dirAUC|big={r['direction_auc_given_big']:.3f}"
    )


def _metric(y, up, down, dd, threshold, metric):
    eu, ed = y >= threshold, y <= -threshold
    if metric == "mean_ap":
        return np.nanmean([average_precision(eu, up), average_precision(ed, down)])
    if metric == "mean_auc":
        return np.nanmean([auc(eu, up), auc(ed, down)])
    if metric == "direction_auc":
        big = eu | ed
        return auc(eu[big], (up - down)[big])
    if metric.startswith("p@"):
        q = float(metric[2:])
        vals = []
        for event, score in ((eu, up), (ed, down)):
            for d in np.unique(dd):
                m = dd == d
                vals.append(top_mean(event[m], score[m], q))
        return float(np.mean(vals))
    raise ValueError(metric)


def paired_bootstrap(y, a, b, day, threshold, metric="mean_ap", n=200, seed=0):
    """Day-block bootstrap of metric(b) - metric(a); a, b = (up, down) score pairs.

    p@q is a mean of per-day values, so those are computed once and only day
    weights are resampled; pooled AP/AUC are recomputed on every replicate.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(y)
    days = np.unique(day[ok])
    by_day = [np.flatnonzero(ok & (day == d)) for d in days]
    picks = [rng.choice(len(days), len(days), replace=True) for _ in range(n)]
    if metric.startswith("p@"):
        q = float(metric[2:])

        def per_day(scores):
            values = np.zeros(len(days))
            for i, rows in enumerate(by_day):
                for event, score in ((y[rows] >= threshold, scores[0][rows]), (y[rows] <= -threshold, scores[1][rows])):
                    values[i] += top_mean(event, score, q) / 2
            return values

        va, vb = per_day(a), per_day(b)
        diffs = np.array([vb[p].mean() - va[p].mean() for p in picks])
        point = float(vb.mean() - va.mean())
    else:
        diffs = []
        for p in picks:
            idx = np.concatenate([by_day[i] for i in p])
            zeros = np.zeros(len(idx))
            diffs.append(_metric(y[idx], b[0][idx], b[1][idx], zeros, threshold, metric) - _metric(y[idx], a[0][idx], a[1][idx], zeros, threshold, metric))
        diffs = np.asarray(diffs)
        rows = np.flatnonzero(ok)
        zeros = np.zeros(len(rows))
        point = float(_metric(y[rows], b[0][rows], b[1][rows], zeros, threshold, metric) - _metric(y[rows], a[0][rows], a[1][rows], zeros, threshold, metric))
    return {"diff": point, "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)), "p_le0": float((diffs <= 0).mean())}
