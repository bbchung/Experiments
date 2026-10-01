"""Horizon census of native features on the dev_train role only (never dev_val/oos).

For every numeric column and horizon: magnitude information as the mean
within-symbol-day AUC of the column for |move| >= threshold (sign-free: the
larger of AUC and 1-AUC), and direction information as the AUC of the column
for up versus down among large moves (pooled, and the mean within symbol).
This describes univariate information and its decay with horizon; it is a
diagnostic for hypothesis generation, not a selection rule for any frozen arm.
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import rankdata

from . import dataset
from . import profile as P
from .train import role_rows


def _auc_ranked(ranks, event):
    n1 = int(event.sum())
    n0 = len(event) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(profile, set_name, sample=250000, seed=0):
    cohort = dataset.load_cohort(profile)
    cache = dataset.open_cache(profile, set_name)
    threshold = float(profile["labels"]["threshold_ticks"])
    rows = role_rows(profile, cohort, "dev_train", "development")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(rows, size=min(sample, len(rows)), replace=False))
    horizons = [int(h) for h in cohort["horizons"]]
    X = np.asarray(cache["X"][rows], dtype=np.float64)
    sym = cohort["symbol"][rows]
    out = []
    # Symbol identifiability: share of a column's finite variance explained by symbol
    # means (eta squared). Near one, a column mostly encodes which symbol a row is.
    symbols = np.unique(sym)
    eta = {}
    for j, name in enumerate(cache["numeric"]):
        x = X[:, j]
        fin = np.isfinite(x)
        if fin.sum() < 1000 or np.nanvar(x[fin]) == 0:
            eta[name] = np.nan
            continue
        grand = x[fin].mean()
        between = sum(((x[fin & (sym == s)].mean() - grand) ** 2) * (fin & (sym == s)).sum() for s in symbols if (fin & (sym == s)).any())
        eta[name] = float(between / (((x[fin] - grand) ** 2).sum()))
    for k, h in enumerate(horizons):
        ok = cohort["valid"][rows, k]
        y = cohort["y"][rows, k]
        big = ok & (np.abs(y) >= threshold)
        for j, name in enumerate(cache["numeric"]):
            x = X[:, j]
            fin = np.isfinite(x) & ok
            rec = {"horizon": h, "column": name, "family": name.split(".")[0], "coverage": float(fin.mean()), "symbol_eta2": eta[name]}
            if fin.sum() < 1000:
                out.append(rec)
                continue
            # Magnitude: pooled AUC for big (sign-free) and mean within-symbol AUC.
            r = rankdata(x[fin])
            a = _auc_ranked(r, big[fin])
            rec["mag_auc_pooled"] = max(a, 1 - a) if np.isfinite(a) else np.nan
            within = []
            for s in np.unique(sym):
                m = fin & (sym == s)
                if 0 < big[m].sum() < m.sum():
                    within.append(_auc_ranked(rankdata(x[m]), big[m]))
            w = np.nanmean(within) if within else np.nan
            rec["mag_auc_symbol"] = max(w, 1 - w) if np.isfinite(w) else np.nan
            # Direction among large moves.
            m = fin & big
            if m.sum() >= 200:
                up = y[m] > 0
                d = _auc_ranked(rankdata(x[m]), up)
                rec["dir_auc_pooled"] = d
                per = []
                for s in np.unique(sym[m]):
                    mm = sym[m] == s
                    if 0 < up[mm].sum() < mm.sum() and mm.sum() >= 30:
                        per.append(_auc_ranked(rankdata(x[m][mm]), up[mm]))
                rec["dir_auc_symbol"] = float(np.mean(per)) if per else np.nan
            out.append(rec)
        print(f"census {set_name} h={h} done", flush=True)
    target = P.work(profile) / "census" / f"{set_name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["horizon", "column", "family", "coverage", "symbol_eta2", "mag_auc_pooled", "mag_auc_symbol", "dir_auc_pooled", "dir_auc_symbol"]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)
    return target
