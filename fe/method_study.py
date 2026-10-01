"""M1 analysis: single-seed versus seed-ensemble null spread under the judge.

Uses the persisted per-seed probabilities of the B and B_null arms (six seeds
in total). For every split of the six seeds into two disjoint triples the null
contrast is recomputed with the frozen bootstrap, which gives ten correlated
ensemble null contrasts; single-seed pairs give the single-seed null spread.
No new fit is made.
"""

from __future__ import annotations

import itertools

import numpy as np

from ...io import write_yaml
from . import dataset, judge
from . import profile as P


def run(profile, round_name="m1", horizons=(60, 300)):
    rule = profile["judge"]
    threshold = float(profile["labels"]["threshold_ticks"])
    cohort = dataset.load_cohort(profile)
    all_h = [int(h) for h in cohort["horizons"]]
    root = P.work(profile) / "fits" / round_name
    out = {}
    for h in horizons:
        a = np.load(root / "B" / f"dev-h{h}.npz")
        b = np.load(root / "B_null" / f"dev-h{h}.npz")
        if not np.array_equal(a["index"], b["index"]):
            raise ValueError("null arms were scored on different rows")
        idx = a["index"]
        seeds = np.concatenate([a["p_seeds"], b["p_seeds"]]).astype(np.float64)
        y = cohort["y"][idx, all_h.index(h)]
        day, sym = cohort["day"][idx], cohort["symbol"][idx]
        single = [judge.metrics(y, s, day, sym, threshold) for s in seeds]
        spread = {k: float(np.ptp([m[k] for m in single])) for k in ("side_ap", "big_ap", "dir_auc", "big_auc_within")}
        splits = []
        for triple in itertools.combinations(range(6), 3):
            if 0 not in triple:
                continue
            other = tuple(i for i in range(6) if i not in triple)
            pa, pb = seeds[list(triple)].mean(axis=0), seeds[list(other)].mean(axis=0)
            boot = judge.paired(y, pa, pb, day, threshold, draws=rule["bootstrap_draws"], block_days=rule["block_days"], seed=rule["bootstrap_seed"], tail=rule["tail"])
            ma, mb = judge.metrics(y, pa, day, sym, threshold), judge.metrics(y, pb, day, sym, threshold)
            decisions = {claim: judge.decide(rule, ma, mb, boot, claim=claim)["supported"] for claim in ("joint", "magnitude", "direction")}
            splits.append({"triple": list(triple), "diff": {k: boot[k]["diff"] for k in boot}, "lower": {k: boot[k]["lower"] for k in boot}, "supported": decisions})
        ensemble_abs = {k: float(np.mean([abs(s["diff"][k]) for s in splits])) for k in judge.POOLED}
        pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        single_abs = {k: float(np.mean([abs(single[i][k] - single[j][k]) for i, j in pairs])) for k in judge.POOLED}
        out[str(h)] = {
            "single_seed_range": spread,
            "mean_abs_null_difference_single_seed": single_abs,
            "mean_abs_null_difference_three_seed": ensemble_abs,
            "supported_null_splits": {c: int(sum(s["supported"][c] for s in splits)) for c in ("joint", "magnitude", "direction")},
            "splits": splits,
        }
        print(h, "single", single_abs, "ensemble", ensemble_abs, "supported", out[str(h)]["supported_null_splits"], flush=True)
    write_yaml(root / "method-study.yaml", out)
    return out
