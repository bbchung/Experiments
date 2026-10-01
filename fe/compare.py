"""Score a round: every arm on identical rows, candidates against the round baseline.

A round (profile ``rounds.<name>``) names a stage, horizons, a baseline arm and
candidate arms with their claims (joint / magnitude / direction). Fits come
from ``AstraResearch.Experiments.fe.train``; evaluation reads the persisted seed-ensemble
probabilities and applies the frozen judge. Symbol groups (development and,
at the final stage, holdout) are reported and guarded separately.
"""

from __future__ import annotations

import csv
import json

import numpy as np

from ...io import ContractError, write_yaml
from . import dataset, judge, train
from . import profile as P


def fit_round(profile, round_name, horizons=None, arms=None):
    spec = profile["rounds"][round_name]
    names = [spec["baseline"], *[a for a in spec["arms"] if a != spec["baseline"]]]
    for name in names:
        if arms and name not in arms:
            continue
        arm = spec["arms"][name]
        train.fit_arm(profile, round_name, name, arm, spec["stage"], horizons or spec["horizons"], seeds=arm.get("seeds"))


def load_scores(profile, round_name, arm, stage, h):
    path = P.work(profile) / "fits" / round_name / arm / f"{stage}-h{h}.npz"
    data = np.load(path)
    return data["index"], data["p"].astype(np.float64)


def score_round(profile, round_name):
    spec = profile["rounds"][round_name]
    rule = profile["judge"]
    threshold = float(profile["labels"]["threshold_ticks"])
    cohort = dataset.load_cohort(profile)
    all_h = [int(h) for h in cohort["horizons"]]
    stage, base = spec["stage"], spec["baseline"]
    groups = {"development": set(P.symbols(profile, "development"))}
    if stage == "final":
        groups["holdout"] = set(P.symbols(profile, "holdout"))
    report = {"round": round_name, "stage": stage, "baseline": base, "judge_identity": spec.get("_judge_identity"), "horizons": {}}
    rows_csv = []
    for h in spec["horizons"]:
        k = all_h.index(int(h))
        idx_b, _ = load_scores(profile, round_name, base, stage, h)
        y = cohort["y"][idx_b, k]
        day, sym = cohort["day"][idx_b], cohort["symbol"][idx_b]
        per_arm = {}
        for arm in spec["arms"]:
            idx, p = load_scores(profile, round_name, arm, stage, h)
            if not np.array_equal(idx, idx_b):
                raise ContractError(f"{arm} h={h} was not scored on the baseline's rows")
            m = {"all": judge.metrics(y, p, day, sym, threshold)}
            for g, members in groups.items():
                sel = np.isin(sym, list(members))
                m[g] = judge.metrics(y[sel], p[sel], day[sel], sym[sel], threshold)
            per_arm[arm] = (p, m)
            rows_csv.append({"horizon": h, "arm": arm, **{f"{g}:{key}": v for g, mm in m.items() for key, v in mm.items()}})
        out = {"arms": {a: v[1] for a, v in per_arm.items()}, "contrasts": {}}
        boots = {}
        for contrast in spec.get("contrasts", []):
            cand, ref, claim = contrast["candidate"], contrast.get("baseline", base), contrast["claim"]
            # Claims of the same pair share one resampling.
            if (cand, ref) not in boots:
                boots[cand, ref] = judge.paired(
                    y,
                    per_arm[ref][0],
                    per_arm[cand][0],
                    day,
                    threshold,
                    draws=rule["bootstrap_draws"],
                    block_days=rule["block_days"],
                    seed=rule["bootstrap_seed"],
                    tail=rule["tail"],
                )
            boot = boots[cand, ref]
            group_pairs = {g: (per_arm[ref][1][g], per_arm[cand][1][g]) for g in groups} if len(groups) > 1 else None
            decision = judge.decide(rule, per_arm[ref][1]["all"], per_arm[cand][1]["all"], boot, group_pairs, claim)
            daily = judge.daily_wins(y, per_arm[ref][0], per_arm[cand][0], day, threshold)
            out["contrasts"][f"{cand}-vs-{ref}:{claim}"] = {"bootstrap": boot, "daily_side_ap_win_rate": daily, "baseline_side_ap": per_arm[ref][1]["all"]["side_ap"], **decision}
            print(
                f"{round_name} h={h} {cand} vs {ref} [{claim}] supported={decision['supported']} win={daily:.2f} "
                + " ".join(f"{m}:{boot[m]['diff']:+.4f}[{boot[m]['lower']:+.4f},{boot[m]['upper']:+.4f}]" for m in boot),
                flush=True,
            )
        report["horizons"][str(h)] = out
    report["overall"] = {}
    for name in {n for block in report["horizons"].values() for n in block["contrasts"]}:
        per = {h: block["contrasts"][name] for h, block in report["horizons"].items() if name in block["contrasts"]}
        report["overall"][name] = judge.overall(rule, per, int(rule["primary_horizon"]))
        print(f"{round_name} overall {name}: supported={report['overall'][name]['supported']}", flush=True)
    target = P.work(profile) / "fits" / round_name
    write_yaml(target / "report.yaml", json.loads(json.dumps(report, default=float)))
    with (target / "metrics.csv").open("w", newline="") as stream:
        fields = sorted({k for r in rows_csv for k in r}, key=lambda c: (c not in ("horizon", "arm"), c))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_csv)
    return report
