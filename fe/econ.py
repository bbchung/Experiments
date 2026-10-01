"""Signal-level economics of scored fits (pure signal, mid to mid, before costs).

For each day the most confident directional rows (top q of max(P(up), P(down))
across symbols) are taken in their predicted direction; reported are the
share that became a large move that way, the share that became a large move
the other way, and the mean signed move in ticks and in bps of the symbol's
screen-window price. Diagnostic only: not part of the frozen judge.
"""

from __future__ import annotations

import numpy as np

from ...io import read_yaml
from . import dataset
from . import profile as P

LADDER = [(0.0, 0.01), (10.0, 0.05), (50.0, 0.1), (100.0, 0.5), (500.0, 1.0), (1000.0, 5.0)]


def tail(profile, round_name, quantiles=(0.01, 0.05)) -> dict:
    cohort = dataset.load_cohort(profile)
    all_h = [int(h) for h in cohort["horizons"]]
    spec = profile["rounds"][round_name]
    prices = {x["symbol"]: x["mid_price"] for x in read_yaml(P.work(profile) / "universe" / "universe.yaml")["chosen"]}
    tick_bps = {s: 1e4 * [t for b, t in LADDER if px >= b][-1] / px for s, px in prices.items()}
    out = {}
    for h in spec["horizons"]:
        for arm in spec["arms"]:
            data = np.load(P.work(profile) / "fits" / round_name / arm / f"{spec['stage']}-h{h}.npz")
            idx, p = data["index"], data["p"].astype(np.float64)
            y, sym, day = cohort["y"][idx, all_h.index(int(h))], cohort["symbol"][idx], cohort["day"][idx]
            side = np.where(p[:, 2] >= p[:, 0], 1, -1)
            conf = np.maximum(p[:, 2], p[:, 0])
            bps = np.array([tick_bps[s] for s in sym])
            rec = {}
            for q in quantiles:
                chosen = []
                for d in np.unique(day):
                    m = np.flatnonzero(day == d)
                    chosen.append(m[np.argsort(-conf[m], kind="stable")[: max(1, round(q * len(m)))]])
                sel = np.concatenate(chosen)
                signed = side[sel] * y[sel]
                rec[f"top{q:g}"] = {
                    "right_big": float((signed >= 5).mean()),
                    "wrong_big": float((signed <= -5).mean()),
                    "mean_ticks": float(signed.mean()),
                    "mean_bps": float((signed * bps[sel]).mean()),
                }
            out[f"{arm}@{h}"] = rec
            print(
                f"{round_name} h={h} {arm:10s} " + " | ".join(f"{k}: right {v['right_big']:.3f} wrong {v['wrong_big']:.3f} {v['mean_bps']:+.1f} bps" for k, v in rec.items()),
                flush=True,
            )
    return out
