"""Readable summaries of a scored round and TRAIN-model family importance."""

from __future__ import annotations

import json
from collections import defaultdict

from ...io import read_yaml
from . import profile as P

KEYS = ("big_rate", "side_ap", "big_ap", "big_auc_within", "dir_auc", "dir_auc_symbol")


def summary(profile, round_name) -> str:
    report = read_yaml(P.work(profile) / "fits" / round_name / "report.yaml")
    lines = [f"# {round_name} ({report['stage']}), baseline {report['baseline']}"]
    for h, block in report["horizons"].items():
        lines.append(f"\n## h={h}\n")
        groups = list(next(iter(block["arms"].values())).keys())
        for g in groups:
            lines.append(f"{g}: " + " | ".join(KEYS))
            for arm, m in block["arms"].items():
                lines.append(f"  {arm:16s} " + " ".join(f"{m[g][k]:.4f}" for k in KEYS))
        for name, c in block["contrasts"].items():
            b = c["bootstrap"]
            parts = " ".join(f"{k} {b[k]['diff']:+.4f}[{b[k]['lower']:+.4f},{b[k]['upper']:+.4f}]" for k in b)
            failed = [k for k, v in c["checks"].items() if not v]
            lines.append(f"  {name} [{c['claim']}] supported={c['supported']} win={c['daily_side_ap_win_rate']:.2f} {parts} failed={failed}")
    for name, o in report.get("overall", {}).items():
        lines.append(f"overall {name}: supported={o['supported']} {o.get('checks', o.get('reason'))}")
    return "\n".join(lines)


def family_importance(profile, round_name, arm, stage, h) -> list[tuple[str, float]]:
    """Mean over seeds of PredictionValuesChange importance summed by set:family."""
    spec = profile["rounds"][round_name]["arms"][arm]
    seeds = spec.get("seeds") or profile["training"]["seeds"]
    total: dict[str, float] = defaultdict(float)
    for seed in seeds:
        path = P.work(profile) / "fits" / round_name / arm / f"{stage}-h{h}-s{seed}-importance.json"
        for column, value in json.loads(path.read_text()):
            set_name, _, name = column.partition(":")
            total[f"{set_name}:{name.split('.')[0]}"] += value / len(seeds)
    return sorted(total.items(), key=lambda t: -t[1])
