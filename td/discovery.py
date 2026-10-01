"""Cheap target screening inside an outer training window, never feature search."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ...io import digest, read_yaml, write_yaml
from .evaluation import date_view, economics
from .targets import column, project, truth_columns


def feasible_targets(frame, targets, splits, cfg, output):
    """Reject unsupported training hypotheses without fitting or looking at X.

    Movement diagnostics are not realizable profits. Only an absence of enough
    even hindsight-profitable common-horizon trades rejects economic support;
    a target's barrier or label magnitude need not itself exceed trading costs.
    """
    columns = ["sample_id", "day", "symbol", "SampleTime", *truth_columns(cfg["truth"])]
    truth = frame[columns]
    training = [date_view(truth, split["train"]) for split in splits]
    g, cost = cfg["gates"], cfg["economics"]["round_trip_ticks"]
    opportunities = []
    for rows in training:
        returns = rows[column("return", cfg["economics"]["horizon_seconds"])].to_numpy()
        observed = np.isfinite(returns)
        possible, _ = economics(rows, np.where(observed & (np.abs(returns) > cost), np.sign(returns), 0), cfg["economics"])
        supported = possible["triggers"] >= g["min_triggers"] and possible["coverage"] >= g["min_coverage"]
        opportunities.append(
            {
                "rows": len(rows),
                "observed_returns": int(observed.sum()),
                "unknown_returns": int((~observed).sum()),
                "absolute_return_quantiles_ticks": np.quantile(np.abs(returns[observed]), [0.5, 0.9, 0.99]).tolist() if observed.any() else [],
                "hindsight_triggers": possible["triggers"],
                "hindsight_coverage": possible["coverage"],
                "supported": True if supported else None if (~observed).any() else False,
            }
        )
    records, retained = [], []
    for target in targets:
        reasons, stats = [], []
        for i, rows in enumerate(training):
            y = project(rows, target, cfg["truth"])
            heads = []
            if len(rows) < g["min_rows"]:
                reasons.append(f"fold_{i}:insufficient_training_rows")
            for values in y.T:
                observed = np.isfinite(values)
                values = values[observed]
                if len(values) < g["min_rows"]:
                    reasons.append(f"fold_{i}:insufficient_observed_training_rows")
                head = {
                    "observed_rows": len(values),
                    "unknown_rows": int((~observed).sum()),
                    "std": float(np.std(values)) if len(values) else None,
                    "quantiles": np.quantile(values, [0.01, 0.5, 0.99]).tolist() if len(values) else [],
                }
                if target["task"] == "binary_pair":
                    counts = [int((values == c).sum()) for c in [0, 1]]
                    head["class_counts"] = counts
                    if min(counts) < g["min_class_count"]:
                        reasons.append(f"fold_{i}:insufficient_class_support")
                elif not len(values) or np.var(values) == 0:
                    reasons.append(f"fold_{i}:constant_target")
                heads.append(head)
            stats.append({"train_days": splits[i]["train"], "heads": heads})
        if any(row["supported"] is None for row in opportunities):
            reasons.append("unresolved_common_horizon_opportunities")
        if any(row["supported"] is False for row in opportunities):
            reasons.append("insufficient_common_horizon_opportunities")
        records.append({"target": target, "eligible": not reasons, "reasons": sorted(set(reasons)), "training": stats})
        if not reasons:
            retained.append(target)
    insufficient = any("unresolved" in reason or "insufficient_observed" in reason for record in records for reason in record["reasons"])
    write_yaml(
        output / "feasibility.yaml", {"training_only": True, "fits": 0, "insufficient_observed_support": insufficient, "opportunities": opportunities, "candidates": records}
    )
    return retained


def sample_population(frame, cfg, output):
    """Hash only sample identity/seed; share this fixed panel across all targets."""
    d, seed = cfg["discovery"], cfg["proxy"]["seed"]
    hashes = np.array([int.from_bytes(hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest(), "big") for key in frame.sample_id], dtype=np.uint64)
    selected, partitions = [], []
    for (day, symbol), indices in frame.groupby(["day", "symbol"], sort=True).indices.items():
        count = min(d["max_rows_per_partition"], max(1, math.ceil(len(indices) * d["sample_fraction"])))
        order = np.lexsort((frame.sample_id.iloc[indices].to_numpy(), hashes[indices]))
        selected.extend(indices[order[:count]].tolist())
        partitions.append({"day": str(day), "symbol": str(symbol), "full_rows": len(indices), "screen_rows": count})
    sampled = frame.iloc[sorted(selected)].reset_index(drop=True)
    sampled[["sample_id", "day", "symbol", "SampleTime"]].to_parquet(output / "screen-population.parquet", index=False)
    write_yaml(
        output / "screen-population.yaml",
        {
            "method": "blake2b64(seed:sample_id), lowest hashes per day/symbol, original chronological order",
            "sample_fraction": d["sample_fraction"],
            "max_rows_per_partition": d["max_rows_per_partition"],
            "seed": seed,
            "sample_digest": digest(sampled.sample_id.tolist()),
            "feature_manifest_digest": digest(cfg["data"]["features"]),
            "partitions": partitions,
        },
    )
    return sampled


def promote(leaderboard, cfg):
    """Use common economics, then within-task ranks, never cross-task raw skill.

    Screening nominations are hypotheses, not economic qualification. A zero
    policy coverage can still nominate a predictable target for confirmation.
    """
    eligible = [r for r in leaderboard if r["status"] == "measured" and r["skill"] > 0 and r["predictive_stability"] >= cfg["gates"]["positive_fold_fraction"]]
    ranked = []
    for task in sorted({r["target"]["task"] for r in eligible}):
        ordered = sorted(
            [r for r in eligible if r["target"]["task"] == task],
            key=lambda r: (-r["economic_value"], -r["predictive_stability"], -r["skill"], -r["coverage"], r["target"]["complexity"], r["target"]["id"]),
        )
        ranked.extend((rank, row) for rank, row in enumerate(ordered))
    ranked.sort(key=lambda item: (-item[1]["economic_value"], item[0], item[1]["target"]["complexity"], item[1]["target"]["id"]))
    return [row["target"] for _, row in ranked[: cfg["discovery"]["max_promoted"]]]


def staged_search(frame, features, targets, splits, cfg, output, search, *, predictions):
    output.mkdir(parents=True, exist_ok=True)
    feasible = feasible_targets(frame, targets, splits, cfg, output)
    feasibility = read_yaml(output / "feasibility.yaml")
    print(f"TD discovery {output}: {len(feasible)}/{len(targets)} feasible candidates", flush=True)
    screening, confirmed, promoted = [], [], []
    if feasible:
        sampled = sample_population(frame, cfg, output)
        proxy = {**cfg["proxy"], **{k: cfg["discovery"][k] for k in ["iterations", "depth"]}}
        screening, _ = search(sampled, features, feasible, splits, {**cfg, "proxy": proxy}, output / "screen", predictions=predictions)
        del sampled
        promoted = promote(screening, cfg)
        print(f"TD discovery {output}: {len(promoted)} candidates promoted to full confirmation", flush=True)
        if promoted:
            confirmed, _ = search(frame, features, promoted, splits, cfg, output / "confirm", predictions=predictions)
    result = {
        "candidate_count": len(targets),
        "feasible_count": len(feasible),
        "insufficient_observed_support": not feasible and feasibility["insufficient_observed_support"],
        "screened_count": len(screening),
        "promoted": promoted,
        "promotion_basis": "positive reproducible skill; common economics then within-task rank; nominations are not qualified targets",
        "screen_leaderboard": screening,
        "confirmation_leaderboard": confirmed,
        "screen_fits": sum(r["fits"] for r in screening),
        "confirmation_fits": sum(r["fits"] for r in confirmed),
        "screen_trees": sum(r["trees"] for r in screening),
        "confirmation_trees": sum(r["trees"] for r in confirmed),
    }
    write_yaml(output / "discovery.yaml", result)
    return confirmed, result
