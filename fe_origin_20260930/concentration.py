"""Describe where fixed native endpoint support and ranked hits concentrate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from study_data import native_classes, nonoverlap

from AstraResearch.Experiments.signal.metrics import auc, average_precision
from AstraResearch.io import ContractError, read_yaml, write_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    args = parser.parse_args()
    root = args.root.resolve()
    records, summaries = [], []
    for panel in ("stock", "txf", "exf"):
        study = root / f"study-{panel}"
        if not (study / "completed.yaml").exists():
            raise ContractError("Concentration reporting requires completed fixed comparisons")
        for role in ("calibration", "forward"):
            projection = study / ("tune" if role == "calibration" else role)
            rows = pd.read_parquet(projection / "rows.parquet")
            classes, known = native_classes(rows, 300)
            population = known.copy()
            if role == "calibration":
                dates = read_yaml(study / "nomination.yaml")["calibration_dates"]
                population &= rows.day.isin(dates).to_numpy()
            sparse = nonoverlap(rows.day.to_numpy(), rows.symbol.to_numpy(), rows.SampleTime.to_numpy(), 300)
            selected_rows = rows.loc[population].reset_index(drop=True)
            chosen_sparse, classes = sparse[population], classes[population]
            labels = np.column_stack([classes == 1, classes == 2])
            daily_groups = selected_rows.groupby("day", sort=True).indices
            symbol_groups = selected_rows.groupby("symbol", sort=True).indices
            artifact = study / ("calibration-results.yaml" if role == "calibration" else "forward-results-300.yaml")
            metrics = read_yaml(artifact)
            candidates = {arm: (study / f"{arm}-300-{'tune' if role == 'calibration' else 'forward'}-scores.npy", values) for arm, values in metrics.items()}
            factorized = root / "study-factorized"
            if panel == "stock" and (factorized / "completed.yaml").exists():
                values = read_yaml(factorized / f"{role}-results.yaml")["factorized"]
                candidates["factorized"] = (factorized / f"factorized-300-{'tune' if role == 'calibration' else 'forward'}-scores.npy", values)
            for arm, (path, result) in candidates.items():
                scores = np.load(path, mmap_mode="r")[population]
                if scores.shape != (len(selected_rows), 2) or not np.isfinite(scores).all():
                    raise ContractError("Concentration scores differ from the recorded native population")
                mass = np.zeros_like(scores, dtype=float)
                daily_correct = []
                for indices in daily_groups.values():
                    for side in (0, 1):
                        score = scores[indices, side]
                        quota = max(1, round(0.01 * len(score)))
                        cutoff = -np.partition(-score, quota - 1)[quota - 1]
                        above, tied = score > cutoff, score == cutoff
                        selected = above.astype(float)
                        selected[tied] = (quota - int(above.sum())) / int(tied.sum())
                        mass[indices, side] = selected / (quota * len(daily_groups) * 2)
                    daily_correct.append(float((mass[indices] * labels[indices]).sum()))
                correct, opposite = mass * labels, mass * labels[:, ::-1]
                if (
                    not np.isclose(mass.sum(), 1, atol=1e-12)
                    or not np.isclose(correct.sum(), result["mean_p@0.01"], atol=1e-12)
                    or not np.isclose(opposite.sum(), result["mean_wrong@0.01"], atol=1e-12)
                ):
                    raise ContractError("Concentration tie handling differs from the frozen ranking metrics")
                by_symbol = []
                for symbol, indices in symbol_groups.items():
                    event = labels[indices].any(axis=1)
                    sparse_events = chosen_sparse[indices] & event
                    ap = [average_precision(labels[indices, side], scores[indices, side]) for side in (0, 1)]
                    record = {
                        "panel": panel,
                        "role": role,
                        "arm": arm,
                        "symbol": str(symbol),
                        "known_rows": len(indices),
                        "big_endpoints": int(event.sum()),
                        "big_endpoint_days": int(selected_rows.iloc[indices].loc[event, "day"].nunique()),
                        "nonoverlap_big_endpoints": int(sparse_events.sum()),
                        "nonoverlap_big_endpoint_days": int(selected_rows.iloc[indices].loc[sparse_events, "day"].nunique()),
                        "up_ap": ap[0],
                        "down_ap": ap[1],
                        "mean_ap": float(np.nanmean(ap)),
                        "direction_auc_given_big": auc(labels[indices, 0][event], (scores[indices, 0] - scores[indices, 1])[event]) if event.any() else np.nan,
                        "equal_day_two_side_top1_mass": float(mass[indices].sum()),
                        "correct_top1_contribution": float(correct[indices].sum()),
                        "opposite_top1_contribution": float(opposite[indices].sum()),
                    }
                    records.append(record)
                    by_symbol.append(record)
                total_correct = float(correct.sum())
                summaries.append(
                    {
                        "panel": panel,
                        "role": role,
                        "arm": arm,
                        "maximum_symbol_top1_mass": max(value["equal_day_two_side_top1_mass"] for value in by_symbol),
                        "top_three_symbol_top1_mass": sum(sorted((value["equal_day_two_side_top1_mass"] for value in by_symbol), reverse=True)[:3]),
                        "maximum_symbol_correct_hit_share": max(value["correct_top1_contribution"] for value in by_symbol) / total_correct if total_correct else None,
                        "maximum_day_correct_hit_share": max(daily_correct) / total_correct if total_correct else None,
                        "metric_reconciliation": True,
                    }
                )
    pd.DataFrame(records).to_csv(root / "symbol-concentration-300.csv", index=False)
    write_yaml(
        root / "concentration-summary.yaml",
        {
            "purpose": "descriptive_fixed_family_only_no_nomination_admission_or_threshold_changes",
            "top1_mass": "equal_day_equal_up_down_quota_with_fractional_boundary_ties",
            "interpretation": "Within-symbol metrics and concentration expose population alternatives; they do not identify causes or independent replications",
            "comparisons": summaries,
        },
    )
    print({"symbol_records": len(records), "comparisons": len(summaries)}, flush=True)


if __name__ == "__main__":
    main()
