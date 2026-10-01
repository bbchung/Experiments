"""Descriptive native-value cohorts; these diagnostics never nominate models."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from study_data import native_classes, nonoverlap

from AstraResearch.io import write_yaml


def permute_native_finite(values, rows):
    """Negative control: joint within-symbol-day shuffle; preserve native states."""
    result = np.array(values, copy=True)
    finite = np.isfinite(values).all(axis=1)
    for (day, symbol), indices in rows.groupby(["day", "symbol"], sort=True).indices.items():
        selected = indices[finite[indices]]
        seed = int.from_bytes(hashlib.sha256(f"20260930:{day}:{symbol}".encode()).digest()[:8], "little")
        order = np.random.default_rng(seed).permutation(selected)
        result[selected] = values[order]
    return result


def describe(matrix, rows, features, directory, role):
    classes, known = native_classes(rows, 300)
    positions = {name: index for index, name in enumerate(features)}
    fixed = nonoverlap(rows.day.to_numpy(), rows.symbol.to_numpy(), rows.SampleTime.to_numpy(), 300)
    cases = {"all": np.ones(len(rows), dtype=bool)}
    age = (rows.SampleTime.to_numpy() - rows.SampleBookTime.to_numpy()) / 1_000_000
    if (age < 0).any():
        raise AssertionError("A native sample contains a future book")
    cases.update({"quote_age_at_most_1s": age <= 1, "quote_age_above_5s": age > 5})
    flow_name = "FlowResponseSurprise.0.flow_surprise0.0"
    response_name = "FlowResponseSurprise.0.response_innovation0.0"
    flow = None
    if flow_name in positions and response_name in positions:
        flow = np.asarray(matrix[:, positions[flow_name]], dtype=float)
        response = np.asarray(matrix[:, positions[response_name]], dtype=float)
        available = np.isfinite(flow) & np.isfinite(response)
        cases["flow_response_finite"] = available
        cases["flow_response_unavailable"] = ~available
        directional = available & (np.abs(flow) >= 1)
        alignment = np.sign(flow) * response
        cases["high_flow_unexpectedly_easy_response"] = directional & (alignment >= 1)
        cases["high_flow_unexpectedly_absorbed_response"] = directional & (alignment <= -1)
        cases["high_flow_reference_like_response"] = directional & (np.abs(alignment) < 1)
    records = []
    for name, mask in cases.items():
        for (day, symbol), indices in rows.groupby(["day", "symbol"], sort=True).indices.items():
            selected = indices[mask[indices] & known[indices]]
            sparse = selected[fixed[selected]]
            record = {
                "role": role,
                "day": str(day),
                "symbol": str(symbol),
                "case": name,
                "origins": int(mask[indices].sum()),
                "known": len(selected),
                "up": int((classes[selected] == 1).sum()),
                "down": int((classes[selected] == 2).sum()),
                "nonoverlap_known": len(sparse),
                "nonoverlap_up": int((classes[sparse] == 1).sum()),
                "nonoverlap_down": int((classes[sparse] == 2).sum()),
            }
            if flow is not None and name.startswith("high_flow_"):
                up, down = classes[sparse] == 1, classes[sparse] == 2
                direction = flow[sparse] >= 0
                record.update(nonoverlap_flow_aligned=int(np.where(direction, up, down).sum()), nonoverlap_flow_opposed=int(np.where(direction, down, up).sum()))
            records.append(record)
    pd.DataFrame(records).to_csv(directory / f"mechanism-{role}.csv", index=False)
    write_yaml(
        directory / f"mechanism-{role}.yaml",
        {
            "role": role,
            "purpose": "descriptive_support_and_competing_explanations_no_model_selection",
            "cohort_source": "native_observed_values_and_sample_keys",
            "flow_threshold_native_z": 1,
            "response_alignment_threshold_native_z": 1,
            "interpretation": "easy_response_continuation_competes_with_transient_impact_reversal; observational_cohorts_do_not_identify_a_cause",
            "nonoverlap": "fixed_native_time_origins_before_scores_or_labels_not_independence_across_symbols",
        },
    )
