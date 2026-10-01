"""Native indicator codebooks and coherent factorized model probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from study_data import native_classes

from AstraResearch.io import ContractError


def targets(rows, horizon=300):
    classes, known = native_classes(rows, horizon)
    big = known & (classes != 0)
    return {
        "known": known,
        "big": big,
        "event_label": (classes[known] != 0).astype(np.int8),
        "direction_label": rows[f"mid_endpoint.up.5[{horizon}s]"].to_numpy()[big],
    }


def weights(rows, known, half_life):
    origins = rows.loc[known, ["day", "symbol"]]
    counts = origins.groupby(["day", "symbol"])["day"].transform("size").to_numpy()
    age = (pd.to_datetime(origins.day.max()) - pd.to_datetime(origins.day)).dt.days.to_numpy()
    result = np.exp2(-age / half_life) / counts
    return result / result.mean()


def side_scores(event_probability, conditional_up_probability):
    event = np.asarray(event_probability, dtype=float)
    direction = np.asarray(conditional_up_probability, dtype=float)
    if event.shape != direction.shape or event.ndim != 1 or not np.isfinite([event, direction]).all() or np.any((event < 0) | (event > 1) | (direction < 0) | (direction > 1)):
        raise ContractError("Factorized binary predictions must have aligned finite probabilities")
    return np.column_stack([event * direction, event * (1.0 - direction)])
