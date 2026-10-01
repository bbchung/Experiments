"""Select original native values only; no Python feature calculations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...io import ContractError
from ..representation_research import data as original
from ..representation_research_v4 import data as audited

KEYS = audited.KEYS
C18_NUMERIC = audited.C_NUMERIC
C_CATEGORICAL = audited.C_CATEGORICAL
FLOW_NUMERIC = ("FlowResponseSurprise.0.flow_surprise0.0", "FlowResponseSurprise.0.response_innovation0.0", "FlowResponseSurprise.0.response_coupling0.0")
C15_NUMERIC = tuple(name for name in C18_NUMERIC if name not in FLOW_NUMERIC)
ARMS = {
    "baseline": {"base": "current_nominal", "families": []},
    "c15": {"base": "native_c15", "families": []},
    "c18": {"base": "native_c18", "families": []},
}


def arm_names(profile, arm):
    if arm not in ARMS or profile["arms"].get(arm) != ARMS[arm]:
        raise ContractError("V5 requires one exact prospective original-native representation arm")
    if arm == "baseline":
        numeric, nominal, _ = original.baseline_scope(profile)
        return list(numeric), list(nominal)
    return list(C15_NUMERIC if arm == "c15" else C18_NUMERIC), list(C_CATEGORICAL)


def load_role(profile, role, arm):
    """Identical original rows/labels; exact selectors preserve original vintage."""
    if role not in ("train", "es", "calibration", "development"):
        raise ContractError("V5 requires an original frozen scientific role")
    numeric, nominal = arm_names(profile, arm)
    native_role = "train" if role == "train" else "forward" if role == "development" else "tune"
    parent = Path(profile["paths"]["parent_study"]) / native_role
    matrix, all_rows, available, _ = original.load_projection(parent)
    audited.validate_keys(all_rows)
    indices = np.flatnonzero(audited._mask(profile, all_rows, role))
    x = audited._select_numeric(matrix, available, numeric, indices)
    nominal_frame = pd.read_parquet(parent / "nominal.parquet", columns=[*KEYS, *nominal])
    audited.validate_keys(nominal_frame)
    if not nominal_frame[list(KEYS)].equals(all_rows[list(KEYS)]):
        raise ContractError("V5 original numeric and categorical complete row identities disagree")
    audited._categorical_schema(nominal_frame, nominal)
    # Original projection is already float32. No normalization, imputation,
    # categorical recoding, native rebuilding or feature availability filtering.
    cat = nominal_frame.iloc[indices][nominal].to_numpy(dtype=object, copy=True)
    return all_rows.iloc[indices].reset_index(drop=True).copy(deep=True), x, cat, numeric, nominal
