"""Select native representations and join exact observations; no feature math.

Native float64 H15 values and categories remain unchanged until the final
CatBoost conversion. The compact context uses the original full projection,
including its original producer vintage and all native missing states.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...feature_values import model_values
from ...io import ContractError
from ..representation_research import data as original

KEYS = ("day", "symbol", "SampleTime", "SampleBookTime", "SampleBookSeq")
NUMERIC = (
    "external_impulse_target_unit",
    "peer_work_fraction",
    "target_response_5ticks",
    "target_known_work_imbalance",
    "target_known_work_strength",
    "source_persistence",
    "mark_receive_age_300s",
)
CATEGORICAL = ("j_receive_relation", "j_response_alignment", "k_peer_phase", "k_target_phase")
CANONICAL = tuple(f"H15.{name}.0" for name in (*NUMERIC, *CATEGORICAL))
C_NUMERIC = (
    "SpreadState.0.cur_spread_ticks.0",
    "PriceFormationPath.0.sticky_rv_ticks.1",
    "IntradayRegime.0.magnitude_session_range.0",
    "TimeInfo.0.session_time.0",
    "IntradayRegime.0.direction_open_displacement.0",
    "IntradayRegime.0.direction_vwap_displacement.0",
    "IntradayRegime.0.direction_range_position.0",
    "BookSideCapacity.0.book_imbalance.0",
    "BookClockFlow.0.signed_trade_over_depth.2",
    "BookClockFlow.0.trade_over_depth.2",
    "BookClockFlow.0.known_trade_share.2",
    "BookClockFlow.0.net_supply_imbalance_over_depth.2",
    "TradeVolumeRegime.0.vol_regime_rel_total1.0",
    "FlowResponseSurprise.0.flow_surprise0.0",
    "FlowResponseSurprise.0.response_innovation0.0",
    "FlowResponseSurprise.0.response_coupling0.0",
    "PressureResponseState.0.absorption_bid_net_supply_ratio.0",
    "PressureResponseState.0.absorption_ask_net_supply_ratio.0",
)
C_CATEGORICAL = ("CurrentBook.1.sticky_side.0", "PriceFormationPath.0.cause_dominant.1")
ARMS = {
    "baseline": {"base": "current_nominal", "families": []},
    "compact": {"base": "compact_native", "families": []},
    "peer_k": {"base": "compact_native", "families": ["peer_k"]},
    "peer_j": {"base": "compact_native", "families": ["peer_j"]},
}


def validate_keys(frame):
    """Reject key coercion, nulls, duplicates and unavailable book identities."""
    if not frame.columns.is_unique or not set(KEYS).issubset(frame.columns):
        raise ContractError("H15 requires complete unambiguous five-field observation keys")
    if frame[list(KEYS)].isna().any().any() or frame.duplicated(list(KEYS)).any():
        raise ContractError("H15 requires unique nonnull five-field observation keys")
    for name in KEYS[:2]:
        if any(not isinstance(value, str) or not value for value in frame[name].unique()):
            raise ContractError("H15 day and symbol identities must remain canonical nonempty strings")
    if any(len(day) != 8 or not day.isascii() or not day.isdigit() or day < "20260101" for day in frame.day.unique()):
        raise ContractError("H15 observations must retain the declared 2026-or-later calendar identities")
    for name in KEYS[2:]:
        if frame[name].dtype != np.dtype("int64"):
            raise ContractError("H15 native observation times and book sequence must retain exact int64 storage")
        values = frame[name].to_numpy()
        if (values < 0).any() or (values >= 2**53).any():
            raise ContractError("H15 observation keys crossed the exact native integer domain")
    if (frame.SampleTime <= 0).any() or (frame.SampleBookTime <= 0).any() or (frame.SampleBookTime > frame.SampleTime).any():
        raise ContractError("H15 book observations must already be available at their original sample")


def _numeric_schema(frame, names, dtype):
    if not set(names).issubset(frame.columns) or any(frame[name].dtype != np.dtype(dtype) for name in names):
        raise ContractError("Selected native numeric fields changed their declared storage or descriptors")


def _categorical_schema(frame, names):
    if not set(names).issubset(frame.columns):
        raise ContractError("Selected native categorical descriptors are missing")
    for name in names:
        if frame[name].isna().any() or any(not isinstance(value, str) for value in frame[name].unique()):
            raise ContractError("Native categories must retain explicit strings and sentinel tokens; no null repair")


def _mid_bits(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != np.float64 or expected.dtype != np.float64 or actual.ndim != 1 or actual.shape != expected.shape:
        raise ContractError("H15 raw-mid alignment requires unchanged one-dimensional float64 origin observations")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all() or (actual <= 0).any() or (expected <= 0).any():
        raise ContractError("H15 origin mids must remain positive finite native observations")
    if not np.array_equal(actual.view(np.uint64), expected.view(np.uint64)):
        raise ContractError("H15 native raw-mid bits differ from the original observation")


def exact_join(anchors, native, expected_mids=None):
    """Preserve parent order and float64 bits; never asof, fill, cast or filter.

    Extra native observations are permitted. Every requested original key must
    exist. Parent raw mids must be supplied explicitly or in the anchor table.
    """
    validate_keys(anchors)
    validate_keys(native)
    _numeric_schema(native, ["OriginMidPrice", *CANONICAL[: len(NUMERIC)]], "float64")
    _categorical_schema(native, CANONICAL[len(NUMERIC) :])
    wanted = pd.MultiIndex.from_frame(anchors[list(KEYS)])
    provided = pd.MultiIndex.from_frame(native[list(KEYS)])
    if not wanted.isin(provided).all():
        raise ContractError("H15 is missing original exact book identities; no asof/fill or replacement rows")
    aligned = native.set_index(list(KEYS)).reindex(wanted)
    if expected_mids is None:
        if "OriginMidPrice" not in anchors:
            raise ContractError("H15 exact join requires the original raw-mid alignment witness")
        expected_mids = anchors.OriginMidPrice.to_numpy()
    _mid_bits(aligned.OriginMidPrice.to_numpy(), expected_mids)
    values = aligned[["OriginMidPrice", *CANONICAL]].reset_index(drop=True).copy(deep=True)
    return pd.concat([anchors[list(KEYS)].reset_index(drop=True).copy(deep=True), values], axis=1)


def read_native(path):
    """Read the canonical shared cache with Arrow null/type checks before pandas."""
    columns = [*KEYS, "OriginMidPrice", *CANONICAL]
    schema = pq.read_schema(path)
    if len(schema.names) != len(columns) or set(schema.names) != set(columns):
        raise ContractError("H15 shared cache must contain only original keys, raw mid and the canonical eleven fields")
    table = pq.read_table(path, columns=columns)
    for name in columns:
        column = table[name]
        if column.null_count:
            raise ContractError("H15 cache cannot replace explicit native sentinel states with Arrow nulls")
        if name in KEYS[2:]:
            valid = column.type == pa.int64()
        elif name in (*KEYS[:2], *CANONICAL[len(NUMERIC) :]):
            valid = pa.types.is_string(column.type) or pa.types.is_large_string(column.type)
        else:
            valid = column.type == pa.float64()
        if not valid:
            raise ContractError(f"H15 cache field {name} changed its native key/numeric/category storage")
    frame = table.to_pandas()
    validate_keys(frame)
    return frame


def _mask(profile, rows, role):
    """The original frozen cohort: no feature-dependent sampling or filtering."""
    mask = rows.day.isin(profile["splits"][role]).to_numpy() & original.sampled_mask(rows, profile["sampling"]["interval_seconds"])
    for horizon in profile["horizons"]:
        _, complete = original.native_classes(rows, horizon)
        mask &= complete
    if role in ("train", "es", "calibration"):
        mask &= ~rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).to_numpy()
    return mask


def _select_numeric(matrix, available, selected, indices):
    if matrix.dtype != np.float32 or len(set(available)) != len(available) or len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ContractError("Original native projection or selected numeric descriptors changed")
    positions = {name: index for index, name in enumerate(available)}
    # Advanced indexing already owns storage independent of the parent mmap.
    return np.asarray(matrix[np.ix_(indices, [positions[name] for name in selected])], dtype=np.float32, order="C")


def _model_input(frame, numeric, categorical):
    """Convert only at the consumer boundary; reject finite/zero state collapse."""
    fields = numeric + categorical
    result, cats = model_values(frame[fields], fields, categorical)
    for name in numeric:
        source, target = frame[name].to_numpy(), result[name].to_numpy()
        if (np.isfinite(source) & ~np.isfinite(target)).any() or ((source != 0) & np.isfinite(source) & (target == 0)).any():
            raise ContractError("CatBoost float32 conversion crossed a native finite or exact nonzero boundary")
    return result, cats


def load_role(profile, role, arm):
    """Select C or the original baseline, then select native K/J without FE math."""
    if role not in ("train", "es", "calibration", "development") or arm not in ARMS or profile["arms"].get(arm) != ARMS[arm]:
        raise ContractError("V4 requires a declared original role and one exact prospective representation arm")
    if profile["compact"] != {"numeric_fields": list(C_NUMERIC), "categorical_fields": list(C_CATEGORICAL)}:
        raise ContractError("V4 compact context differs from its source-audited native descriptor selection")
    if profile["h15"]["numeric_fields"] != list(NUMERIC) or profile["h15"]["categorical_fields"] != list(CATEGORICAL):
        raise ContractError("V4 H15 native descriptor order changed")
    if profile["horizons"] != [60, 120, 180, 300] or profile["sampling"]["interval_seconds"] != 30:
        raise ContractError("V4 must inherit the original four-horizon completeness and 30s observation cohort")
    native_role = "train" if role == "train" else "forward" if role == "development" else "tune"
    root = Path(profile["paths"]["output"])
    parent = Path(profile["paths"]["parent_study"]) / native_role
    matrix, all_rows, available, _ = original.load_projection(parent)
    validate_keys(all_rows)
    numeric, nominal, _ = original.baseline_scope(profile)
    if arm != "baseline":
        numeric, nominal = list(C_NUMERIC), list(C_CATEGORICAL)
    else:
        numeric, nominal = list(numeric), list(nominal)
    mask = _mask(profile, all_rows, role)
    indices = np.flatnonzero(mask)
    rows = all_rows.iloc[indices].reset_index(drop=True).copy(deep=True)
    x = _select_numeric(matrix, available, numeric, indices)
    nominal_frame = pd.read_parquet(parent / "nominal.parquet", columns=[*KEYS, *nominal])
    validate_keys(nominal_frame)
    if not nominal_frame[list(KEYS)].equals(all_rows[list(KEYS)]):
        raise ContractError("Original native numeric and categorical complete row identities disagree")
    _categorical_schema(nominal_frame, nominal)
    selected_nominal, cats = _model_input(nominal_frame.iloc[indices][nominal], [], nominal)
    cat = selected_nominal[cats].to_numpy(dtype=object, copy=True)
    if arm in ("peer_k", "peer_j"):
        native = read_native(root / "shared" / f"{native_role}-h15.parquet")
        if not native[list(KEYS)].equals(all_rows[list(KEYS)]):
            raise ContractError("H15 shared cache must retain the complete original parent order and cohort")
        mids = np.load(root / "shared" / f"mid-{native_role}.npy", mmap_mode="r")
        _mid_bits(native.OriginMidPrice.to_numpy(), mids)
        # Full-parent alignment was verified above; slicing does not change keys,
        # native numerical bits, categorical tokens or inherited label rows.
        extra_names = list(CANONICAL[: len(NUMERIC)])
        offset = len(NUMERIC) + (2 if arm == "peer_k" else 0)
        extra_cats = list(CANONICAL[offset : offset + 2])
        extras, _ = _model_input(native.iloc[indices][extra_names + extra_cats], extra_names, extra_cats)
        x = np.asarray(np.column_stack([x, extras[extra_names].to_numpy(dtype=np.float32)]), dtype=np.float32, order="C")
        cat = np.asarray(np.column_stack([cat, extras[extra_cats].to_numpy(dtype=object)]), dtype=object, order="C")
        numeric += extra_names
        nominal += extra_cats
    return rows, x, cat, numeric, nominal
