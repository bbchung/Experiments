"""Exact native observation joins; no labels, row selection or sentinel repair."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...feature_values import model_values
from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from ..representation_research import runner as inherited
from ..representation_temporal import validate_clocks
from ..representation_temporal import verify as verify_temporal

KEYS = ["day", "symbol", "SampleTime", "SampleBookTime", "SampleBookSeq"]
NUMERIC = ("direction", "log_work_strength", "signed_after_containment_share", "signed_after_repair_share", "delayed_repair_fraction", "signed_renewed_work", "mid_progress_5ticks")
CATEGORICAL = ("phase",)
CANONICAL = [f"H10.{name}.0" for name in (*NUMERIC, *CATEGORICAL)]
PHASES = {"warmup", "inactive", "contained", "repaired", "renewed", "resumed", "stale", "censored"}
CELL_METADATA = ["OriginMidPrice", *[f"H10_{variant}_processed_cluster_{clock}_time" for variant in ("ordered", "reset") for clock in ("exchange", "available")]]


def validate_dependencies(receipt):
    if not receipt.get("sources") or not receipt.get("inputs"):
        raise ContractError("Native validation must close explicit producer/export/parser source and raw/config dependencies")
    for record in receipt["sources"] + receipt["inputs"]:
        if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record) or file_hash(Path(record["path"])) != record["sha256"]:
            raise ContractError("Native producer/source/input no longer matches its validated replay receipt")


def producer_lineage(profile):
    receipt = read_yaml(Path(profile["h10"]["producer_lineage_receipt"]))
    if receipt.get("schema") != "native-component-lineage-validation-v2" or not all(
        receipt.get(key) is True for key in ("passed", "baseline_parent_reproduced", "current_control_invariant")
    ):
        raise ContractError("V3 requires separately validated historical baseline and current H10 component lineage")
    for key in ("method_contract_identity", "original_binary_sha256", "current_binary_sha256", "h10_binary_sha256"):
        if not isinstance(receipt.get(key), str) or re.fullmatch(r"[a-f0-9]{64}", receipt[key]) is None:
            raise ContractError("Native component lineage lacks its frozen method or producer identity")
    if receipt["h10_binary_sha256"] != file_hash(Path(profile["h10"]["binary"])):
        raise ContractError("H10 binary differs from its separately validated component lineage")
    validate_dependencies(receipt)
    if not isinstance(receipt.get("method_contract_path"), str) or not receipt["method_contract_path"]:
        raise ContractError("Native component lineage has no frozen method manifest path")
    method_path = Path(receipt["method_contract_path"])
    anchor = method_path.with_name(method_path.name + ".identity")
    if not method_path.is_file() or not anchor.is_file():
        raise ContractError("Native component lineage has no frozen method manifest and identity anchor")
    method = read_yaml(method_path)
    if (
        not isinstance(method, dict)
        or method.get("identity") != receipt["method_contract_identity"]
        or method["identity"] != digest({key: value for key, value in method.items() if key != "identity"})
        or anchor.read_text(encoding="ascii").strip() != method["identity"]
    ):
        raise ContractError("Native component method differs from its frozen self-identity or anchor")
    validate_dependencies(method)
    input_paths = {str(Path(record["path"]).resolve()) for record in receipt["inputs"]}
    if any(str(path.resolve()) not in input_paths for path in (method_path, anchor)):
        raise ContractError("Native component method manifest and anchor must be frozen lineage inputs")
    for scope in ("sources", "inputs"):
        actual = {str(Path(record["path"]).resolve()): record["sha256"] for record in receipt[scope]}
        if any(actual.get(str(Path(record["path"]).resolve())) != record["sha256"] for record in method[scope]):
            raise ContractError("Native component lineage omitted part of its frozen method source/input closure")
    return receipt


def prefix_validation(profile):
    receipt = read_yaml(Path(profile["h10"]["prefix_validation_receipt"]))
    if receipt.get("schema") != "native-receive-prefix-validation-v1" or not all(
        receipt.get(key) is True for key in ("passed", "genuine_straddle_found", "common_origin_present", "all_common_origin_bits_exact", "compiled_source_snapshot_bound")
    ):
        raise ContractError("H10 requires an actual receive-prefix replay with genuine straddle and exact common-origin bits")
    if receipt.get("binary_sha256") != file_hash(Path(profile["h10"]["binary"])):
        raise ContractError("H10 receive-prefix proof changed its immutable native binary")
    validate_dependencies(receipt)
    return receipt


def temporal_validation(profile):
    """Require the independently frozen clock-domain study before FE use."""
    cfg = profile["h10"]
    method_path = Path(cfg["temporal_method_contract"]).resolve()
    anchor = Path(str(method_path) + ".identity")
    method = verify_temporal(method_path)
    identity = cfg.get("temporal_method_identity")
    if not isinstance(identity, str) or re.fullmatch(r"[a-f0-9]{64}", identity) is None or method["identity"] != identity:
        raise ContractError("V3 temporal methodology differs from its predeclared scientific identity")
    binary_node = {"path": str(Path(cfg["binary"]).resolve()), "sha256": file_hash(Path(cfg["binary"]))}
    if binary_node not in method["inputs"]:
        raise ContractError("V3 H10 producer binary is outside the frozen temporal study")
    receipt_path = Path(cfg["temporal_validation_receipt"]).resolve()
    receipt = read_yaml(receipt_path)
    if receipt.get("method_contract_identity") != identity or receipt.get("method_contract_path") != str(method_path):
        raise ContractError("V3 temporal validation receipt changed its frozen method binding")
    payload = {key: value for key, value in receipt.items() if key not in ("method_contract_identity", "method_contract_path", "sources", "inputs")}
    if payload != method["validation"]:
        raise ContractError("V3 temporal validation differs from the independently frozen evidence")
    validate_dependencies(receipt)
    for scope in ("sources", "inputs"):
        required = list(method[scope])
        if scope == "inputs":
            required.extend({"path": str(path), "sha256": file_hash(path)} for path in (method_path, anchor))
        if any(record not in receipt[scope] for record in required):
            raise ContractError("V3 temporal receipt omitted its frozen method source/input closure")
    return {**receipt, "inputs": receipt["inputs"] + [{"path": str(receipt_path), "sha256": file_hash(receipt_path)}]}


def native_validation(profile):
    cfg = profile["h10"]
    receipt = read_yaml(Path(cfg["validation_receipt"]))
    if receipt.get("passed") is not True or receipt.get("binary_sha256") != file_hash(Path(cfg["binary"])):
        raise ContractError("H10 requires passed native support/semantics/parity validation bound to its immutable binary")
    validate_dependencies(receipt)
    lineage = producer_lineage(profile)
    prefix = prefix_validation(profile)
    temporal = temporal_validation(profile)
    return {
        **receipt,
        "sources": receipt["sources"] + lineage["sources"] + prefix["sources"] + temporal["sources"],
        "inputs": receipt["inputs"] + lineage["inputs"] + prefix["inputs"] + temporal["inputs"],
    }


def validate_keys(frame):
    if any(name not in frame for name in KEYS) or frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ContractError("Native H10 requires unique complete five-field observation keys")
    for name in KEYS[2:]:
        if not pd.api.types.is_integer_dtype(frame[name]):
            raise ContractError("Native H10 times and book sequence must be exact integers")
    if (frame.SampleBookTime > frame.SampleTime).any() or (frame.SampleBookSeq < 0).any():
        raise ContractError("Native H10 crossed its observation-time or book-sequence boundary")


def exact_join(anchors, native, producer):
    """Additional native rows are allowed; every parent observation must exist."""
    anchors, native = anchors.copy(), native.copy()
    for frame in (anchors, native):
        frame["day"], frame["symbol"] = frame.day.astype(str), frame.symbol.astype(str)
        validate_keys(frame)
    fields = [f"{producer}.{field}.0" for field in (*NUMERIC, *CATEGORICAL)]
    if not set(fields).issubset(native):
        raise ContractError("H10 native export is missing a declared numeric or categorical predictor")
    wanted = pd.MultiIndex.from_frame(anchors[KEYS])
    provided = pd.MultiIndex.from_frame(native[KEYS])
    if not wanted.isin(provided).all():
        raise ContractError("Native H10 is missing original exact book identities; no asof, forwardfill or replacement rows")
    values = native.set_index(KEYS)[fields].reindex(wanted).reset_index(drop=True)
    values.columns = CANONICAL
    values, cats = model_values(values, CANONICAL, [CANONICAL[-1]])
    if cats != [CANONICAL[-1]]:
        raise ContractError("H10 phase changed its categorical schema")
    return pd.concat([anchors[KEYS].reset_index(drop=True), values], axis=1)


def validate_cell(anchors, native, expected_mids, producers):
    """Check exact parent mid and native causal/state metadata before casting."""
    validate_keys(anchors)
    validate_keys(native)
    wanted = pd.MultiIndex.from_frame(anchors[KEYS])
    if not wanted.isin(pd.MultiIndex.from_frame(native[KEYS])).all() or not set(CELL_METADATA).issubset(native):
        raise ContractError("H10 native cell lacks original identities or required integrity metadata")
    aligned = native.set_index(KEYS).reindex(wanted)
    actual_mid, expected_mids = aligned.OriginMidPrice.to_numpy(), np.asarray(expected_mids)
    if actual_mid.dtype != np.float64 or expected_mids.dtype != np.float64 or actual_mid.shape != expected_mids.shape:
        raise ContractError("H10 raw-mid integrity comparison requires unchanged float64 parent observations")
    if not np.array_equal(actual_mid.view(np.uint64), expected_mids.view(np.uint64)):
        raise ContractError("H10 native raw mid changed bits at an original historical observation")
    origin = anchors.SampleTime.to_numpy()
    for family, producer in producers.items():
        fields = [f"{producer}.{field}.0" for field in NUMERIC]
        clocks = [f"H10_{family.removeprefix('h10_')}_processed_cluster_{clock}_time" for clock in ("exchange", "available")]
        if any(aligned[field].dtype != np.float64 for field in fields + clocks):
            raise ContractError("H10 native numeric/diagnostic storage must retain float64 precision")
        values = aligned[fields].to_numpy()
        phase = aligned[f"{producer}.phase.0"]
        if phase.isna().any() or any(not isinstance(value, str) for value in phase.unique()) or not phase.isin(PHASES).all():
            raise ContractError("H10 native cell contains an unsupported phase category")
        warmup, inactive, missing = phase.eq("warmup").to_numpy(), phase.eq("inactive").to_numpy(), phase.isin({"stale", "censored"}).to_numpy()
        active = ~(warmup | inactive | missing)
        if (
            not np.isneginf(values[warmup]).all()
            or not (values[inactive].view(np.uint64) == 0).all()
            or not np.isnan(values[missing]).all()
            or not np.isfinite(values[active]).all()
            or not np.isin(values[active, 0], [-1.0, 1.0]).all()
            or not (values[active, 1] > 0.0).all()
        ):
            raise ContractError("H10 native phase crossed its numeric warmup/inactive/uncomputable/direction contract")
        exchange, available = (aligned[field].to_numpy() for field in clocks)
        validate_clocks(exchange, available, origin, observed=active)


def read_native(path, columns):
    """One narrow read preserves Arrow null/type distinctions before pandas."""
    table = pq.read_table(path, columns=columns)
    for name in columns:
        column = table[name]
        if column.null_count:
            raise ContractError("H10 native export has Arrow nulls; missing states must remain explicit numeric/string values")
        if name in KEYS[2:]:
            if column.type != pa.int64():
                raise ContractError("H10 native observation keys must retain their int64 storage")
        elif name.endswith(".phase.0"):
            if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
                raise ContractError("H10 native phase must retain its categorical string storage")
        elif column.type != pa.float64():
            raise ContractError("H10 native numeric/diagnostic storage must retain float64 precision")
    return table.to_pandas()


def prepare(profile):
    root = inherited.output(profile)
    if (root / "frozen-contract.yaml").exists():
        raise ContractError("Frozen V3 feature artifacts cannot be prepared or overwritten")
    (root / "shared" / "h10-preparation.yaml").unlink(missing_ok=True)
    native_validation(profile)
    native_root = Path(profile["h10"]["native_root"])
    inputs, caches = {}, []
    for role in ("train", "tune", "forward"):
        parent = Path(profile["paths"]["parent_study"]) / role
        anchors = pd.read_parquet(parent / "rows.parquet", columns=KEYS)
        anchors["day"], anchors["symbol"] = anchors.day.astype(str), anchors.symbol.astype(str)
        mids = np.load(root / "shared" / f"mid-{role}.npy", mmap_mode="r")
        if len(mids) != len(anchors) or mids.dtype != np.float64:
            raise ContractError("Inherited H10 parent raw-mid cache changed its row count or float64 storage")
        positions_in_parent = np.arange(len(anchors))
        requested = (
            profile["splits"]["train"] if role == "train" else profile["splits"]["development"] if role == "forward" else profile["splits"]["es"] + profile["splits"]["calibration"]
        )
        selected = anchors.day.isin(requested).to_numpy()
        anchors = anchors.loc[selected].reset_index(drop=True)
        positions_in_parent = positions_in_parent[selected]
        if not len(anchors):
            raise ContractError(f"No original native anchor support in the declared {role} dates")
        validate_keys(anchors)
        frames = {family: [] for family in profile["h10"]["producers"]}
        positions = {family: [] for family in frames}
        all_fields = [f"{producer}.{field}.0" for producer in profile["h10"]["producers"].values() for field in (*NUMERIC, *CATEGORICAL)]
        for (day, symbol), indices in anchors.groupby(["day", "symbol"], sort=True).indices.items():
            path = native_root / day / "data" / day / symbol / "values.parquet"
            if not path.is_file():
                raise ContractError(f"Missing H10 native output for original observed cell {day}/{symbol}; no backfill")
            native = read_native(path, [*KEYS[2:], *all_fields, *CELL_METADATA])
            native.insert(0, "symbol", symbol)
            native.insert(0, "day", day)
            validate_cell(anchors.iloc[indices], native, mids[positions_in_parent[indices]], profile["h10"]["producers"])
            inputs[str(path)] = file_hash(path)
            for family, producer in profile["h10"]["producers"].items():
                frames[family].append(exact_join(anchors.iloc[indices], native, producer))
                positions[family].extend(indices)
        for family, parts in frames.items():
            combined = pd.concat(parts, ignore_index=True)
            combined.index = positions[family]
            combined = combined.sort_index().reset_index(drop=True)
            if not combined[KEYS].equals(anchors[KEYS].reset_index(drop=True)):
                raise ContractError("H10 preparation changed complete parent anchor order")
            path = root / "shared" / f"{role}-{family}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(path, index=False)
            caches.append({"path": str(path), "sha256": file_hash(path), "rows": len(combined), "numeric": CANONICAL[:-1], "categorical": CANONICAL[-1:]})
    receipt = {
        "passed": True,
        "labels_read": False,
        "catboost_fits": 0,
        "all_original_cells_mid_clock_phase_validated": True,
        "native_files": inputs,
        "caches": caches,
        "join": "exact_parent_five_keys_all_original_rows",
    }
    write_yaml(root / "shared" / "h10-preparation.yaml", receipt)
    return receipt


def load_role(profile, role, arm, *, baseline=None):
    """Reuse the baseline's exact population and add H10 without filtering rows."""
    rows, x, cat, names, cats = inherited.load_role(profile, role, "baseline") if baseline is None else baseline
    names, cats = list(names), list(cats)
    families = profile["arms"][arm]["families"]
    if not families:
        return rows, x, cat, names, cats
    if len(families) != 1 or families[0] not in profile["h10"]["producers"]:
        raise ContractError("H10 arm must use exactly one predeclared native chronology variant")
    native_role = "train" if role == "train" else "forward" if role == "development" else "tune"
    frame = pd.read_parquet(inherited.output(profile) / "shared" / f"{native_role}-{families[0]}.parquet")
    # Cached canonical names bind the variant via the frozen file/profile recipe.
    synthetic_desc = "H10"
    joined = exact_join(rows[KEYS], frame, synthetic_desc)
    extras, extra_cats = model_values(joined, CANONICAL, CANONICAL[-1:])
    numeric = np.asarray(extras[CANONICAL[:-1]], dtype=np.float32)
    nominal = extras[extra_cats].to_numpy(dtype=object)
    x = np.asarray(np.column_stack([x, numeric]), dtype=np.float32, order="C")
    cat = np.asarray(np.column_stack([cat, nominal]), dtype=object, order="C")
    return rows, x, cat, names + CANONICAL[:-1], cats + extra_cats
