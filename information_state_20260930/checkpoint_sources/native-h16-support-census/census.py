"""Frozen label-free native raw-message census, never model predictors."""

import ctypes as ct
import importlib.util
from collections import Counter
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PREFIX = OUTPUT.parent / "native-h15-receive-prefix"
spec = importlib.util.spec_from_file_location("h16_census_prefix", PREFIX / "prepare_receive_prefix.py")
raw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(raw)
SYMBOLS = ("3481", "2344", "2337", "2330")
CONTRACT = {
    "schema": "h16-actual-auction-anchor-census-v1",
    "day": "20260119",
    "symbols": list(SYMBOLS),
    "receive_start": 1768784400000000,
    "receive_end": 1768798800000000,
    "source": "unchanged complete native Jan19 BIN spools from frozen H15 prefix selection",
    "actual_match_witness": "positive Trade quantity, AUCTION set, TRIAL and SUSPEND unset; no inferred side",
    "read": "raw message fields and status/own-E counts only, no labels, feature values or training",
    "models_permitted": 0,
}


def bind():
    receipt_path = PREFIX / "preparation-receipt.yaml"
    receipt = raw.read_yaml(receipt_path)
    selection_path = PREFIX / "frozen-selection-method.yaml"
    selection, _ = raw.verify_selection(selection_path)
    spools = {Path(n["path"]).stem: n for n in receipt["spools"]}
    raw.require(set(SYMBOLS).issubset(spools), "fixed original census spools missing; no replacement")
    inputs = [raw.record(receipt_path), raw.record(selection_path), raw.record(selection_path.with_suffix(".yaml.identity")), *[spools[s] for s in SYMBOLS]]
    for node in inputs:
        raw.check_record(node)
    return {
        "schema": "h16-actual-auction-anchor-census-bound-v1",
        "contract": CONTRACT,
        "profile_identity": raw.digest(CONTRACT),
        "sources": raw.unique_records([*selection["sources"], raw.record(Path(__file__))]),
        "inputs": raw.unique_records(inputs),
        "spools": {s: spools[s] for s in SYMBOLS},
        "abi": raw.abi(),
    }


def census(path):
    method = raw.canonical_method(path, "h16-actual-auction-anchor-census-method-v1")
    bound_path = OUTPUT / "bound-plan.yaml"
    raw.require(method["plan"] == raw.record(bound_path), "census plan changed")
    bound = raw.read_yaml(bound_path)
    raw.require(bound == bind(), "census contract/source evidence changed")
    for scope in ("sources", "inputs"):
        for node in method[scope]:
            raw.check_record(node)
    destination = OUTPUT / "census.yaml"
    raw.require(not destination.exists(), "fresh native source census required")
    cells = []
    for symbol in SYMBOLS:
        counts, trades, groups, witnesses = Counter(), Counter(), [], []
        previous, ordinal = 0, 0
        with Path(bound["spools"][symbol]["path"]).open("rb") as stream:
            while True:
                row = raw.read_record(stream, symbol, previous, ordinal)
                if row is None:
                    break
                previous, ordinal = row["receive"], ordinal + 1
                if not CONTRACT["receive_start"] <= row["receive"] <= CONTRACT["receive_end"]:
                    continue
                counts[f"{row['type']}:status{row['status']}"] += 1
                if row["type"] != "T":
                    continue
                trade = raw.Trade.from_buffer_copy(row["bytes"][ct.sizeof(raw.Header) :])
                if trade.quantity <= 0:
                    continue
                trades[f"status{row['status']}"] += 1
                if not (row["status"] & 2 and not row["status"] & 5 and raw.positive_clock(row["exchange"]) and raw.positive_clock(row["receive"])):
                    continue
                if not groups or groups[-1]["exchange"] != row["exchange"]:
                    groups.append({"exchange": row["exchange"], "first_receive": row["receive"], "prints": 0})
                groups[-1]["prints"] += 1
                if len(witnesses) < 4:
                    witnesses.append({**raw.evidence(row), "price": float(trade.price), "quantity": int(trade.quantity)})
        cells.append(
            {"symbol": symbol, "message_counts": dict(counts), "positive_trade_counts": dict(trades), "actual_auction_own_e_groups": groups, "first_actual_witnesses": witnesses}
        )
    raw.write_yaml(
        destination,
        {
            "schema": "h16-actual-auction-anchor-census-result-v1",
            "method_identity": method["identity"],
            "profile_identity": raw.digest(CONTRACT),
            "cells": cells,
            "sources": method["sources"],
            "inputs": method["inputs"],
            "labels_read": False,
            "python_features_generated": False,
            "model_fits": 0,
            "native_replays_launched": 0,
            "predictive_admission": False,
        },
    )
    return {
        "receipt": str(destination),
        "summary": [{"symbol": c["symbol"], "counts": c["positive_trade_counts"], "actual_groups": len(c["actual_auction_own_e_groups"])} for c in cells],
    }
