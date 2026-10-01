"""Strict new-price correction to the preregistered native support proxy.

Strict quote-location certainty follows CausalTradeSideInference confidence=1.
Additional receive/exchange freshness and whole-cluster censoring deliberately
make this a conservative independent proxy, not full native module parity.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

DIRECTORY = Path(__file__).resolve().parent
REFERENCE = DIRECTORY / "preregistered-v1"
OUTPUT = REFERENCE / "distinct-price-v2"
spec = importlib.util.spec_from_file_location("h10_census", DIRECTORY / "census.py")
primitive = importlib.util.module_from_spec(spec)

sys.modules[spec.name] = primitive
spec.loader.exec_module(primitive)


def known_direction(row, previous):
    if previous is None or int(row["StatusMask"], 16) & 7:
        return 0
    quantity = int(row["TradeVolume"])
    price = float(row["Price"])
    if quantity <= 0 or not 0 < price < float("inf"):
        return 0
    if not 0 <= row["_receive"] - previous.receive <= primitive.FRESH_MICROS or not 0 <= row["_event"] - previous.exchange <= primitive.FRESH_MICROS:
        return 0
    location = 1 if price >= previous.ask[0] else -1 if price <= previous.bid[0] else 0
    feed = int(row["Side"])
    sign = 1 if feed == 1 else -1 if feed == 2 else location if feed == 0 else 0
    return sign if sign != 0 and sign == location else 0


def mapped_cluster(cluster, previous, contract):
    mapped = []
    signs = set()
    feed_signs = set()
    temporary = previous
    for row in cluster:
        if int(row["StatusMask"], 16) & 7:
            mapped.append((row, None, 0))
            temporary = None
        elif row["Type"] == "B":
            temporary = primitive.normalized_book(row, contract)
            mapped.append((row, temporary, 0))
        elif row["Type"] == "T":
            sign = known_direction(row, temporary)
            if sign:
                signs.add(sign)
            if int(row["TradeVolume"]) > 0 and int(row["Side"]) in (1, 2):
                feed_signs.add(int(row["Side"]))
            mapped.append((row, None, sign))
        else:
            mapped.append((row, None, 0))
    return mapped, len(signs) > 1 or len(feed_signs) > 1


def census(day, symbol, maximum_gap):
    path = primitive.SOURCE / symbol / (day + ".csv.zst")
    contract_path = primitive.CONTRACTS / (day + ".csv")
    if not path.exists() or not contract_path.exists():
        return {"day": day, "symbol": symbol, "status": "missing_skip_no_replacement"}
    with contract_path.open() as stream:
        contract = next((row for row in csv.DictReader(stream) if row["symbol"] == symbol), None)
    if contract is None:
        return {"day": day, "symbol": symbol, "status": "missing_contract_skip_no_replacement"}
    epoch = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC).timestamp()) * 1_000_000
    start, end = epoch + 4_200_000_000, epoch + 18_000_000_000
    previous = None
    pending = []
    interval_uncertain = False
    pattern = None
    last_exchange = 0
    counts = Counter()
    examples = []
    durations = []
    rows = primitive.schema_rows(path)
    header = next(rows)
    for event_time, cluster, available in primitive.closed_clusters(rows):
        active = start <= available < end
        mapped, mixed = mapped_cluster(cluster, previous, contract)
        backwards = event_time < last_exchange
        last_exchange = event_time
        if active:
            counts["exchange_clusters"] += 1
            counts["raw_rows"] += len(cluster)
            counts["mixed_clusters_censored"] += mixed
            counts["exchange_regressions_censored"] += backwards
        if mixed or backwards:
            pattern = None
            interval_uncertain = True
        for row, current, sign in mapped:
            if int(row["StatusMask"], 16) & 7:
                if active:
                    counts["noncontinuous_rows_censored"] += 1
                previous = None
                pending = []
                interval_uncertain = False
                pattern = None
                continue
            if row["Type"] == "T":
                if active:
                    counts["continuous_trades"] += 1
                    counts["continuous_trade_quantity"] += max(int(row["TradeVolume"]), 0)
                if sign == 0 or mixed or backwards:
                    if active:
                        counts["unknown_or_mixed_trades_censored"] += 1
                    interval_uncertain = True
                    pattern = None
                    continue
                if active:
                    counts["strict_known_trades"] += 1
                    counts["strict_known_quantity"] += int(row["TradeVolume"])
                if pattern:
                    if sign != pattern["sign"]:
                        if active:
                            counts["opposite_demand_censored"] += 1
                        pattern = None
                    elif row["_receive"] - pattern["last_demand_receive"] > maximum_gap:
                        if active:
                            counts["demand_gap_censored"] += 1
                        pattern = None
                    else:
                        pattern["last_demand_receive"] = row["_receive"]
                pending.append((sign, float(row["Price"]), int(row["TradeVolume"]), event_time, row["_receive"]))
                continue
            if row["Type"] != "B" or current is None or backwards:
                if active:
                    counts["invalid_or_unsupported_book_censored"] += 1
                previous = None
                pending = []
                interval_uncertain = False
                pattern = None
                continue
            if active:
                counts["valid_continuous_books"] += 1
            if previous is None:
                previous = current
                pending = []
                interval_uncertain = False
                continue
            fresh = 0 <= current.receive - previous.receive <= primitive.FRESH_MICROS and 0 <= current.exchange - previous.exchange <= primitive.FRESH_MICROS
            if interval_uncertain or not fresh or mixed:
                if active:
                    counts["uncertain_or_stale_interval_censored"] += 1
                pattern = None
                previous = current
                pending = []
                interval_uncertain = mixed
                continue
            if pattern and current.receive - pattern["last_demand_receive"] > maximum_gap:
                if active:
                    counts["demand_gap_censored"] += 1
                pattern = None
            if pattern:
                tracked = pattern["renew_price"] if pattern["phase"] == "renewed" else pattern["anchor_price"]
                old_quantity = previous.quantity_at(pattern["sign"], tracked)
                new_quantity = current.quantity_at(pattern["sign"], tracked)
                if old_quantity is None or new_quantity is None:
                    if active:
                        counts["nonobservable_anchor_censored"] += 1
                    pattern = None
            if (
                pattern
                and pattern["phase"] == "contained"
                and (
                    not primitive.eq(previous.prices(pattern["sign"])[0], pattern["anchor_price"]) or not primitive.eq(current.prices(pattern["sign"])[0], pattern["anchor_price"])
                )
            ):
                # Repair observed after a touch change cannot retroactively be
                # placed before that renewal in the ordered chain.
                if active:
                    counts["touch_renewal_before_delayed_repair_censored"] += 1
                pattern = None
            if (
                pattern
                and not pending
                and pattern["phase"] in ("contained", "repaired")
                and event_time > pattern["contain_exchange"]
                and primitive.eq(previous.prices(pattern["sign"])[0], pattern["anchor_price"])
                and primitive.eq(current.prices(pattern["sign"])[0], pattern["anchor_price"])
            ):
                direction = pattern["sign"]
                anchor = pattern["anchor_price"]
                supply = current.quantity_at(direction, anchor) - previous.quantity_at(direction, anchor)
                if supply > 0:
                    if active:
                        counts["delayed_quote_only_repair_events"] += 1
                    pattern["phase"] = "repaired"
                    pattern["repair_exchange"] = event_time
                    pattern["repair_available"] = available
                    pattern["repair_quantity"] = pattern.get("repair_quantity", 0) + supply
            if pending:
                directions = {item[0] for item in pending}
                if len(directions) != 1:
                    if active:
                        counts["mixed_book_interval_censored"] += 1
                    pattern = None
                else:
                    direction = next(iter(directions))
                    touch = previous.prices(direction)[0]
                    exact = [item for item in pending if primitive.eq(item[1], touch)]
                    quantity = sum(item[2] for item in exact)
                    remaining = current.quantity_at(direction, touch)
                    if quantity > 0 and remaining is not None:
                        if active:
                            counts["observable_exact_touch_challenges"] += 1
                        if (
                            pattern
                            and pattern["phase"] == "renewed"
                            and direction == pattern["sign"]
                            and primitive.eq(touch, pattern["renew_price"])
                            and min(item[3] for item in exact) > pattern["renew_exchange"]
                        ):
                            distinct_anchor = not primitive.eq(touch, pattern["anchor_price"])
                            if active and not distinct_anchor:
                                counts["anchor_return_resumption_excluded"] += 1
                            if active and distinct_anchor:
                                counts["complete_delayed_repair_chains"] += 1
                                counts["buy_complete_chains" if direction == 1 else "sell_complete_chains"] += 1
                                duration = (available - pattern["start_available"]) / 1e6
                                durations.append(duration)
                                if len(examples) < 8:
                                    examples.append({**pattern, "resume_exchange": event_time, "resume_available": available, "resume_price": touch, "resume_quantity": quantity})
                            if distinct_anchor:
                                pattern = None
                        if primitive.eq(current.prices(direction)[0], touch) and remaining > 0:
                            if active:
                                counts["contained_exact_touch_challenges"] += 1
                            if pattern is None:
                                pattern = {
                                    "sign": direction,
                                    "phase": "contained",
                                    "anchor_price": touch,
                                    "start_available": available,
                                    "contain_exchange": event_time,
                                    "last_demand_receive": max(item[4] for item in exact),
                                }
                    elif quantity > 0:
                        if active:
                            counts["nonobservable_challenge_censored"] += 1
                        pattern = None
            # Renewal follows repair and retains its mark; a subsequent attack
            # is evaluated above against the price already known before it.
            if pattern and pattern["phase"] == "repaired" and event_time > pattern["repair_exchange"]:
                touch = current.prices(pattern["sign"])[0]
                if not primitive.eq(touch, pattern["anchor_price"]):
                    pattern["phase"] = "renewed"
                    pattern["renew_exchange"] = event_time
                    pattern["renew_available"] = available
                    pattern["renew_price"] = touch
                    if active:
                        counts["repaired_then_touch_renewal"] += 1
            elif pattern and pattern["phase"] == "renewed":
                touch = current.prices(pattern["sign"])[0]
                if not primitive.eq(touch, pattern["renew_price"]):
                    pattern["renew_exchange"] = event_time
                    pattern["renew_available"] = available
                    pattern["renew_price"] = touch
                    if active:
                        counts["additional_renewal_before_resumption"] += 1
            previous = current
            pending = []
            interval_uncertain = False
    for field in (
        "complete_delayed_repair_chains",
        "buy_complete_chains",
        "sell_complete_chains",
        "mixed_clusters_censored",
        "exchange_regressions_censored",
        "noncontinuous_rows_censored",
        "nonobservable_anchor_censored",
        "demand_gap_censored",
        "touch_renewal_before_delayed_repair_censored",
        "anchor_return_resumption_excluded",
    ):
        counts.setdefault(field, 0)
    return {
        "day": day,
        "symbol": symbol,
        "status": "observed",
        "raw_path": str(path),
        "raw_sha256": primitive.digest(path),
        "compressed_bytes": path.stat().st_size,
        "contract_sha256": primitive.digest(contract_path),
        "raw_schema": header,
        "counts": dict(counts),
        "complete_chain_duration_seconds": {
            "minimum": min(durations) if durations else None,
            "maximum": max(durations) if durations else None,
            "mean": sum(durations) / len(durations) if durations else None,
        },
        "chain_examples": examples,
        "final_cluster_right_censored": True,
        "final_incomplete_episode_right_censored": bool(pattern),
    }


def main():
    registration = REFERENCE / "preregistration.yaml"
    preregistration = yaml.safe_load(registration.read_text())
    eligibility_path = REFERENCE / "eligibility-receipt.yaml"
    eligibility = yaml.safe_load(eligibility_path.read_text())
    if eligibility["preregistration_sha256"] != primitive.digest(registration):
        raise RuntimeError("eligibility preregistration identity drift")
    symbols = eligibility["selected_symbols"]
    destination = OUTPUT / "support-receipt.yaml"
    if destination.exists():
        raise RuntimeError("support receipt already exists; no implicit overwrite or rerun")
    script_receipt = {
        "preregistration_sha256": primitive.digest(registration),
        "eligibility_receipt_sha256": primitive.digest(eligibility_path),
        "support_script_sha256": primitive.digest(Path(__file__)),
        "primitive_script_sha256": primitive.digest(DIRECTORY / "census.py"),
        "selected_symbols": symbols,
        "fixed_dates": preregistration["support_dates"],
        "known_side_rule": "CausalTradeSideInference_confidence1_fresh_prior_quote_concordant_feed_or_quote_location_no_fallback",
        "qualification": "conservative_independent_proxy_not_native_module_parity_no_predictive_evidence",
        "earlier_broad_renewal_support_receipt_sha256": primitive.digest(REFERENCE / "support-receipt.yaml"),
        "semantic_correction": "resumption_touch_must_differ_from_original_repaired_anchor_price_under_native_price_equality",
        "earlier_counts_seen": True,
        "gate_universe_calendar_gap_and_known_side_rules_unchanged": True,
    }
    # Bind the implementation before the selected support inputs are read.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "support-implementation.yaml").write_text(yaml.safe_dump(script_receipt, sort_keys=False))
    cells = [census(day, symbol, preregistration["support_max_demand_gap_seconds"] * 1_000_000) for day in preregistration["support_dates"] for symbol in symbols]
    observed = [cell for cell in cells if cell["status"] == "observed"]
    chains = sum(cell["counts"]["complete_delayed_repair_chains"] for cell in observed)
    buys = sum(cell["counts"]["buy_complete_chains"] for cell in observed)
    sells = sum(cell["counts"]["sell_complete_chains"] for cell in observed)
    checks = {
        "fixed_scope_complete": len(symbols) == 2 and len(observed) == 4,
        "per_present_cell_at_least_five_chains": bool(observed) and all(cell["counts"]["complete_delayed_repair_chains"] >= 5 for cell in observed),
        "combined_at_least_twenty": chains >= 20,
        "both_directions": buys > 0 and sells > 0,
    }
    receipt = {
        **script_receipt,
        "schema": "h10-native-primitive-support-distinct-price-correction-v2",
        "labels_read": False,
        "model_fits": 0,
        "native_replay": False,
        "thread_count": 1,
        "checks": checks,
        "support_pass": all(checks.values()),
        "combined_complete_chains": chains,
        "buy_complete_chains": buys,
        "sell_complete_chains": sells,
        "partitions": cells,
    }
    destination.write_text(yaml.safe_dump(receipt, sort_keys=False))
    print("support", checks, "chains", chains, "buy", buys, "sell", sells, flush=True)
    for cell in cells:
        print(cell["day"], cell["symbol"], cell["status"], cell.get("counts"), flush=True)


if __name__ == "__main__":
    main()
