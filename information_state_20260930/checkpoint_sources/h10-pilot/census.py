"""Bounded TRAIN-only raw event census; no labels, features, model or replay.

This is a conservative observable-sequence diagnostic, not native FE parity.
Group completion makes all members of an exchange-time cluster observable.
Known demand requires a feed side and matching, fresh, earlier touch; mixed
clusters, unknown trades and nonobservable price levels censor the chain.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
import zstandard

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
DAYS = ("20260119", "20260120")
SYMBOLS = ("2337", "2609")
SOURCE = Path("/mnt/data0/marketdata/tse/kgi/stock")
CONTRACTS = Path("/mnt/data0/contract/tse/stock")
EPSILON = 1e-8  # sdk/math/compare.hpp, price equality only; never a ratio floor.
FRESH_MICROS = 5_000_000
SEQUENCE_CAP_MICROS = 300_000_000
SOURCE_FILES = (
    "src/oms/modules/md/trade_book_md/trade_book_md.cpp",
    "src/oms/modules/tw/twse_filter/twse_filter.cpp",
    "src/oms/modules/feature/observable_book.h",
    "src/oms/modules/feature/order_flow/trade_flow/trade_side_inference.h",
    "src/oms/modules/feature/microstructure/liquidity/liquidity_event_classifier.h",
    "src/oms/modules/feature/microstructure/liquidity/pressure_response_state/pressure_response_state.cpp",
    "src/oms/modules/feature/microstructure/liquidity/absorption_release_lifecycle/absorption_release_lifecycle.cpp",
    "src/sdk/math/compare.hpp",
    "src/sdk/types/coco_type.h",
    "src/msg/md_msg.h",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eq(left: float, right: float) -> bool:
    return abs(left - right) <= EPSILON


def schema_rows(path: Path):
    with path.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as decompressed, io.TextIOWrapper(decompressed) as stream:
        reader = csv.DictReader(stream)
        yield reader.fieldnames
        previous_receive = 0
        for ordinal, row in enumerate(reader):
            original = int(row["Timestamp"])
            receive = max(original, previous_receive)  # CsvLoader receive-clock contract.
            previous_receive = receive
            row["_ordinal"] = ordinal
            row["_receive"] = receive
            row["_receive_regressed"] = original < receive
            exchange = int(row["ExchangeTime"])
            row["_event"] = exchange if exchange > 0 else receive
            yield row


def closed_clusters(rows):
    """Close a cluster only when the next distinct event time is received.

    The next row is an observed closure marker, not an earlier availability
    assumption. The file's final cluster stays right censored.
    """
    cluster = []
    event_time = None
    for row in rows:
        if cluster and row["_event"] != event_time:
            yield event_time, cluster, row["_receive"]
            cluster = []
        event_time = row["_event"]
        cluster.append(row)


@dataclass
class Book:
    receive: int
    exchange: int
    bid: list[float]
    ask: list[float]
    bid_quantity: list[int]
    ask_quantity: list[int]

    def prices(self, direction: int):
        return self.ask if direction == 1 else self.bid

    def quantities(self, direction: int):
        return self.ask_quantity if direction == 1 else self.bid_quantity

    def quantity_at(self, direction: int, price: float):
        prices = self.prices(direction)
        quantities = self.quantities(direction)
        for level, candidate in enumerate(prices):
            if eq(candidate, price):
                return quantities[level]
        # ObservableBook::observes_price: no invented depletion beyond full L5.
        observable = len(prices) < 5 or (price - prices[-1] <= EPSILON if direction == 1 else prices[-1] - price <= EPSILON)
        return 0 if observable else None


def normalized_book(row, contract) -> Book | None:
    if int(row["StatusMask"], 16) & 7:
        return None
    sides = []
    for side, limit_key in (("Bid", "limit_up"), ("Ask", "limit_down")):
        depth = int(row[side + "Depth"])
        if depth < 1 or depth > 5:
            return None
        prices = [float(row[f"{side}Price{i}"]) for i in range(1, depth + 1)]
        quantities = [int(row[f"{side}Vol{i}"]) for i in range(1, depth + 1)]
        if prices[0] == 0:
            prices[0] = float(contract[limit_key])
            if depth > 1 and eq(prices[1], prices[0]):
                quantities[0] += quantities[1]
                del prices[1]
                del quantities[1]
        if any(not math.isfinite(price) or price <= 0 for price in prices) or any(quantity < 0 for quantity in quantities) or quantities[0] <= 0:
            return None
        sides.append((prices, quantities))
    bid, bid_quantity = sides[0]
    ask, ask_quantity = sides[1]
    if ask[0] - bid[0] <= EPSILON:
        return None
    return Book(row["_receive"], row["_event"], bid, ask, bid_quantity, ask_quantity)


def census(day: str, symbol: str):
    path = SOURCE / symbol / (day + ".csv.zst")
    contract_path = CONTRACTS / (day + ".csv")
    if not path.exists() or not contract_path.exists():
        return {"day": day, "symbol": symbol, "status": "missing_skip"}, []
    with contract_path.open() as handle:
        contract = next((row for row in csv.DictReader(handle) if row["symbol"] == symbol), None)
    if contract is None:
        return {"day": day, "symbol": symbol, "status": "contract_missing_skip"}, []
    epoch = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC).timestamp() * 1e6)
    start = epoch + 70 * 60 * 1_000_000  # 09:10 Asia/Taipei = 01:10 UTC.
    end = epoch + 5 * 60 * 60 * 1_000_000  # 13:00 Asia/Taipei.
    counts = Counter()
    previous = None
    pending = []
    pending_uncertain = False
    pattern = None
    last_exchange = 0
    samples = []
    generator = schema_rows(path)
    schema = next(generator)
    for event_time, cluster, available in closed_clusters(generator):
        active = start <= available < end
        if active:
            counts["exchange_clusters"] += 1
            counts["rows"] += len(cluster)
            counts["receive_regressions_clamped"] += sum(row["_receive_regressed"] for row in cluster)
        sides = {int(row["Side"]) for row in cluster if row["Type"] == "T" and int(row["TradeVolume"]) > 0 and int(row["StatusMask"], 16) == 0}
        mixed_cluster = 1 in sides and 2 in sides
        backward = event_time < last_exchange
        last_exchange = event_time
        if mixed_cluster or backward:
            pending_uncertain = True
            pattern = None
            if active:
                counts["mixed_exchange_clusters" if mixed_cluster else "exchange_regressions"] += 1
        for row in cluster:
            status = int(row["StatusMask"], 16)
            if status & 7:
                previous = None
                pending = []
                pending_uncertain = False
                pattern = None
                if active:
                    counts["noncontinuous_rows"] += 1
                continue
            if row["Type"] == "T":
                quantity = int(row["TradeVolume"])
                price = float(row["Price"])
                side = int(row["Side"])
                if active:
                    counts["continuous_trades"] += 1
                    counts["continuous_trade_quantity"] += max(quantity, 0)
                fresh = previous is not None and 0 <= row["_receive"] - previous.receive <= FRESH_MICROS and 0 <= event_time - previous.exchange <= FRESH_MICROS
                sign = 1 if side == 1 else -1 if side == 2 else 0
                location_sign = 0
                if fresh:
                    location_sign = 1 if price >= previous.ask[0] else -1 if price <= previous.bid[0] else 0
                known = quantity > 0 and math.isfinite(price) and price > 0 and fresh and sign != 0 and sign == location_sign and not mixed_cluster and not backward
                if not known:
                    pending_uncertain = True
                    pattern = None
                    if active:
                        counts["uncertain_trades"] += 1
                        counts["uncertain_trade_quantity"] += max(quantity, 0)
                    continue
                if active:
                    counts["strict_known_trades"] += 1
                    counts["strict_known_trade_quantity"] += quantity
                pending.append((sign, price, quantity, event_time))
                continue
            if row["Type"] != "B":
                if active:
                    counts["unsupported_type"] += 1
                pending_uncertain = True
                pattern = None
                continue
            if active:
                counts["continuous_books"] += 1
            current = normalized_book(row, contract)
            if current is None or backward:
                if active:
                    counts["invalid_or_non_strict_books"] += 1
                previous = None
                pending = []
                pending_uncertain = False
                pattern = None
                continue
            if active:
                counts["valid_strict_books"] += 1
            if previous is not None:
                distinct = {item[0] for item in pending}
                fresh_interval = 0 <= current.receive - previous.receive <= FRESH_MICROS and 0 <= current.exchange - previous.exchange <= FRESH_MICROS
                if pending_uncertain or len(distinct) > 1 or not fresh_interval:
                    if active:
                        counts["uncertain_or_stale_book_intervals"] += 1
                    pattern = None
                else:
                    if pattern and available - pattern["start"] > SEQUENCE_CAP_MICROS:
                        if active:
                            counts["sequence_cap_censored"] += 1
                        pattern = None
                    renewed_now = False
                    if pattern and pattern["phase"] == "contained" and event_time > pattern["contain_exchange"]:
                        touch = current.prices(pattern["sign"])[0]
                        if not eq(touch, pattern["anchor_price"]):
                            pattern["phase"] = "renewed"
                            pattern["renew_exchange"] = event_time
                            pattern["renew_available"] = available
                            pattern["renew_price"] = touch
                            pattern["renew_adverse"] = pattern["sign"] * (touch - pattern["anchor_price"]) < -EPSILON
                            renewed_now = True
                            if active:
                                counts["contained_then_price_renewal"] += 1
                    if pending:
                        sign = next(iter(distinct))
                        if pattern and pattern["sign"] != sign:
                            if active:
                                counts["opposite_challenge_censored"] += 1
                            pattern = None
                        old_touch = previous.prices(sign)[0]
                        exact = [item for item in pending if eq(item[1], old_touch)]
                        quantity = sum(item[2] for item in exact)
                        remaining = current.quantity_at(sign, old_touch)
                        if quantity > 0 and remaining is not None:
                            if active:
                                counts["observable_touch_challenges"] += 1
                                counts["challenge_quantity"] += quantity
                            if (
                                pattern
                                and pattern["phase"] == "renewed"
                                and not renewed_now
                                and event_time > pattern["renew_exchange"]
                                and min(item[3] for item in exact) > pattern["renew_exchange"]
                                and eq(old_touch, pattern["renew_price"])
                            ):
                                if active:
                                    counts["strict_sequence_resumptions"] += 1
                                    counts["adverse_renewal_resumptions"] += int(pattern["renew_adverse"])
                                    counts["sequence_within_60s"] += int(available - pattern["start"] <= 60_000_000)
                                    if len(samples) < 8:
                                        samples.append(
                                            {**pattern, "resume_available": available, "resume_exchange": event_time, "resume_price": old_touch, "resume_quantity": quantity}
                                        )
                                pattern = None
                            contained = eq(current.prices(sign)[0], old_touch) and remaining > 0
                            if contained:
                                if active:
                                    counts["contained_touch_challenges"] += 1
                                    supply = remaining - previous.quantities(sign)[0] + quantity
                                    counts["containments_with_positive_net_supply"] += int(supply > 0)
                                if pattern is None:
                                    pattern = {
                                        "sign": sign,
                                        "phase": "contained",
                                        "start": available,
                                        "contain_exchange": event_time,
                                        "anchor_price": old_touch,
                                    }
                        elif quantity > 0:
                            if active:
                                counts["nonobservable_touch_challenges"] += 1
                            pattern = None
            previous = current
            pending = []
            pending_uncertain = mixed_cluster
    total_quantity = counts["continuous_trade_quantity"]
    result = {
        "day": day,
        "symbol": symbol,
        "status": "observed",
        "raw_file": str(path),
        "raw_compressed_bytes": path.stat().st_size,
        "raw_sha256": digest(path),
        "contract_file": str(contract_path),
        "contract_sha256": digest(contract_path),
        "contract_record": contract,
        "schema": schema,
        "counts": dict(counts),
        "strict_known_quantity_share": counts["strict_known_trade_quantity"] / total_quantity if total_quantity else None,
        "sequence_examples": samples,
    }
    return result, schema


def main():
    results = [census(day, symbol)[0] for day in DAYS for symbol in SYMBOLS]
    receipt = {
        "schema": "h10-train-native-primitive-census-v1",
        "qualification": "support_only_conservative_proxy_not_native_module_parity_or_predictive_evidence",
        "labels_read": False,
        "model_fits": 0,
        "native_replay": False,
        "thread_count": 1,
        "fixed_dates": list(DAYS),
        "fixed_symbols": list(SYMBOLS),
        "clock": "091000_to_130000_Asia_Taipei_group_completion_available_time",
        "cluster_availability": "first_next_distinct_exchange_time_row_receive_time_final_cluster_right_censored",
        "strict_known_rule": "feed_side_and_concordant_fresh_previous_quote_location_only_no_tick_rule_or_current_quote_inference",
        "mixed_cluster_rule": "entire_exchange_time_cluster_uncertain_until_complete_no_side_order_invented",
        "renewal_rule": "later_distinct_exchange_cluster_opposing_touch_changes_after_observed_containment",
        "resumption_rule": "later_distinct_exchange_cluster_known_same_direction_trade_at_already_renewed_previous_touch_mutually_observable",
        "max_sequence_seconds": 300,
        "missing_policy": "skip_without_replacement",
        "source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES},
        "script_sha256": digest(Path(__file__)),
        "zstandard_version": zstandard.__version__,
        "partitions": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "receipt.yaml").write_text(yaml.safe_dump(receipt, sort_keys=False), encoding="utf-8")
    for result in results:
        print(result["day"], result["symbol"], result["status"], result.get("counts"), result.get("strict_known_quantity_share"))


if __name__ == "__main__":
    main()
