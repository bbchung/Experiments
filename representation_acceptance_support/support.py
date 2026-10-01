"""Causal quote-visit ownership and historical bilateral-testing landmarks.

This is a target-only observable-sequence proxy. Cluster closure, inference and
sampling are explicitly specified; native producer parity has not been proved.
No labels, trainable estimators or sampling-generated source rows enter it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

MICROS = 1_000_000
MAX_BOOK_AGE = 5 * MICROS
RECENT_SUPPORT_AGE = 300 * MICROS
FIXED_CELLS = (("20260119", "2308"), ("20260119", "2317"), ("20260120", "2308"), ("20260120", "2317"))
FIELDS = ("visit_phase", "completed_mark_direction", "accepted_anchor_offset_5ticks", "bilateral_test_strength", "completed_receive_age_300s")


@dataclass(frozen=True)
class Contract:
    limit_up: float
    limit_down: float


@dataclass(frozen=True)
class Event:
    kind: str
    receive: int
    exchange: int
    status: int = 0
    bid: tuple[float, ...] = ()
    ask: tuple[float, ...] = ()
    bid_quantity: tuple[int, ...] = ()
    ask_quantity: tuple[int, ...] = ()
    price: float = 0.0
    quantity: int = 0
    side: int = 0
    raw_receive: int = 0
    wire_exchange: int = 0
    sequence: int = 0
    ordinal: int = 0


@dataclass(frozen=True)
class Origin:
    time: int
    book_time: int = 0
    book_sequence: int = 0


@dataclass(frozen=True)
class Book:
    receive: int
    exchange: int
    bid: tuple[int, ...]
    ask: tuple[int, ...]
    bid_quantity: tuple[int, ...]
    ask_quantity: tuple[int, ...]

    @property
    def pair(self):
        return self.bid[0], self.ask[0]

    @property
    def mid(self):
        return (self.bid[0] + self.ask[0]) // 2


@dataclass
class Visit:
    identity: int
    direction: int
    book: Book
    exchange: int
    available: int
    bid_work: int = 0
    ask_work: int = 0
    bid_test_exchange: int = 0
    ask_test_exchange: int = 0
    completed: bool = False


@dataclass(frozen=True)
class Landmark:
    identity: int
    direction: int
    mid: int
    bid_work: int
    ask_work: int
    bid_reference: int
    ask_reference: int
    available: int
    exchange: int

    @property
    def strength(self):
        return min(self.bid_work / (self.bid_reference + self.bid_work), self.ask_work / (self.ask_reference + self.ask_work))


def half_cent_key(price: float) -> int:
    """Bound one native-price ULP and one scaling ULP, never an absolute eps."""
    if not math.isfinite(price) or price <= 0:
        raise ValueError("physical price must be positive and finite")
    scaled = price * 200.0
    if not math.isfinite(scaled) or scaled >= 2**53:
        raise ValueError("half-cent identity exceeds exact floating domain")
    bound = math.ulp(price) * 200.0 + math.ulp(scaled)
    if bound >= 0.125:
        raise ValueError("half-cent identity has excessive ULP uncertainty")
    nearest = round(scaled)
    if nearest <= 0 or abs(scaled - nearest) > bound:
        raise ValueError("physical price is off the half-cent grid")
    return nearest


def share_price_key(price: float) -> int:
    key = half_cent_key(price)
    step = 2 if key < 2000 else 10 if key < 10000 else 20 if key < 20000 else 100 if key < 100000 else 200 if key < 200000 else 1000
    if key % step:
        raise ValueError("quote violates ordinary-share tick ladder")
    return key


def millitick_coordinate(key: int) -> int:
    if type(key) is not int or not 0 < key <= 2**63 - 1 - 4_000_000:
        raise ValueError("millitick coordinate requires bounded positive integer key")
    bounds = (0, 2000, 10000, 20000, 100000, 200000)
    slopes = (500, 100, 50, 10, 5, 1)
    result = 0
    for index, (lower, slope) in enumerate(zip(bounds, slopes, strict=True)):
        span = max(key - lower, 0)
        if index + 1 < len(bounds):
            span = min(span, bounds[index + 1] - lower)
        result += span * slope
    return result


def normalize_book(event: Event, contract: Contract) -> Book | None:
    if event.kind != "B" or event.status & 7:
        return None
    sides = []
    try:
        for prices, quantities, limit, order in ((event.bid, event.bid_quantity, contract.limit_up, -1), (event.ask, event.ask_quantity, contract.limit_down, 1)):
            if not 1 <= len(prices) <= 5 or len(prices) != len(quantities):
                return None
            if any(type(quantity) is not int or not 0 <= quantity <= 2**31 - 1 for quantity in quantities):
                return None
            prices, quantities = list(prices), list(quantities)
            if prices[0] == 0.0:
                prices[0] = limit
                if len(prices) > 1 and share_price_key(prices[1]) == share_price_key(limit):
                    quantities[0] += quantities.pop(1)
                    prices.pop(1)
                    if quantities[0] > 2**31 - 1:
                        return None  # no proxy claim over overflowing native Qty normalization
            keys = tuple(share_price_key(price) for price in prices)
            if quantities[0] <= 0 or any((right - left) * order <= 0 for left, right in pairwise(keys)):
                return None
            sides.append((keys, tuple(quantities)))
    except ValueError:
        return None
    if sides[0][0][0] >= sides[1][0][0]:
        return None
    return Book(event.receive, event.exchange, sides[0][0], sides[1][0], sides[0][1], sides[1][1])


def known_direction(event: Event, previous: Book | None) -> int:
    """Only the native confidence=1 quote-location subset; no tick-rule fallback."""
    if previous is None or event.status & 7 or event.quantity <= 0:
        return 0
    if not 0 <= event.receive - previous.receive <= MAX_BOOK_AGE or not 0 <= event.exchange - previous.exchange <= MAX_BOOK_AGE:
        return 0
    try:
        price = share_price_key(event.price)
    except ValueError:
        return 0
    location = 1 if price >= previous.ask[0] else -1 if price <= previous.bid[0] else 0
    feed = 1 if event.side == 1 else -1 if event.side == 2 else 0
    if event.side not in (0, 1, 2) or not location or (feed and feed != location):
        return 0
    return feed or location


class SupportState:
    def __init__(self, contract: Contract):
        # The fixed universe contains ordinary shares; reject incompatible limits.
        share_price_key(contract.limit_up)
        share_price_key(contract.limit_down)
        self.contract = contract
        self.previous: Book | None = None
        self.received_book: Book | None = None
        self.visit: Visit | None = None
        self.landmark: Landmark | None = None
        self.completed: list[Landmark] = []
        self.counts = Counter()
        self.cluster: list[Event] = []
        self.cluster_exchange = 0
        self.last_receive = 0
        self.last_exchange = 0
        self.processed_available = 0
        self.processed_exchange = 0
        self.interval_positive_print = False
        self.ever_valid = False
        self.censor_reason = ""
        self.visit_reason = ""
        self.next_identity = 0

    def censor(self, reason: str):
        self.previous = None
        self.visit = None
        self.landmark = None
        self.interval_positive_print = False
        self.censor_reason = reason
        self.visit_reason = ""
        self.counts[f"censored_{reason}"] += 1

    def clear_visit(self, reason: str):
        """Lack of current testing attribution cannot falsify a completed fact."""
        self.previous = None
        self.visit = None
        self.interval_positive_print = False
        self.visit_reason = reason
        self.counts[f"visit_cleared_{reason}"] += 1

    def feed(self, event: Event):
        if type(event.quantity) is not int or not -(2**31) <= event.quantity < 2**31 or type(event.status) is not int or not -(2**31) <= event.status < 2**31:
            raise ValueError("native int32 quantity and status required")
        if type(event.side) is not int or not 0 <= event.side <= 255:
            raise ValueError("native unsigned byte side required")
        if type(event.receive) is not int or type(event.exchange) is not int or not 0 < event.receive < 2**53 or not 0 < event.exchange < 2**53:
            raise ValueError("positive integer clock domains required")
        if event.kind == "T" and event.quantity <= 0:
            # TwseFilter suppresses these trade callbacks. They cannot close an
            # observed exchange cluster or count as reported positive work.
            self.counts["nonpositive_trades_filtered"] += 1
            return
        if event.kind not in ("B", "T"):
            raise ValueError("unsupported source event kind")
        if event.receive < self.last_receive:
            raise ValueError("receive callback regressed; loader clamp must precede state")
        if event.exchange < self.last_exchange:
            self.censor("exchange_regression")
            self.cluster = []
            self.cluster_exchange = 0
            self.received_book = None
        if self.cluster and event.exchange != self.cluster_exchange:
            self._close(event.receive)
            self.cluster = []
        self.cluster_exchange = event.exchange
        self.cluster.append(event)
        self.last_receive, self.last_exchange = event.receive, event.exchange
        if event.status & 7:
            self.censor("noncontinuous")
            self.received_book = None
        elif event.kind == "B":
            earlier = self.received_book
            self.received_book = normalize_book(event, self.contract)
            if self.received_book is None:
                self.censor("invalid_book")
            elif earlier is not None and (event.receive - earlier.receive > MAX_BOOK_AGE or event.exchange - earlier.exchange > MAX_BOOK_AGE):
                self.clear_visit("book_interval_gap")

    def _close(self, available: int):
        self.processed_available = available
        self.processed_exchange = self.cluster_exchange
        self.counts["closed_clusters"] += 1
        temporary = self.previous
        cluster_positive_print = any(event.kind == "T" and event.quantity > 0 for event in self.cluster)
        mapped = []
        signs, feeds = set(), set()
        poisoned, hard_epoch = False, False
        for event in self.cluster:
            if event.status & 7:
                poisoned = True
                hard_epoch = True
                temporary = None
                mapped.append((event, None, 0))
            elif event.kind == "B":
                temporary = normalize_book(event, self.contract)
                poisoned |= temporary is None
                hard_epoch |= temporary is None
                mapped.append((event, temporary, 0))
            else:
                direction = known_direction(event, temporary) if event.quantity > 0 else 0
                if event.quantity > 0:
                    poisoned |= direction == 0
                    if direction:
                        signs.add(direction)
                    if event.side in (1, 2):
                        feeds.add(event.side)
                elif event.quantity < 0:
                    poisoned = True
                mapped.append((event, None, direction))
        if poisoned or len(signs) > 1 or len(feeds) > 1:
            if hard_epoch:
                self.censor("hard_epoch_cluster")
            else:
                self.clear_visit("ambiguous_cluster")
            return
        for event, book, direction in mapped:
            if event.kind == "B":
                self._book(book, available, cluster_positive_print)
            elif event.quantity > 0:
                self.interval_positive_print = True
                self._trade(event, direction, available)

    def _book(self, book: Book, available: int, cluster_positive_print: bool):
        previous = self.previous
        self.ever_valid = True
        self.censor_reason = ""
        self.visit_reason = ""
        if previous is not None:
            if not 0 <= book.receive - previous.receive <= MAX_BOOK_AGE or not 0 <= book.exchange - previous.exchange <= MAX_BOOK_AGE:
                self.clear_visit("book_interval_gap")
                previous = None
            if self.visit is not None and book.pair != self.visit.book.pair:
                self.visit = None
                self.counts["visits_ended_at_pair_change"] += 1
            bid_change = book.bid[0] - previous.bid[0] if previous else 0
            ask_change = book.ask[0] - previous.ask[0] if previous else 0
            if previous and bid_change * ask_change > 0 and not self.interval_positive_print and not cluster_positive_print:
                self.next_identity += 1
                self.visit = Visit(self.next_identity, 1 if bid_change > 0 else -1, book, book.exchange, available)
                self.counts["quote_led_visits"] += 1
        self.previous = book
        self.interval_positive_print = False

    def _trade(self, event: Event, direction: int, available: int):
        visit = self.visit
        if visit is None or visit.completed or event.exchange <= visit.exchange:
            return
        if self.previous is None or self.previous.pair != visit.book.pair:
            return
        price = share_price_key(event.price)
        if direction == -1 and price == visit.book.bid[0]:
            visit.bid_work += event.quantity
            visit.bid_test_exchange = event.exchange
        elif direction == 1 and price == visit.book.ask[0]:
            visit.ask_work += event.quantity
            visit.ask_test_exchange = event.exchange
        else:
            return  # known through-touch print cannot test the exact offered mark
        if visit.bid_work > 2**63 - 1 or visit.ask_work > 2**63 - 1:
            self.censor("quantity_overflow")
            return
        if visit.bid_work > 0 and visit.ask_work > 0 and visit.bid_test_exchange != visit.ask_test_exchange:
            landmark = Landmark(
                visit.identity,
                visit.direction,
                visit.book.mid,
                visit.bid_work,
                visit.ask_work,
                visit.book.bid_quantity[0],
                visit.book.ask_quantity[0],
                available,
                event.exchange,
            )
            visit.completed = True
            self.landmark = landmark
            self.completed.append(landmark)
            self.counts["completed_landmarks"] += 1

    def snapshot(self, origin: Origin) -> dict:
        if origin.time <= self.last_receive:
            raise ValueError("snapshot must precede callbacks at equal receive time")
        book = self.received_book
        phase = "warmup" if not self.ever_valid else "inactive"
        if self.censor_reason:
            phase = "censored"
        elif book is None or not 0 <= origin.time - book.receive <= MAX_BOOK_AGE:
            phase = "stale" if self.ever_valid else "warmup"
        elif self.visit_reason:
            phase = "untestable"
        elif self.visit is not None and book.pair == self.visit.book.pair:
            visit = self.visit
            phase = "bilateral" if visit.completed else "bid_tested" if visit.bid_work else "ask_tested" if visit.ask_work else "provisional"
        missing = -math.inf if not self.ever_valid else math.nan
        record = {
            "SampleTime": origin.time,
            "SampleBookTime": origin.book_time,
            "SampleBookSeq": origin.book_sequence,
            "visit_phase": phase,
            "completed_mark_direction": missing,
            "accepted_anchor_offset_5ticks": missing,
            "bilateral_test_strength": missing,
            "completed_receive_age_300s": missing,
            "landmark_id": 0,
            "landmark_available": 0,
            "landmark_exchange": 0,
            "processed_available": self.processed_available,
            "processed_exchange": self.processed_exchange,
            "censor_reason": self.censor_reason,
            "visit_reason": self.visit_reason,
        }
        mark = self.landmark
        if mark is not None and phase not in ("warmup", "censored", "stale"):
            if not 0 < mark.available < origin.time:
                raise ValueError("completed landmark was not available before snapshot callback")
            delta = 0 if book.mid == mark.mid else millitick_coordinate(book.mid) - millitick_coordinate(mark.mid)
            record.update(
                completed_mark_direction=mark.direction,
                accepted_anchor_offset_5ticks=delta / 5000.0,
                bilateral_test_strength=mark.strength,
                completed_receive_age_300s=(origin.time - mark.available) / RECENT_SUPPORT_AGE,
                landmark_id=mark.identity,
                landmark_available=mark.available,
                landmark_exchange=mark.exchange,
            )
        return record


def replay(events: Iterable[Event], origins: Iterable[Origin], contract: Contract):
    """Snapshot before every callback with clamped R >= S; final E group unclosed."""
    origins = list(origins)
    if not origins:
        raise ValueError("original source origins required")
    if any(
        type(origin.time) is not int
        or not 0 < origin.time < 2**53
        or type(origin.book_time) is not int
        or not 0 <= origin.book_time <= origin.time
        or type(origin.book_sequence) is not int
        or not 0 <= origin.book_sequence < 2**53
        for origin in origins
    ):
        raise ValueError("origin times must be positive integer source keys")
    if any(right.time <= left.time for left, right in pairwise(origins)):
        raise ValueError("origin keys must be strictly increasing")
    state, snapshots, index = SupportState(contract), [], 0
    for event in events:
        while index < len(origins) and origins[index].time <= event.receive:
            snapshots.append(state.snapshot(origins[index]))
            index += 1
        if index == len(origins):
            break  # no source suffix after the final original origin is needed
        state.feed(event)
    # EOF is not an observed closure marker. Never process or invent a final row.
    while index < len(origins):
        snapshots.append(state.snapshot(origins[index]))
        index += 1
    return state, snapshots


def cell_summary(day: str, symbol: str, state: SupportState, snapshots: list[dict]) -> dict:
    if (day, symbol) not in FIXED_CELLS or not snapshots:
        raise ValueError("nonfixed cell or missing original origins")
    start, end = snapshots[0]["SampleTime"], snapshots[-1]["SampleTime"]
    completions = [mark for mark in state.completed if start <= mark.available <= end]
    recent = {row["landmark_id"] for row in snapshots if row["landmark_id"] and 0 < row["SampleTime"] - row["landmark_available"] <= RECENT_SUPPORT_AGE}
    # A completion from before the declared session cannot meet this session gate.
    recent &= {mark.identity for mark in completions}
    phases = Counter(row["visit_phase"] for row in snapshots)
    return {
        "day": day,
        "symbol": symbol,
        "status": "observed_proxy",
        "completions": len(completions),
        "sampled_distinct_recent_landmarks": len(recent),
        "up_completions": sum(mark.direction == 1 for mark in completions),
        "down_completions": sum(mark.direction == -1 for mark in completions),
        "contrast_origins": sum(phases[name] for name in ("provisional", "bid_tested", "ask_tested")),
        "origin_count": len(snapshots),
        "phase_counts": dict(phases),
        "event_counts": dict(state.counts),
        "open_final_cluster_rows": len(state.cluster),
    }


def evaluate_gates(cells: list[dict]) -> dict:
    """Fixed four-cell source-support rule. No predictive admission is performed."""
    identities = [(cell["day"], cell["symbol"]) for cell in cells]
    if len(set(identities)) != len(identities) or set(identities) != set(FIXED_CELLS):
        raise ValueError("exact fixed cells required; no replacement or duplicate cell")
    observed = [cell for cell in cells if cell["status"] == "observed_proxy"]
    if any(cell["status"] not in ("observed_proxy", "missing_skip_no_replacement", "missing_contract_skip_no_replacement") for cell in cells):
        raise ValueError("support cell status must be observed or explicitly missing")
    for cell in observed:
        for field in ("completions", "sampled_distinct_recent_landmarks", "up_completions", "down_completions", "contrast_origins"):
            if type(cell.get(field)) is not int or cell[field] < 0:
                raise ValueError("support counts require observed nonnegative integers")
    checks = {
        "at_least_one_fixed_cell_observed": bool(observed),
        "each_present_cell_five_completions": all(cell["completions"] >= 5 for cell in observed),
        "each_present_cell_five_sampled_distinct_recent_landmarks": all(cell["sampled_distinct_recent_landmarks"] >= 5 for cell in observed),
        "five_completions_each_direction": all(sum(cell[f"{direction}_completions"] for cell in observed) >= 5 for direction in ("up", "down")),
        "twenty_provisional_or_unilateral_origins": sum(cell["contrast_origins"] for cell in observed) >= 20,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "all_fixed_cells_observed": len(observed) == len(FIXED_CELLS),
        "observed_cell_count": len(observed),
        "missing_cells": [{"day": cell["day"], "symbol": cell["symbol"], "status": cell["status"]} for cell in cells if cell["status"] != "observed_proxy"],
        "native_support_proved": False,
        "predictive_evidence": False,
    }
