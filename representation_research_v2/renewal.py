"""H9: sampled-price renewal arrival, conditional marks and signed episodes.

Only physical TWSE listed-share mids and origin keys enter this representation.
Labels are neither accepted nor read. The reference uses strictly earlier date
slots; a gap breaks the observed price chain and left-censors its first dwell.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MICROS = 1_000_000
DAY_MICROS = 86_400 * MICROS
MILLITICKS = 1000
FIVE_TICKS = 5 * MILLITICKS
AGE_EDGES_SECONDS = np.array([10, 30, 60, 120, 300], dtype=np.int64)
FIELDS = (
    "renewal_observed",
    "prior_clock_renewal_probability",
    "prior_age_renewal_probability",
    "prior_dwell_survival",
    "current_day_renewal_frequency_excess",
    "prior_conditional_step_size_5ticks",
    "last_mark_size_relative",
    "last_mark_direction",
    "signed_run_progress_5ticks",
    "signed_run_persistence",
    "prior_expected_mark_direction",
    "prior_reference_coverage",
)


def half_cent_keys(prices):
    """Recover physical 0.005 TWD keys, accepting only bounded float64 ULP error.

    One native-price ULP plus one scaling ULP bounds supported midpoint/scaling
    rounding. Require this uncertainty below one eighth of a grid unit before
    rounding, leaving a substantial margin from the half-unit rounding boundary.
    This is not an absolute epsilon and never repairs an off-grid physical price.
    """
    prices = np.asarray(prices)
    if prices.dtype != np.float64:
        raise ValueError("OriginMidPrice must retain native float64 precision")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("OriginMidPrice must be positive and finite")
    scaled = prices * 200.0
    if not np.isfinite(scaled).all() or (scaled >= 2**53).any():
        raise ValueError("physical half-cent keys cannot be recovered exactly")
    tolerance = np.spacing(prices) * 200.0 + np.spacing(scaled)
    if (tolerance >= 0.125).any():
        raise ValueError("physical half-cent keys have excessive ULP uncertainty")
    nearest = np.rint(scaled)
    residual = np.abs(scaled - nearest)
    if (residual > tolerance).any():
        raise ValueError("OriginMidPrice violates the physical TWSE half-cent grid")
    return nearest.astype(np.int64)


def millitick_coordinate(keys):
    """Exact rational native share-ladder coordinate, expressed in 1/1000 tick.

    Native price bands end at 10/50/100/500/1000 TWD, with tick sizes
    .01/.05/.1/.5/1/5. In half-cent units their slopes are exactly integer
    500/100/50/10/5/1 milliticks; the last native band extends upward.
    """
    keys = np.asarray(keys)
    if not np.issubdtype(keys.dtype, np.integer) or (keys <= 0).any():
        raise ValueError("millitick coordinate requires positive integer half-cent keys")
    # The largest slope applies below the first boundary; beyond 1000 TWD the
    # slope is one. This tighter bound also protects intermediate arithmetic.
    if (keys > np.iinfo(np.int64).max - 4_000_000).any():
        raise ValueError("millitick coordinate exceeds its exact integer domain")
    bounds = (0, 2000, 10000, 20000, 100000, 200000)
    slopes = (500, 100, 50, 10, 5, 1)
    result = np.zeros(keys.shape, dtype=np.int64)
    for index, (lower, slope) in enumerate(zip(bounds, slopes, strict=True)):
        upper = bounds[index + 1] if index + 1 < len(bounds) else None
        span = np.maximum(keys - lower, 0)
        if upper is not None:
            span = np.minimum(span, upper - lower)
        result += span * slope
    return result


@dataclass
class _Stats:
    clocks: int
    clock_risk: np.ndarray = field(init=False)
    clock_events: np.ndarray = field(init=False)
    age_risk: np.ndarray = field(init=False)
    age_events: np.ndarray = field(init=False)
    mark_count: np.ndarray = field(init=False)
    mark_sum: np.ndarray = field(init=False)
    continuation_count: np.ndarray = field(init=False)
    continuation_same: np.ndarray = field(init=False)
    dwells: list = field(default_factory=list)

    def __post_init__(self):
        self.clock_risk = np.zeros(self.clocks, dtype=np.int64)
        self.clock_events = np.zeros(self.clocks, dtype=np.int64)
        self.age_risk = np.zeros((self.clocks, len(AGE_EDGES_SECONDS) + 1), dtype=np.int64)
        self.age_events = np.zeros_like(self.age_risk)
        self.mark_count = np.zeros(self.clocks, dtype=np.int64)
        self.mark_sum = np.zeros(self.clocks, dtype=np.int64)
        self.continuation_count = np.zeros((self.clocks, 3), dtype=np.int64)
        self.continuation_same = np.zeros_like(self.continuation_count)


def _ratio(numerator, denominator, valid):
    result = np.full(np.shape(denominator), np.nan, dtype=np.float64)
    valid = valid & (denominator > 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _reference(history, config, clocks):
    empty = _Stats(clocks)
    arrays = ("clock_risk", "clock_events", "age_risk", "age_events", "mark_count", "mark_sum", "continuation_count", "continuation_same")
    totals = {name: sum((getattr(stats, name) for _, stats in history), np.zeros_like(getattr(empty, name))) for name in arrays}
    days = {
        name: sum((getattr(stats, name) > 0 for _, stats in history), np.zeros_like(getattr(empty, name)))
        for name in ("clock_risk", "age_risk", "mark_count", "continuation_count")
    }
    minimum_days = config["minimum_days"]
    clock_p = _ratio(totals["clock_events"], totals["clock_risk"], (days["clock_risk"] >= minimum_days) & (totals["clock_risk"] >= config["minimum_risk_intervals"]))
    age_p = _ratio(totals["age_events"], totals["age_risk"], (days["age_risk"] >= minimum_days) & (totals["age_risk"] >= config["minimum_risk_intervals"]))
    mark_mean = _ratio(totals["mark_sum"], totals["mark_count"], (days["mark_count"] >= minimum_days) & (totals["mark_count"] >= config["minimum_marks"]))
    continuation = _ratio(
        totals["continuation_same"],
        totals["continuation_count"],
        (days["continuation_count"] >= minimum_days) & (totals["continuation_count"] >= config["minimum_marks"]),
    )
    # Discrete Kaplan-Meier includes fully observed right-censored dwells.
    # The initial left-censored dwell is never counted as a complete episode.
    survival = {}
    for clock in range(clocks):
        observations = [(day, duration, completed) for day, stats in history for bucket, duration, completed in stats.dwells if bucket == clock]
        if len({day for day, _, _ in observations}) < minimum_days or len(observations) < config["minimum_marks"]:
            continue
        durations = np.array([duration for _, duration, _ in observations], dtype=np.int64)
        completed = np.array([complete for _, _, complete in observations], dtype=bool)
        total_hist = np.bincount(durations)
        event_hist = np.bincount(durations[completed], minlength=len(total_hist))
        at_risk = np.cumsum(total_hist[::-1])[::-1]
        hazard = _ratio(event_hist, at_risk, at_risk > 0)
        curve = np.cumprod(1.0 - hazard)
        curve[at_risk < config["minimum_marks"]] = np.nan
        survival[clock] = curve
    return {
        "clock_probability": clock_p,
        "age_probability": age_p,
        "mark_mean_milliticks": mark_mean,
        "continuation": continuation,
        "coverage": days["clock_risk"] / float(config["history_days"]),
        "survival": survival,
    }


def _day(times, coordinates, config, reference):
    interval = config["native_interval_seconds"] * MICROS
    clock_seconds = config["clock_bucket_seconds"]
    clocks = 86400 // clock_seconds
    clock = ((times % DAY_MICROS) // (clock_seconds * MICROS)).astype(np.int64)
    values = np.full((len(times), len(FIELDS)), np.nan)
    column = {name: index for index, name in enumerate(FIELDS)}
    stats = _Stats(clocks)
    last_renewal = -1
    last_direction = 0
    last_mark = 0
    run_length = 0
    run_anchor = 0
    day_renewals = 0
    supported_risk = 0
    supported_events = 0
    expected_events = 0.0

    def censor(previous):
        if last_renewal >= 0:
            duration = int((times[previous] - times[last_renewal]) // interval)
            stats.dwells.append((int(clock[last_renewal]), duration, False))

    for index in range(len(times)):
        bucket = clock[index]
        known_pair = index > 0 and times[index] - times[index - 1] == interval
        if not known_pair:
            if index > 0:
                censor(index - 1)
            last_renewal = -1
            last_direction = 0
            last_mark = 0
            run_length = 0
        else:
            previous_bucket = clock[index - 1]
            move = int(coordinates[index] - coordinates[index - 1])
            renewed = move != 0
            stats.clock_risk[previous_bucket] += 1
            stats.clock_events[previous_bucket] += int(renewed)
            values[index, column["renewal_observed"]] = float(renewed)
            if last_renewal >= 0:
                previous_age_seconds = int((times[index - 1] - times[last_renewal]) // MICROS)
                age_bucket = np.searchsorted(AGE_EDGES_SECONDS, previous_age_seconds, side="right")
                stats.age_risk[previous_bucket, age_bucket] += 1
                stats.age_events[previous_bucket, age_bucket] += int(renewed)
            prior_probability = reference["clock_probability"][previous_bucket]
            if np.isfinite(prior_probability):
                supported_risk += 1
                supported_events += int(renewed)
                expected_events += prior_probability
            if renewed:
                direction = 1 if move > 0 else -1
                if last_renewal >= 0:
                    duration = int((times[index] - times[last_renewal]) // interval)
                    stats.dwells.append((int(clock[last_renewal]), duration, True))
                    prior_run_bucket = min(run_length, 3) - 1
                    stats.continuation_count[clock[last_renewal], prior_run_bucket] += 1
                    stats.continuation_same[clock[last_renewal], prior_run_bucket] += int(direction == last_direction)
                if direction == last_direction:
                    run_length += 1
                else:
                    run_length = 1
                    run_anchor = index - 1
                day_renewals += 1
                last_renewal = index
                last_direction = direction
                last_mark = abs(move)
                stats.mark_count[bucket] += 1
                stats.mark_sum[bucket] += last_mark
        values[index, column["prior_clock_renewal_probability"]] = reference["clock_probability"][bucket]
        values[index, column["prior_reference_coverage"]] = reference["coverage"][bucket]
        mark_mean = reference["mark_mean_milliticks"][bucket]
        values[index, column["prior_conditional_step_size_5ticks"]] = mark_mean / FIVE_TICKS
        if supported_risk:
            values[index, column["current_day_renewal_frequency_excess"]] = (supported_events - expected_events) / supported_risk
        if last_renewal >= 0:
            age_seconds = int((times[index] - times[last_renewal]) // MICROS)
            age_bucket = np.searchsorted(AGE_EDGES_SECONDS, age_seconds, side="right")
            values[index, column["prior_age_renewal_probability"]] = reference["age_probability"][bucket, age_bucket]
            curve = reference["survival"].get(int(clock[last_renewal]))
            age_steps = age_seconds // config["native_interval_seconds"]
            if curve is not None and age_steps < len(curve):
                values[index, column["prior_dwell_survival"]] = curve[age_steps]
            if np.isfinite(mark_mean):
                # One exact whole tick is a declared price-resolution floor.
                values[index, column["last_mark_size_relative"]] = last_mark / max(mark_mean, MILLITICKS)
            values[index, column["last_mark_direction"]] = last_direction
            values[index, column["signed_run_progress_5ticks"]] = (int(coordinates[index]) - int(coordinates[run_anchor])) / FIVE_TICKS
            values[index, column["signed_run_persistence"]] = last_direction * run_length / day_renewals
            # The reference conditions on the previous renewal's clock, not
            # the clock of a later quiet origin that reads the held mark.
            continuation = reference["continuation"][clock[last_renewal], min(run_length, 3) - 1]
            values[index, column["prior_expected_mark_direction"]] = last_direction * (2.0 * continuation - 1.0)
    if len(times):
        censor(len(times) - 1)
    return values, stats


def _configuration(profile, days):
    hypotheses = profile.get("hypotheses", {})
    supplied = dict(hypotheses.get("renewal", {}))
    config = {
        "market_tick_domain": "twse_listed_share",
        "native_interval_seconds": 10,
        "clock_bucket_seconds": 1800,
        "history_days": 20,
        "minimum_days": 5,
        "minimum_risk_intervals": 32,
        "minimum_marks": 20,
        **supplied,
    }
    if config["market_tick_domain"] != "twse_listed_share" or config["native_interval_seconds"] != 10:
        raise ValueError("H9 requires the declared TWSE share ladder and original 10 s grid")
    for name in ("clock_bucket_seconds", "history_days", "minimum_days", "minimum_risk_intervals", "minimum_marks"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if 86400 % config["clock_bucket_seconds"] or config["clock_bucket_seconds"] % config["native_interval_seconds"]:
        raise ValueError("clock buckets must divide the day and contain whole native intervals")
    if config["minimum_days"] > config["history_days"]:
        raise ValueError("minimum_days exceeds the reference date slots")
    calendar = config.get("calendar_days", hypotheses.get("session_prior", {}).get("calendar_days"))
    if not calendar:
        raise ValueError("declare calendar_days; missing slots cannot be compressed")
    calendar = sorted({str(day) for day in calendar})
    if any(len(day) != 8 or not day.isdigit() or day < "20260101" for day in calendar) or not set(days).issubset(calendar):
        raise ValueError("calendar must contain every observed 2026 date")
    config["calendar_days"] = calendar
    return config


def build(frame, profile):
    """Return the twelve H9 fields aligned to the original row order/index.

    Extra columns, including native endpoint labels, are never read. All rows
    survive feature availability. Prior reference updates occur after a whole
    date has been processed, including unlabeled held-symbol observations.
    """
    required = ("day", "symbol", "SampleTime", "OriginMidPrice")
    if any(name not in frame for name in required):
        raise ValueError("H9 needs day, symbol, SampleTime and native OriginMidPrice")
    if frame[list(required)].isna().any().any():
        raise ValueError("H9 origin keys and physical mids must not be missing")
    days = frame.day.astype(str).to_numpy()
    symbols = frame.symbol.astype(str).to_numpy()
    times = frame.SampleTime.to_numpy()
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("SampleTime requires integer native microseconds")
    keys = pd.DataFrame({"day": days, "symbol": symbols, "SampleTime": times})
    if keys.duplicated().any():
        raise ValueError("duplicate H9 origin keys")
    config = _configuration(profile, days)
    mids = frame.OriginMidPrice.to_numpy()
    physical = half_cent_keys(mids)
    coordinates = millitick_coordinate(physical)
    result = np.full((len(frame), len(FIELDS)), np.nan)
    clocks = 86400 // config["clock_bucket_seconds"]
    for symbol_positions in keys.groupby("symbol", sort=True).indices.values():
        symbol_positions = np.asarray(symbol_positions)
        grouped = keys.iloc[symbol_positions].groupby("day", sort=True).indices
        history = deque(maxlen=config["history_days"])
        for day in config["calendar_days"]:
            reference = _reference(history, config, clocks)
            if day in grouped:
                positions = symbol_positions[np.asarray(grouped[day])]
                positions = positions[np.argsort(times[positions], kind="stable")]
                values, stats = _day(times[positions], coordinates[positions], config, reference)
                result[positions] = values
            else:
                stats = _Stats(clocks)
            history.append((day, stats))
    output = pd.DataFrame(result, columns=FIELDS, index=frame.index)
    scaled = mids * 200.0
    output.attrs["half_cent_validation"] = {
        "samples": len(mids),
        "maximum_grid_residual": float(np.max(np.abs(scaled - physical), initial=0.0)),
        "maximum_ulp_bound": float(np.max(np.spacing(mids) * 200.0 + np.spacing(scaled), initial=0.0)),
        "resolution_twd": 0.005,
        "tick_coordinate_units": "exact_integer_milliticks",
    }
    return output
