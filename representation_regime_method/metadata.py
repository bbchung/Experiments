"""Classify announced exchange mechanisms from strictly prior metadata only.

These helpers do not select a research universe, change a profile or consume
raw streams, features, labels or model outcomes. Absence is never silently
interpreted as a complete published schedule.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ...io import ContractError, file_hash

SCHEMA = "astra-exante-regime-method-draft-v1"
FLOOR = "20260101"
FIELDS = ("公布日期", "證券代號", "處置起迄時間", "處置內容")
MINUTES = {"五": 5, "十": 10, "十五": 15, "二十": 20, "三十": 30, "六十": 60}


def _require(condition, message):
    if not condition:
        raise ContractError(message)


def _day(value):
    _require(type(value) is str and re.fullmatch(r"[0-9]{8}", value) is not None, "canonical YYYYMMDD string required")
    try:
        date(int(value[:4]), int(value[4:6]), int(value[6:]))
    except ValueError as error:
        raise ContractError("invalid Gregorian day") from error
    return value


def roc_day(value):
    value = unicodedata.normalize("NFKC", value).strip()
    match = re.fullmatch(r"([0-9]{3})(?:/?)([0-9]{2})(?:/?)([0-9]{2})", value)
    _require(match is not None, "explicit ROC YYYMMDD or YYY/MM/DD date required")
    year, month, day = map(int, match.groups())
    return _day(f"{year + 1911:04d}{month:02d}{day:02d}")


@dataclass(frozen=True)
class Notice:
    symbol: str
    announcement_day: str
    start_day: str
    end_day: str
    interval_seconds: int | None
    conditional_extension: bool


@dataclass(frozen=True)
class Snapshot:
    day: str
    path: str
    sha256: str
    notices: tuple[Notice, ...]


def parse_notice(row, snapshot_day):
    _day(snapshot_day)
    _require(set(FIELDS).issubset(row), "disposition metadata lacks semantic columns")
    _require(all(type(row[name]) is str and row[name].strip() for name in FIELDS), "empty/non-string notice fields")
    symbol = row["證券代號"].strip()
    announcement = roc_day(row["公布日期"])
    _require(announcement <= snapshot_day, "future announcement in an alleged prior snapshot")
    span = unicodedata.normalize("NFKC", row["處置起迄時間"]).strip()
    match = re.fullmatch(r"([0-9]{3}/[0-9]{2}/[0-9]{2})[~～]([0-9]{3}/[0-9]{2}/[0-9]{2})", span)
    _require(match is not None, "explicit inclusive disposition start/end required")
    start, end = map(roc_day, match.groups())
    _require(announcement <= start <= end, "announcement/start/end order invalid")
    text = unicodedata.normalize("NFKC", row["處置內容"])
    intervals = []
    for token in re.findall(r"約每([0-9]+|五|十|十五|二十|三十|六十)分鐘撮合一次", text):
        minutes = int(token) if token.isascii() and token.isdigit() else MINUTES[token]
        _require(minutes > 0, "nonpositive auction interval")
        intervals.append(minutes * 60)
    _require(len(set(intervals)) <= 1, "conflicting auction intervals within notice")
    return Notice(symbol, announcement, start, end, intervals[0] if intervals else None, "順延" in text or "調整處置迄日" in text)


def read_snapshot(path, snapshot_day):
    path = Path(path)
    _day(snapshot_day)
    _require(snapshot_day >= FLOOR and path.name == f"{snapshot_day}.csv", "snapshot source date outside fixed2026 domain")
    if not path.is_file():
        return None
    before = file_hash(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames), "unambiguous CSV header required")
        _require(set(FIELDS).issubset(reader.fieldnames), "disposition snapshot schema changed")
        notices = tuple(parse_notice(row, snapshot_day) for row in reader)
    _require(len(set(notices)) == len(notices), "duplicate semantic notices require source resolution")
    _require(file_hash(path) == before, "disposition snapshot changed while reading")
    return Snapshot(snapshot_day, str(path.resolve()), before, notices)


def prior_snapshot_day(day, calendar):
    """Previous declared trading slot only; never nearest existing file fallback."""
    _day(day)
    _require(type(calendar) is list and all(type(x) is str for x in calendar) and calendar == sorted(set(calendar)), "fixed unique chronological calendar required")
    for candidate in calendar:
        _day(candidate)
    _require(day in calendar and day >= FLOOR, "target day outside fixed research calendar")
    position = calendar.index(day)
    return calendar[position - 1] if position and calendar[position - 1] >= FLOOR else None


def classify(day, symbol, snapshot, *, expected_snapshot_day):
    """Return mechanism eligibility, not event freshness or predictive admission."""
    _day(day)
    _require(day >= FLOOR and type(symbol) is str and bool(symbol), "canonical research symbol-day required")
    result = {"schema": SCHEMA, "day": day, "symbol": symbol, "regime": "unknown", "continuous_eligible": False, "batch_eligible": False, "method_adopted": False}
    if expected_snapshot_day is None:
        return {**result, "reason": "no_prior_calendar_slot_within_research_floor"}
    _day(expected_snapshot_day)
    _require(FLOOR <= expected_snapshot_day < day, "snapshot must be strictly prior; same-day announcements are not pre-session evidence")
    if snapshot is None:
        return {**result, "reason": "missing_fixed_prior_snapshot_no_fallback"}
    _require(isinstance(snapshot, Snapshot), "parsed immutable metadata snapshot required")
    _require(snapshot.day == expected_snapshot_day, "different snapshot/date substitution rejected")
    result["snapshot"] = {"day": snapshot.day, "path": snapshot.path, "sha256": snapshot.sha256}
    own = [notice for notice in snapshot.notices if notice.symbol == symbol]
    active = [notice for notice in own if notice.start_day <= day <= notice.end_day]
    if any(notice.announcement_day < FLOOR for notice in active):
        return {**result, "reason": "active_announcement_before_research_floor"}
    if len(active) > 1:
        return {**result, "reason": "multiple_active_notices_not_resolved_by_last_write"}
    if active:
        notice = active[0]
        if notice.interval_seconds is None:
            return {**result, "reason": "active_disposition_without_explicit_batch_interval"}
        return {
            **result,
            "regime": "batch_auction",
            "batch_eligible": True,
            "reason": "prior_published_inclusive_active_schedule",
            "announcement_day": notice.announcement_day,
            "start_day": notice.start_day,
            "end_day": notice.end_day,
            "interval_seconds": notice.interval_seconds,
        }
    if any(notice.end_day < day and notice.conditional_extension for notice in own):
        return {**result, "reason": "expired_schedule_has_unresolved_conditional_extension"}
    return {**result, "reason": "absence_does_not_prove_complete_published_schedule"}


def concordance(regime, *, continuous_books, noncontinuous_books):
    """Descriptive source validation from existing native status counts only.

    No labels or feature values are input. In particular, a continuous schedule
    is not a promise that every arrival avoids trial/open/halt snapshots.
    """
    _require(regime in {"batch_auction", "continuous", "unknown"}, "unknown regime literal")
    _require(all(type(count) is int and count >= 0 for count in (continuous_books, noncontinuous_books)), "exact nonnegative native status counts required")
    total = continuous_books + noncontinuous_books
    state = (
        "unsupported_no_observations"
        if not total
        else "unclassified"
        if regime == "unknown"
        else "concordant"
        if (noncontinuous_books > 0 if regime == "batch_auction" else continuous_books > 0)
        else "contradicted"
    )
    return {"status": state, "continuous_books": continuous_books, "noncontinuous_books": noncontinuous_books, "predictive_admission": False}
