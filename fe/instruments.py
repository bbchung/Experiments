"""Stock -> single-stock futures product roots from dated contract names.

TAIFEX contract rows leave ``underlying`` empty, so the relation is read from
names: ``<stock name>期貨<MM>`` is the standard stock future of that stock
(mini ``小型`` products are excluded). Stock names may carry a ``*`` marker the
futures name omits. The mapping is frozen from one pre-study date and used for
the whole study; a symbol without a product runs without a futures leg.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from ...io import write_yaml
from . import profile as P

MONTHLY = re.compile(r"^(?P<name>.+)期貨(?P<month>\d\d)$")
PRODUCT = re.compile(r"^(?P<root>[A-Z0-9]{3})[A-L]\d$")


def futures_roots(profile, date: str) -> dict[str, str]:
    with Path(profile["paths"]["contracts"], f"{date}.csv").open(encoding="utf-8") as stream:
        stocks = {r["symbol"]: r["name"].replace("*", "").strip() for r in csv.DictReader(stream)}
    by_name: dict[str, set[str]] = {}
    with Path(profile["paths"]["futures_contracts"], f"{date}.csv").open(encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            m, p = MONTHLY.match(r["name"]), PRODUCT.match(r["symbol"])
            if m and p and not m.group("name").startswith("小型"):
                by_name.setdefault(m.group("name").strip(), set()).add(p.group("root"))
    out = {}
    for symbol in P.symbols(profile):
        roots = by_name.get(stocks.get(symbol, ""), set())
        # Adjusted contracts after corporate actions take a digit (e.g. IR1 beside IRF).
        standard = sorted(r for r in roots if r.endswith("F"))
        if len(standard) == 1:
            out[symbol] = standard[0]
    return out


def freeze_map(profile) -> dict:
    lo, hi = (str(x) for x in profile["universe"]["screen_window"])
    # The latest screen-window date with both stock and futures contract tables.
    dates = sorted(p.stem for p in Path(profile["paths"]["contracts"]).glob("*.csv") if lo <= p.stem <= hi and Path(profile["paths"]["futures_contracts"], p.name).is_file())
    date = dates[-1]
    mapping = futures_roots(profile, date)
    write_yaml(P.work(profile) / "universe" / "futures_map.yaml", {"date": date, "roots": mapping})
    return mapping


MARKET_SUFFIXES = (".bin.zst", ".bin", ".csv.zst", ".csv")


def third_wednesday(year: int, month: int) -> int:
    import calendar

    days = [d for d in range(1, 22) if calendar.weekday(year, month, d) == 2]
    return days[2]


def near_month(profile, root: str, day: str, listed: list[str], trading_days: list[str]) -> str | None:
    """The engine's ``root@1`` rule: the earliest listed delivery, skipping the
    current month on its third Wednesday and on the trading day before it."""
    year, month = int(day[:4]), int(day[4:6])
    third = f"{year:04d}{month:02d}{third_wednesday(year, month):02d}"
    previous = [d for d in trading_days if d < third]
    roll = day == third or (bool(previous) and previous[-1] == day)
    candidates = []
    for symbol in listed:
        p = PRODUCT.match(symbol)
        if not p or p.group("root") != root:
            continue
        m = "ABCDEFGHIJKL".index(symbol[3]) + 1
        digit = int(symbol[4])
        y = year - year % 10 + digit
        if y < year - 1:
            y += 10
        yyyymm = y * 100 + m
        if roll and yyyymm == year * 100 + month:
            continue
        candidates.append((yyyymm, symbol))
    return min(candidates)[1] if candidates else None


def day_pairs(profile, day: str) -> list[tuple[str, str]]:
    """(stock, resolved near-month futures) of every mapped pair whose futures tape exists."""
    path = Path(profile["paths"]["futures_contracts"], f"{day}.csv")
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        listed = [r["symbol"] for r in csv.DictReader(stream)]
    trading = P.calendar_days(profile)
    roots = P.work(profile) / "universe" / "futures_map.yaml"
    from ...io import read_yaml

    out = []
    for stock, root in sorted(read_yaml(roots)["roots"].items()):
        symbol = near_month(profile, root, day, listed, trading)
        if symbol and any(Path(profile["paths"]["futures_data"], symbol, f"{day}{sfx}").is_file() for sfx in MARKET_SUFFIXES):
            out.append((stock, symbol))
    return out
