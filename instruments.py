"""Relationships among stocks, single-stock futures and warrants from daily contract files.

Cross-market research needs to know which futures and warrants reference which
stock, and how many shares one futures contract carries. A dated authoritative
underlying_symbol is preferred; conflicting name evidence is rejected. Older
files use an explicitly labelled legacy name fallback. Contract files name the
underlying in Chinese: futures as '台積電期貨09' or '小型台積電期貨09', warrants by
the exchange short name ('日月光' for 3711 日月光投控). Mapping is explicit and
auditable: the exact name, then the name without '-KY'/'*' marks, then a unique
common-stock name prefix; anything else stays unmapped and is reported rather
than guessed. Index underlyings are labelled as indexes.

Usage: python3.13 -m AstraResearch.Experiments.instruments DAY [--contract-root DIR] [--output DIR]
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from AstraResearch.io import ContractError, write_csv, write_yaml
from AstraResearch.units import security_ladder

CONTRACT_ROOT = Path("/mnt/data0/contract")
INDEX_UNDERLYINGS = {"臺股指": "TAIEX", "櫃買指": "TPEX"}
# TAIFEX stock futures: a standard contract is 2000 shares, a small one 100.
SHARES_PER_CONTRACT = {"standard": 2000, "small": 100}
FUTURES_NAME = re.compile(r"(小型)?(.+?)期貨\d{2}")


def _rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _normal(name):
    return re.sub(r"(-KY|-DR|\*)+$", "", name.strip())


def security_names(contract_root, day):
    """Share and fund names of TWSE/TPEx listings (warrants and convertible bonds excluded)."""
    names = {}
    for exchange in ("tse", "otc"):
        for row in _rows(Path(contract_root) / exchange / "stock" / f"{day}.csv"):
            symbol, name = row["symbol"].strip(), row["name"].strip()
            if security_ladder(symbol) not in {"stock", "fund"}:
                continue
            if names.get(name, symbol) != symbol:
                raise ContractError(f"Ambiguous security name {name} on {day}")
            names[name] = symbol
    if not names:
        raise ContractError(f"No security contract names for {day} under {contract_root}")
    return names


class NameIndex:
    """Underlying-name lookup over one day's listing names, built once."""

    def __init__(self, names):
        self.names = names
        self.normal = {_normal(n): symbol for n, symbol in names.items()}
        self.common = [(n, symbol) for n, symbol in self.normal.items() if re.fullmatch(r"[1-9]\d{3}", symbol)]

    def resolve(self, name):
        """(symbol, rule) of an underlying name, or (None, 'unmapped')."""
        name = name.strip()
        if name in INDEX_UNDERLYINGS:
            return INDEX_UNDERLYINGS[name], "index"
        if name in self.names:
            return self.names[name], "exact"
        if _normal(name) in self.normal:
            return self.normal[_normal(name)], "normalized"
        prefix = _normal(name)
        matches = [symbol for n, symbol in self.common if prefix and n.startswith(prefix)]
        if len(matches) == 1:
            return matches[0], "unique_prefix"
        return None, "unmapped"


def stock_futures(contract_root, day, names=None):
    """One row per mapped futures product; rule records authority or legacy fallback."""
    index = NameIndex(security_names(contract_root, day) if names is None else names)
    products = {}
    for row in _rows(Path(contract_root) / "taifex" / "futures" / f"{day}.csv"):
        symbol, name = row["symbol"].strip(), row["name"].strip()
        match = FUTURES_NAME.fullmatch(name)
        authoritative = (row.get("underlying_symbol") or "").strip()
        if len(symbol) != 5 or (not match and not authoritative):
            continue
        named_underlying, name_rule = index.resolve(match.group(2)) if match else (None, "unmapped")
        if authoritative:
            if authoritative not in index.names.values():
                raise ContractError(f"Authoritative underlying {authoritative} for {symbol} is not listed on {day}")
            if named_underlying is not None and named_underlying != authoritative:
                raise ContractError(f"Authoritative underlying {authoritative} conflicts with name mapping {named_underlying} for {symbol} on {day}")
            underlying, rule = authoritative, "underlying_symbol"
        else:
            underlying, rule = named_underlying, f"legacy_name_{name_rule}"
        if underlying is None or (not authoritative and name_rule == "index"):
            continue
        size = ("small" if match.group(1) else "standard") if match else "unknown"
        entry = products.setdefault(
            symbol[:3],
            {
                "product": symbol[:3],
                "underlying": underlying,
                "underlying_name": match.group(2) if match else "",
                "size": size,
                "shares_per_contract": SHARES_PER_CONTRACT.get(size) if security_ladder(underlying) == "stock" else None,
                "rule": rule,
                "contracts": [],
            },
        )
        if (entry["underlying"], entry["size"]) != (underlying, size):
            raise ContractError(f"Futures product {symbol[:3]} names two underlyings on {day}")
        if entry["rule"] != rule:
            entry["rule"] = "mixed_authority_legacy" if entry["rule"] in {"underlying_symbol", "mixed_authority_legacy"} or rule == "underlying_symbol" else "legacy_name_mixed"
        entry["contracts"].append(symbol)
    return [{**p, "contracts": " ".join(sorted(p["contracts"]))} for p in sorted(products.values(), key=lambda p: p["product"])]


def warrants(contract_root, day, names=None):
    """One row per listed warrant with its underlying, call/put side and maturity."""
    index = NameIndex(security_names(contract_root, day) if names is None else names)
    cache = {}
    rows = []
    for exchange in ("tse", "otc"):
        for row in _rows(Path(contract_root) / exchange / "warrant" / f"{day}.csv"):
            label = (row.get("underlying") or "").strip()
            underlying, rule = cache[label] if label in cache else cache.setdefault(label, index.resolve(label))
            name = row["name"].strip()
            side = "call" if "購" in name else "put" if "售" in name else None
            rows.append(
                {
                    "symbol": row["symbol"].strip(),
                    "exchange": row["exchange"].strip(),
                    "underlying": underlying,
                    "underlying_name": label,
                    "side": side,
                    "maturity_date": (row.get("maturity_date") or "").strip(),
                    "rule": rule,
                }
            )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map stock futures and warrants to their listed underlyings")
    parser.add_argument("day")
    parser.add_argument("--contract-root", default=str(CONTRACT_ROOT))
    parser.add_argument("--output", default=".")
    args = parser.parse_args(argv)
    names = security_names(args.contract_root, args.day)
    futures, listed = stock_futures(args.contract_root, args.day, names), warrants(args.contract_root, args.day, names)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / f"stock-futures-{args.day}.csv", futures, fields=["product", "underlying", "underlying_name", "size", "shares_per_contract", "rule", "contracts"])
    write_csv(output / f"warrants-{args.day}.csv", listed, fields=["symbol", "exchange", "underlying", "underlying_name", "side", "maturity_date", "rule"])
    unmapped = {}
    for row in listed:
        if row["underlying"] is None:
            unmapped[row["underlying_name"]] = unmapped.get(row["underlying_name"], 0) + 1
    summary = {
        "day": args.day,
        "stock_futures": len(futures),
        "warrants": len(listed),
        "warrants_by_rule": {rule: sum(r["rule"] == rule for r in listed) for rule in sorted({r["rule"] for r in listed})},
        "unmapped_warrant_underlyings": dict(sorted(unmapped.items(), key=lambda item: -item[1])),
    }
    write_yaml(output / f"instrument-map-{args.day}.yaml", summary)
    print({k: v for k, v in summary.items() if k != "unmapped_warrant_underlyings"})


if __name__ == "__main__":
    main()
