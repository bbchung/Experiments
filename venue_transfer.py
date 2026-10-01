"""Frozen-score, receive-causal stock/SSF quote markouts; no trading simulation.

Prepare freezes inputs and verifies contract resolution with native metadata only.
Run claims development exposure before label endpoints or raw quotes are opened.
"""

from __future__ import annotations

import sys as _astra_sys
from pathlib import Path as _AstraPath

_astra_repo_root = next(parent.parent for parent in _AstraPath(__file__).resolve().parents if parent.name == "AstraResearch")
_astra_sys.path.insert(0, str(_astra_repo_root))

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml
import zstandard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from AstraResearch.Experiments.instruments import stock_futures
from AstraResearch.io import ContractError, file_hash, file_stamp, lock, read_yaml, write_yaml
from AstraResearch.store import ExposureLedger

KEYS = ["day", "symbol", "SampleTime"]
RAW_COLUMNS = ["Type", "Seq", "Timestamp", "ExchangeTime", "StatusMask", "BidDepth", "AskDepth", "BidPrice1", "AskPrice1", "BidVol1", "AskVol1"]


def local_us(day, time):
    return int(pd.Timestamp(f"{day} {time}", tz="Asia/Taipei").value // 1000)


def path_for(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_config(path):
    config = read_yaml(path)
    if config["last_date"] != "20260826" or config["protected_from"] != "20260828":
        raise ContractError("This preregistration is bounded to the existing development dates")
    return config


def score_paths(config):
    return [
        path_for(config["score_root"]) / f"{config['geometry']}-{arm}-{month}.parquet"
        for arm in config["arms"]
        for month in [config["discovery_month"], *config["validation_months"]]
    ]


def monthly_contract(day, product, contracts, trading_days):
    """Mirror native @1; prepare separately verifies every result with C++."""
    year, month = int(day[:4]), int(day[4:6])
    third = [week[calendar.WEDNESDAY] for week in calendar.monthcalendar(year, month) if week[calendar.WEDNESDAY]][2]
    expiry = f"{year:04d}{month:02d}{third:02d}"
    previous = max(d for d in trading_days if d < expiry)
    roll = day in {expiry, previous}
    ranked = []
    for symbol in contracts:
        if not re.fullmatch(re.escape(product) + r"[A-L][0-9]", symbol):
            continue
        delivery_year = year // 10 * 10 + int(symbol[-1])
        if delivery_year < year - 2:
            delivery_year += 10
        if delivery_year > year + 8:
            delivery_year -= 10
        delivery = delivery_year * 100 + ord(symbol[-2]) - 64
        if not (roll and delivery == year * 100 + month):
            ranked.append((delivery, symbol))
    if not ranked:
        raise ContractError(f"No monthly contract for {day}/{product}")
    return min(ranked)[1]


def csv_metadata(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {r["symbol"]: r for r in csv.DictReader(stream)}


def native_resolve(config, day, products, output):
    """One metadata-only native process per date, no market-data subscriptions."""
    work = output / "native-metadata" / day
    work.mkdir(parents=True)
    declarations = [
        {
            "Desc": f"TaifexInfo.{i}",
            "Spec": {
                "Product": product,
                "TickLadder": "stock_future",
                "PointValue": config["mapping"]["shares_per_contract"],
                "BasicInfo": str(path_for(config["contract_root"]) / "taifex/futures"),
            },
        }
        for i, product in enumerate(sorted(products))
    ]
    result_path = work / "contracts.yaml"
    declarations.append({"Desc": "MarketInfoWriter.0", "Spec": {"OutputPath": str(result_path), "Contracts": [p + "@1" for p in sorted(products)]}})
    config_path = work / "config.yaml"
    write_yaml(config_path, {"Modules": [{"Gid": "", "Decl": declarations}]})
    command = [str(path_for(config["binary"])), "-d", day, "-C", str(work), "--quiet", "--trading-calendar", str(path_for(config["calendar"])), str(config_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    (work / "process.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ContractError(f"Native metadata failed on {day}; see {work}")
    native = yaml.load(result_path.read_text(), Loader=yaml.BaseLoader)["contracts"]
    return {p: native[p + "@1"]["symbol"] for p in products}


def prepare(config_path):
    config = read_config(config_path)
    output = path_for(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "freeze.yaml").exists():
        raise ContractError("Study already frozen; do not overwrite its registration")
    owned = [config_path.resolve(), Path(__file__).resolve(), ROOT / "tests/test_venue_transfer.py"]
    code = [*owned, ROOT / "runs/research/run-pool80.sh"] + [
        ROOT / p
        for p in ["Experiments/instruments.py", "io.py", "store.py", "Experiments/research_walkforward.py", "Experiments/research_data.py", "Experiments/research_replay.py"]
    ]
    code += [
        ROOT.parent / p
        for p in [
            "src/oms/labelers/bid_ask_barrier_labeler/bid_ask_barrier_labeler.cpp",
            "src/oms/api/taifex_symbol_resolve.cpp",
            "src/oms/modules/md/trade_book_md/trade_book_md.cpp",
            "src/marketdata/parsers/twse_parser.cpp",
            "src/oms/modules/contract_importer/contract_importer.cpp",
            "src/oms/model/order.cpp",
        ]
    ]
    linked = subprocess.run(["ldd", str(path_for(config["binary"]))], capture_output=True, text=True, check=True)
    dependencies = sorted({Path(p) for p in re.findall(r"(?:=>\s+)?(/[^\s()]+)", linked.stdout) if Path(p).is_file()})
    inputs = [{"path": str(p), "sha256": file_hash(p)} for p in [*code, *score_paths(config), path_for(config["calendar"]), path_for(config["binary"]), *dependencies]]
    write_yaml(
        output / "environment.yaml",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {m.__name__: m.__version__ for m in [np, pd, pa, yaml, zstandard]},
            "native_binary_revision": "e6c9bb296",
            "native_linked_libraries": linked.stdout,
            "native_dependencies": [str(p) for p in dependencies],
        },
    )
    # Freeze rules, source and score bytes before any endpoint or raw price is read.
    write_yaml(output / "registration.yaml", {"config": config, "inputs": inputs, "stage": "before_endpoint_and_raw_markout_reads"})
    for p in code:
        destination = output / "source" / p.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
    days_symbols = defaultdict(set)
    for p in score_paths(config):
        frame = pq.read_table(p, columns=KEYS).to_pandas()
        if frame.duplicated(KEYS).any() or (frame.day > config["last_date"]).any() or (frame.day < "20260601").any():
            raise ContractError(f"Unexpected score keys or protected dates: {p}")
        for day, group in frame.groupby("day"):
            days_symbols[day].update(group.symbol)
    calendar_doc = read_yaml(path_for(config["calendar"]))
    trading_days = sorted(set(map(str, calendar_doc["trading_days"])) - set(map(str, calendar_doc.get("unscheduled_closures", []))))
    inventory = []
    metadata_hashes = {}
    for day, symbols in sorted(days_symbols.items()):
        metadata_root = path_for(config["contract_root"])
        for relative in ["tse/stock", "otc/stock", "taifex/futures"]:
            p = metadata_root / relative / (day + ".csv")
            if p.exists():
                metadata_hashes[str(p)] = file_hash(p)
        stock_metadata = csv_metadata(metadata_root / "tse/stock" / (day + ".csv"))
        by_symbol = defaultdict(list)
        for row in stock_futures(metadata_root, day):
            if (
                row["size"] == config["mapping"]["size"]
                and row["rule"] == config["mapping"]["required_rule"]
                and re.fullmatch(config["mapping"]["product_pattern"], row["product"])
                and re.fullmatch(r"[1-9][0-9]{3}", row["underlying"])
            ):
                by_symbol[row["underlying"]].append(row)
        mapped = {s: by_symbol[s][0] for s in symbols if len(by_symbol[s]) == 1}
        native = native_resolve(config, day, {row["product"] for row in mapped.values()}, output)
        for symbol in sorted(symbols):
            row = {"day": day, "symbol": symbol, "mapped": symbol in mapped, "mapping_status": "unmapped_or_ambiguous"}
            if symbol in mapped:
                mapping = mapped[symbol]
                contract = monthly_contract(day, mapping["product"], mapping["contracts"].split(), trading_days)
                if native[mapping["product"]] != contract:
                    raise ContractError(f"Python/native contract mismatch: {day}/{symbol}")
                row.update(product=mapping["product"], contract=contract, mapping_status=mapping["rule"], shares=mapping["shares_per_contract"])
                for kind, instrument in [("stock", symbol), ("future", contract)]:
                    path = path_for(config[f"{kind}_root"]) / instrument / f"{day}.csv.zst"
                    row[kind + "_path"] = str(path)
                    row[kind + "_stamp"] = list(file_stamp(path)) if path.exists() else None
                metadata = stock_metadata[symbol]
                row.update(stock_limit_up=float(metadata["limit_up"]), stock_limit_down=float(metadata["limit_down"]), stock_trading_unit=float(metadata["unit"]))
                if row["stock_trading_unit"] != 1000:
                    raise ContractError(f"Unverified stock quote quantity unit: {day}/{symbol}")
                path = path_for(config["label_root"]) / day / "data" / day / symbol / "values.parquet"
                row["label_path"] = str(path)
                row["label_sha256"] = file_hash(path) if path.exists() else None
            inventory.append(row)
        print(f"prepared metadata {day}: {len(mapped)}/{len(symbols)} mapped", flush=True)
    write_yaml(output / "inventory.yaml", inventory)
    write_yaml(
        output / "freeze.yaml",
        {
            "registration_sha256": file_hash(output / "registration.yaml"),
            "inventory_sha256": file_hash(output / "inventory.yaml"),
            "environment_sha256": file_hash(output / "environment.yaml"),
            "inputs": inputs,
            "metadata_hashes": metadata_hashes,
        },
    )
    print(f"FROZEN {output}", flush=True)


def select_intents(frame, config, arm):
    """Spacing depends only on score/time, including invalid and late outcomes."""
    rows = []
    for side, sign in [("buy", 1), ("sell", -1)]:
        column = f"{side}_threshold_{config['quantile']}"
        for month, part in frame.groupby(frame.day.str[:6]):
            thresholds = part[column].unique()
            if len(thresholds) != 1 or not np.isfinite(thresholds[0]):
                raise ContractError(f"Nonconstant frozen threshold: {arm}/{month}/{side}")
            chosen = part.loc[sign * part.edge >= thresholds[0]].sort_values(KEYS, kind="stable")
            for (day, symbol), group in chosen.groupby(["day", "symbol"], sort=True):
                following = -math.inf
                for row in group.itertuples():
                    time = int(row.SampleTime)
                    if time < following:
                        continue
                    following = time + config["spacing_us"]
                    rows.append(
                        {
                            "arm": arm,
                            "side": side,
                            "day": day,
                            "month": month,
                            "symbol": symbol,
                            "SampleTime": time,
                            "score": sign * float(row.edge),
                            "threshold": float(thresholds[0]),
                            "in_session": local_us(day, config["entry_start"]) <= time <= local_us(day, config["entry_cutoff"]),
                        }
                    )
    return pd.DataFrame(rows)


class HashReader(io.RawIOBase):
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        self.digest.update(data)
        buffer[: len(data)] = data
        return len(data)


def quote_events(frame, *, stock, upper, lower, prior_time=0):
    """Preserve raw event order and native receive-clock clamping; retain invalidations."""
    if not frame.Type.isin(["B", "T"]).all():
        raise ContractError("unsupported_raw_event_type")
    raw_time = frame.Timestamp.to_numpy(dtype=np.int64)
    times = np.maximum.accumulate(np.maximum(raw_time, prior_time))
    status = np.array([int(str(value), 16) for value in frame.StatusMask], dtype=np.int64)
    book = frame.Type.to_numpy() == "B"
    # Any noncontinuous observation breaks quoted-state validity; only a book reseeds it.
    keep = book | (status != 0)
    bid, ask = frame.BidPrice1.to_numpy(float).copy(), frame.AskPrice1.to_numpy(float).copy()
    bd, ad = frame.BidDepth.to_numpy(), frame.AskDepth.to_numpy()
    bv, av = frame.BidVol1.to_numpy(), frame.AskVol1.to_numpy()
    if stock:
        bid = np.where((status == 0) & (bd > 0) & (bv > 0) & (bid == 0), upper, bid)
        ask = np.where((status == 0) & (ad > 0) & (av > 0) & (ask == 0), lower, ask)
    valid = book & (status == 0) & (bd > 0) & (ad > 0) & (bv > 0) & (av > 0) & np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
    return (times[keep], np.where(valid[keep], bid[keep], np.nan), np.where(valid[keep], ask[keep], np.nan), bv[keep], av[keep]), int(times[-1]) if len(times) else prior_time


def read_quotes(path, stamp, *, stock=False, upper=math.nan, lower=math.nan):
    if stamp is None or not path.is_file():
        raise ContractError("missing_file")
    if tuple(stamp) != file_stamp(path):
        raise ContractError("source_changed_since_registration")
    arrays = [[], [], [], [], []]
    previous = 0
    types = {name: pa.string() if name in {"Type", "StatusMask"} else pa.float64() if "Price" in name else pa.int64() for name in RAW_COLUMNS}
    with path.open("rb") as raw:
        hashed = HashReader(raw)
        with zstandard.ZstdDecompressor().stream_reader(hashed) as stream:
            reader = pacsv.open_csv(
                stream, read_options=pacsv.ReadOptions(use_threads=False, block_size=1 << 20), convert_options=pacsv.ConvertOptions(include_columns=RAW_COLUMNS, column_types=types)
            )
            for batch in reader:
                values, previous = quote_events(batch.to_pandas(), stock=stock, upper=upper, lower=lower, prior_time=previous)
                for dest, values_ in zip(arrays, values, strict=True):
                    dest.append(values_)
        receipt = {"path": str(path), "sha256": hashed.digest.hexdigest(), "stamp": stamp, "last_observation_time": previous}
    if tuple(stamp) != file_stamp(path):
        raise ContractError("source_changed_while_reading")
    return (*tuple(np.concatenate(a) if a else np.array([], dtype=np.int64 if i == 0 else float) for i, a in enumerate(arrays)), previous), receipt


def quote_at(tape, at, max_age):
    times, bid, ask, bid_volume, ask_volume, *observed_end = tape
    if not np.isfinite(at):
        return None, "unknown_time"
    if observed_end and at > observed_end[0]:
        return None, "past_observed_input_end"
    index = int(np.searchsorted(times, at, side="left")) - 1
    if index < 0:
        return None, "before_first_observation"
    age = int(at) - int(times[index])
    if not np.isfinite(bid[index]) or not np.isfinite(ask[index]):
        return None, "invalid_state"
    if age > max_age:
        return None, "stale"
    return {
        "bid": float(bid[index]),
        "ask": float(ask[index]),
        "bid_volume": float(bid_volume[index]),
        "ask_volume": float(ask_volume[index]),
        "quote_time": int(times[index]),
        "age_us": age,
        "arrival_tied_events": int(np.searchsorted(times, at, side="right") - np.searchsorted(times, at, side="left")),
    }, "ok"


def cash_markout(entry, exit_, side, venue, shares, costs):
    buy = side == "buy"
    entry_price = entry["ask" if buy else "bid"]
    exit_price = exit_["bid" if buy else "ask"]
    sign = 1 if buy else -1
    mid0, mid1 = (entry["bid"] + entry["ask"]) / 2, (exit_["bid"] + exit_["ask"]) / 2
    gross = sign * (exit_price - entry_price) * shares
    if venue == "stock":
        fee = shares * ((entry_price + exit_price) * costs["stock_commission_rate"] + (exit_price if buy else entry_price) * costs["stock_sell_tax_rate"])
    else:
        fee = 2 * costs["future_commission_per_side_twd"] + shares * (entry_price + exit_price) * costs["future_tax_per_side"]
    scale = 10000 / (mid0 * shares)
    return {
        "gross_bps": gross * scale,
        "fee_bps": fee * scale,
        "net_bps": (gross - fee) * scale,
        "stress_net_bps": (gross - costs["stress_multiplier"] * fee) * scale,
        "mid_move_bps": sign * (mid1 - mid0) / mid0 * 10000,
        "crossing_bps": (sign * (mid1 - mid0) * shares - gross) * scale,
        "gross_cash": gross,
        "fee_cash": fee,
        "net_cash": gross - fee,
    }


def evaluate_intent(intent, mapping, config, tapes, errors, stop):
    results = []
    for endpoint in config["exits"]:
        row = {**intent, "exit_rule": endpoint, "mapped": mapping["mapped"], "product": mapping.get("product"), "contract": mapping.get("contract"), "resolved": False}
        row["entry_time"] = intent["SampleTime"] + config["latency_us"]
        row["exit_decision_time"] = stop if endpoint == "stock_stop" else intent["SampleTime"] + config["spacing_us"]
        if not intent["in_session"]:
            row["reason"] = "outside_predeclared_session"
        elif not mapping["mapped"]:
            row["reason"] = mapping["mapping_status"]
        elif not np.isfinite(row["exit_decision_time"]):
            row["reason"] = errors.get("label", "stock_stop_unresolved")
        elif row["exit_decision_time"] < intent["SampleTime"] or row["exit_decision_time"] > intent["SampleTime"] + config["spacing_us"]:
            row["reason"] = "stock_stop_outside_horizon"
        else:
            row["exit_time"] = int(row["exit_decision_time"]) + config["latency_us"]
            if row["exit_time"] >= local_us(intent["day"], config["last_exit"]):
                row["reason"] = "outside_continuous_exit_session"
                results.append(row)
                continue
            quotes = {}
            failures = []
            for venue in ["stock", "future"]:
                if venue in errors:
                    failures.append(venue + ":" + errors[venue])
                    continue
                for point in ["entry", "exit"]:
                    quote, status = quote_at(tapes[venue], row[point + "_time"], config["maximum_quote_age_us"])
                    row[venue + "_" + point + "_status"] = status
                    if quote is None:
                        failures.append(venue + "_" + point + ":" + status)
                    else:
                        quotes[(venue, point)] = quote
                        row.update({venue + "_" + point + "_" + k: v for k, v in quote.items()})
                        required = mapping["shares"] / mapping["stock_trading_unit"] if venue == "stock" else 1
                        if quote["bid_volume"] < required or quote["ask_volume"] < required:
                            failures.append(venue + "_" + point + ":insufficient_visible_top_capacity")
            row["reason"] = ";".join(failures) if failures else "resolved"
            if not failures:
                row["resolved"] = True
                for venue in ["stock", "future"]:
                    row.update(
                        {
                            venue + "_" + k: v
                            for k, v in cash_markout(quotes[(venue, "entry")], quotes[(venue, "exit")], intent["side"], venue, mapping["shares"], config["costs"]).items()
                        }
                    )
                row["vehicle_delta_net_bps"] = row["future_net_bps"] - row["stock_net_bps"]
                long = cash_markout(quotes[("future", "entry")], quotes[("future", "exit")], "buy", "future", mapping["shares"], config["costs"])
                short = cash_markout(quotes[("future", "entry")], quotes[("future", "exit")], "sell", "future", mapping["shares"], config["costs"])
                row["future_always_long_net_bps"] = long["net_bps"]
                row["future_always_short_net_bps"] = short["net_bps"]
                row["future_expected_50_50_net_bps"] = (long["net_bps"] + short["net_bps"]) / 2
                row["future_directional_skill_bps"] = row["future_net_bps"] - row["future_expected_50_50_net_bps"]
        results.append(row)
    return results


def resolve_stops(frame, column):
    """No outcome-based duplicate resolution: only unique native times can be joined."""
    duplicated = frame.SampleTime.duplicated(keep=False)
    unique = frame.loc[~duplicated]
    return dict(zip(unique.SampleTime, unique[column], strict=True)), set(frame.loc[duplicated, "SampleTime"])


def run_job(job):
    mapping, intents, config = job
    output = path_for(config["output"]) / "parts"
    key = mapping["day"] + "-" + mapping["symbol"]
    result_path, receipt_path = output / (key + ".parquet"), output / (key + ".yaml")
    if result_path.exists() and receipt_path.exists():
        if file_hash(result_path) != read_yaml(receipt_path)["result_sha256"]:
            raise ContractError("Corrupted existing diagnostic part: " + key)
        return key, "existing"
    tapes, errors, receipts, stops, ambiguous = {}, {}, [], {}, set()
    if mapping["mapped"] and any(i["in_session"] for i in intents):
        path = Path(mapping["label_path"])
        try:
            if mapping["label_sha256"] is None or file_hash(path) != mapping["label_sha256"]:
                raise ContractError("missing_or_changed_labels")
            column = config["geometry"] + ".end_time[1800s]"
            frame = pq.read_table(path, columns=["SampleTime", column]).to_pandas()
            stops, ambiguous = resolve_stops(frame, column)
        except (OSError, ValueError, pa.ArrowException) as exc:
            errors["label"] = type(exc).__name__ + ":" + str(exc)
        for venue in ["stock", "future"]:
            try:
                tapes[venue], receipt = read_quotes(
                    Path(mapping[venue + "_path"]), mapping[venue + "_stamp"], stock=venue == "stock", upper=mapping["stock_limit_up"], lower=mapping["stock_limit_down"]
                )
                receipts.append(receipt)
            except (OSError, ValueError, pa.ArrowException, zstandard.ZstdError) as exc:
                errors[venue] = type(exc).__name__ + ":" + str(exc)
    rows = [
        r
        for intent in intents
        for r in evaluate_intent(
            intent,
            mapping,
            config,
            tapes,
            {**errors, "label": "ambiguous_native_sample_time"} if intent["SampleTime"] in ambiguous else errors,
            stops.get(intent["SampleTime"], math.nan),
        )
    ]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(result_path)
    write_yaml(
        receipt_path,
        {
            "day": mapping["day"],
            "symbol": mapping["symbol"],
            "sources": receipts,
            "errors": errors,
            "ambiguous_native_timestamps": len(ambiguous),
            "selected_ambiguous_intents": sum(i["SampleTime"] in ambiguous for i in intents),
            "result_sha256": file_hash(result_path),
        },
    )
    return key, f"{len(intents)} intents; errors={list(errors)}"


def block_lower(frame, column, config, *, equal_day=False):
    if frame.empty:
        return None
    daily = frame.groupby("day")[column].agg(["sum", "count"])
    n, settings = len(daily), config["bootstrap"]
    if n < settings["block_days"] * 2:
        return None
    rng = np.random.default_rng(settings["seed"])
    starts = rng.integers(0, n, (settings["resamples"], math.ceil(n / settings["block_days"])))
    indexes = ((starts[:, :, None] + np.arange(settings["block_days"])) % n).reshape(settings["resamples"], -1)[:, :n]
    totals, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
    values = (totals / counts)[indexes].mean(axis=1) if equal_day else totals[indexes].sum(axis=1) / counts[indexes].sum(axis=1)
    return float(np.quantile(values, settings["lower_quantile"]))


def summarize_group(frame, config):
    mapped = frame.loc[frame.in_session & frame.mapped]
    good = mapped.loc[mapped.resolved]
    row = {
        "selected_intents": len(frame),
        "outside_session": int((~frame.in_session).sum()),
        "unmapped": int((frame.in_session & ~frame.mapped).sum()),
        "mapped_intents": len(mapped),
        "mapped_fraction_in_session": len(mapped) / int(frame.in_session.sum()) if frame.in_session.any() else 0,
        "paired_events": len(good),
        "common_support": len(good) / len(mapped) if len(mapped) else 0,
        "paired_days": int(good.day.nunique()),
        "paired_symbols": int(good.symbol.nunique()),
    }
    if good.empty:
        return row
    for c in [
        "stock_net_bps",
        "future_net_bps",
        "future_stress_net_bps",
        "vehicle_delta_net_bps",
        "stock_gross_bps",
        "future_gross_bps",
        "stock_fee_bps",
        "future_fee_bps",
        "stock_mid_move_bps",
        "future_mid_move_bps",
        "future_always_long_net_bps",
        "future_always_short_net_bps",
        "future_expected_50_50_net_bps",
        "future_directional_skill_bps",
    ]:
        row[c] = float(good[c].mean())
        row[c.removesuffix("_bps") + "_equal_day_bps"] = float(good.groupby("day")[c].mean().mean())
    row["future_positive_fraction"] = float((good.future_net_bps > 0).mean())
    row["stock_future_mid_move_correlation"] = float(good.stock_mid_move_bps.corr(good.future_mid_move_bps))
    positive = good.groupby("symbol").future_net_bps.sum().clip(lower=0)
    row["maximum_symbol_positive_profit_share"] = float(positive.max() / positive.sum()) if positive.sum() else 1.0
    cash_positive = good.groupby("symbol").future_net_cash.sum().clip(lower=0)
    row["maximum_symbol_positive_cash_profit_share"] = float(cash_positive.max() / cash_positive.sum()) if cash_positive.sum() else 1.0
    row["future_net_lower95_bps"] = block_lower(good, "future_net_bps", config)
    row["paired_delta_lower95_bps"] = block_lower(good, "vehicle_delta_net_bps", config)
    row["future_net_equal_day_lower95_bps"] = block_lower(good, "future_net_bps", config, equal_day=True)
    row["directional_skill_equal_day_lower95_bps"] = block_lower(good, "future_directional_skill_bps", config, equal_day=True)
    return row


def validate_report_registration(output, config):
    """Standalone reporting cannot silently change a frozen study's policy or costs."""
    registration = output / "registration.yaml"
    freeze = read_yaml(output / "freeze.yaml")
    if file_hash(registration) != freeze["registration_sha256"]:
        raise ContractError("Frozen report registration changed")
    if read_yaml(registration)["config"] != config:
        raise ContractError("Report configuration differs from the frozen registration")


def report(config):
    output = path_for(config["output"])
    validate_report_registration(output, config)
    jobs = read_yaml(output / "jobs.yaml")
    expected = {j["key"] for j in jobs}
    found = {p.stem for p in (output / "parts").glob("*.parquet")}
    if found != expected:
        raise ContractError("Diagnostic parts do not match the complete expected job set")
    frames = []
    for job in jobs:
        path = output / "parts" / (job["key"] + ".parquet")
        receipt = read_yaml(path.with_suffix(".yaml"))
        if file_hash(path) != receipt["result_sha256"]:
            raise ContractError("Diagnostic part checksum mismatch: " + job["key"])
        part = pd.read_parquet(path)
        if (
            len(part) != job["intents"] * len(config["exits"])
            or part.duplicated(["arm", "side", "SampleTime", "exit_rule"]).any()
            or set(part.day) != {job["day"]}
            or set(part.symbol) != {job["symbol"]}
        ):
            raise ContractError("Diagnostic part population mismatch: " + job["key"])
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(output / "intents.parquet")
    summaries = []
    for (arm, endpoint, month), group in frame.groupby(["arm", "exit_rule", "month"]):
        for side in ["both_independent", "buy", "sell"]:
            part = group if side == "both_independent" else group.loc[group.side == side]
            summaries.append({"arm": arm, "exit_rule": endpoint, "month": month, "side": side, **summarize_group(part, config)})
    pd.DataFrame(summaries).to_csv(output / "monthly.csv", index=False)
    frame.groupby(["arm", "exit_rule", "month", "side", "reason"], dropna=False).size().rename("intents").to_csv(output / "coverage-reasons.csv")
    frame.groupby(["arm", "exit_rule", "month", "day", "symbol", "side", "reason"], dropna=False).size().rename("intents").to_csv(output / "coverage-by-day-symbol.csv")
    good = frame.loc[frame.resolved]
    if not good.empty:
        good.groupby(["arm", "exit_rule", "month", "day", "symbol", "side"])[["stock_net_bps", "future_net_bps", "future_stress_net_bps", "vehicle_delta_net_bps"]].agg(
            ["size", "mean", "sum"]
        ).to_csv(output / "by-day-symbol-side.csv")
    decisions = []
    for arm in config["arms"]:
        validation = frame.loc[(frame.arm == arm) & (frame.exit_rule == config["primary_exit"]) & frame.month.isin(config["validation_months"])]
        pooled = summarize_group(validation, config)
        monthly = [summarize_group(validation.loc[validation.month == month], config) for month in config["validation_months"]]
        gate = config["advancement"]
        support = all(m["common_support"] >= gate["minimum_common_support"] and m["paired_events"] >= gate["minimum_paired_events_each_validation_month"] for m in monthly)
        economics = (
            support
            and all(m.get("future_net_bps", -math.inf) > 0 for m in monthly)
            and pooled.get("future_stress_net_bps", -math.inf) > 0
            and pooled.get("maximum_symbol_positive_profit_share", 1) <= gate["maximum_symbol_positive_profit_share"]
        )
        decisions.append(
            {
                "arm": arm,
                "status": "advance_to_native_reconstruction" if economics else "inconclusive_support" if not support else "rejected_economics",
                "pooled_validation": pooled,
                "monthly_validation": dict(zip(config["validation_months"], monthly, strict=True)),
            }
        )
    write_yaml(
        output / "summary.yaml",
        {"study": config["study"], "role": config["evidence_role"], "decisions": decisions, "all_monthly_comparisons": summaries, "limitations": config["limitations"]},
    )
    print(json.dumps(decisions, indent=2, default=str), flush=True)


def run(config_path):
    config = read_config(config_path)
    output = path_for(config["output"])
    with lock(output / "run.lock", blocking=False):
        freeze = read_yaml(output / "freeze.yaml")
        if file_hash(output / "registration.yaml") != freeze["registration_sha256"] or file_hash(output / "inventory.yaml") != freeze["inventory_sha256"]:
            raise ContractError("Frozen registration changed")
        if file_hash(output / "environment.yaml") != freeze["environment_sha256"]:
            raise ContractError("Frozen environment provenance changed")
        for item in freeze["inputs"]:
            if file_hash(Path(item["path"])) != item["sha256"]:
                raise ContractError("Frozen source/input changed: " + item["path"])
        for path, expected in freeze["metadata_hashes"].items():
            if file_hash(Path(path)) != expected:
                raise ContractError("Frozen contract metadata changed: " + path)
        inventory = read_yaml(output / "inventory.yaml")
        ledger = ExposureLedger(path_for(config["exposure_store"]))
        ledger.claim("TWSE", sorted({r["day"] for r in inventory}), "development", config["study"])
        for product in sorted({r["product"] for r in inventory if r["mapped"]}):
            ledger.claim("TAIFEX:" + product, sorted({r["day"] for r in inventory if r.get("product") == product}), "development", config["study"])
        write_yaml(
            output / "exposure-claimed.yaml",
            {"owner": config["study"], "role": "development", "days": sorted({r["day"] for r in inventory}), "products": sorted({r["product"] for r in inventory if r["mapped"]})},
        )
        selections = []
        for arm in config["arms"]:
            frames = [pq.read_table(p, columns=KEYS + ["edge", "buy_threshold_0.95", "sell_threshold_0.95"]).to_pandas() for p in score_paths(config) if f"-{arm}-" in p.name]
            selections.append(select_intents(pd.concat(frames, ignore_index=True), config, arm))
        intents = pd.concat(selections, ignore_index=True)
        intents.to_parquet(output / "selected-intents-before-outcomes.parquet")
        groups = {key: group.to_dict("records") for key, group in intents.groupby(["day", "symbol"])}
        jobs = [(row, groups[(row["day"], row["symbol"])], config) for row in inventory if (row["day"], row["symbol"]) in groups]
        write_yaml(
            output / "jobs.yaml", [{"key": row["day"] + "-" + row["symbol"], "day": row["day"], "symbol": row["symbol"], "intents": len(selected)} for row, selected, _ in jobs]
        )
        write_yaml(output / "status.yaml", {"state": "running", "jobs": len(jobs), "workers": min(4, config["workers"])})
        with ProcessPoolExecutor(max_workers=min(4, config["workers"])) as pool:
            for i, (key, message) in enumerate(pool.map(run_job, jobs), 1):
                if i % 20 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)} {key} {message}", flush=True)
                    write_yaml(output / "status.yaml", {"state": "running", "jobs": len(jobs), "completed": i})
        report(config)
        write_yaml(output / "status.yaml", {"state": "completed", "jobs": len(jobs), "completed": len(jobs), "release_authority": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "report"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "Experiments/e2e_20260926/venue_transfer.yaml")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.protocol)
    elif args.command == "run":
        run(args.protocol)
    else:
        report(read_config(args.protocol))


if __name__ == "__main__":
    main()
