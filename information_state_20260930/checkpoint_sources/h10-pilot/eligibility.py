"""One-pass unlabeled TRAIN eligibility; freeze rules before counting rows."""

from __future__ import annotations

import csv
import hashlib
import io
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
import zstandard

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent / "preregistered-v1"
NOTES = ROOT / "AstraResearch/experiments/information_state_20260930/H10_NOTES.md"
PROFILE = ROOT / "AstraResearch/runs/information_state_20260930/v1/source/profile.yaml"
RAW_ROOT = Path("/mnt/data0/marketdata/tse/kgi/stock")
CONTRACT_ROOT = Path("/mnt/data0/contract/tse/stock")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.hash.update(value)
        return value


def plan(profile):
    return {
        "schema": "h10-native-support-preregistration-v1",
        "prior_exploration_seen": True,
        "prior_exploration_receipt_sha256": digest(OUTPUT.parent / "receipt.yaml"),
        "notes_snapshot_sha256": digest(NOTES),
        "profile_sha256": digest(PROFILE),
        "script_sha256": digest(Path(__file__)),
        "eligibility_calendar": profile["splits"]["train"][:20],
        "universe": profile["universe"]["symbols"],
        "eligibility_clock": "091000_inclusive_to_130000_exclusive_Asia_Taipei_monotone_receive_time",
        "eligibility_rule": "usable_continuous_books/scoped_raw_books>=0.9_and_positive_exposure",
        "usable_book": "mask&(TRIAL|AUCTION|SUSPEND)==0_depths_1_to_5_positive_finite_touches_quantities_and_positive_spread_after_native_zero_touch_normalization",
        "native_mid_validity_count": "positive_finite_two_touch_prices_and_quantities_independent_of_status_or_spread",
        "day_trade_exclusion": False,
        "missing_slot_rule": "empty_exposure_never_replace_date_or_extend_twenty_slots",
        "selection_rule": "first_two_qualified_symbols_in_ascending_symbol_order",
        "support_dates": ["20260119", "20260120"],
        "support_chain": "strict_known_exact_previous_touch_execution_containment_later_quote_only_observable_repair_later_touch_renewal_later_known_same_side_new_touch_execution",
        "support_max_demand_gap_seconds": 30,
        "support_minimum_complete_chains_per_present_cell": 5,
        "support_minimum_complete_chains_combined": 20,
        "support_both_directions": True,
        "support_missing_cell_rule": "skip_no_replacement_report_fixed_proof_incomplete",
        "unknown_mixed_out_of_order_or_censored_rule": "censor_never_manufacture_known_demand_or_zero_queue",
        "labels_read": False,
        "model_fits": 0,
        "native_replay": False,
        "thread_count": 1,
        "scope": "support_only_no_FE_effect_comparison_or_admission",
        "source_sha256": {
            name: digest(ROOT / name)
            for name in [
                "src/msg/md_msg.h",
                "src/sdk/math/compare.hpp",
                "src/oms/modules/md/trade_book_md/trade_book_md.cpp",
                "src/oms/modules/tw/twse_filter/twse_filter.cpp",
                "src/oms/model/quote.cpp",
                "src/oms/modules/feature/observable_book.h",
                "src/oms/modules/feature/order_flow/trade_flow/trade_side_inference.h",
            ]
        },
    }


def scan_file(path, day, symbol, contract):
    result = {
        "day": day,
        "symbol": symbol,
        "path": str(path),
        "status": "observed",
        "compressed_bytes": path.stat().st_size,
        "scoped_books": 0,
        "continuous_status_books": 0,
        "usable_continuous_books": 0,
        "native_mid_valid_books": 0,
        "trial_auction_books": 0,
        "suspended_books": 0,
        "receive_regressions_clamped": 0,
    }
    epoch = int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC).timestamp()) * 1_000_000
    start = epoch + 4_200_000_000
    end = epoch + 18_000_000_000
    previous_receive = 0
    with path.open("rb") as raw:
        hashing = HashingReader(raw)
        with zstandard.ZstdDecompressor().stream_reader(hashing, closefd=False) as decompressed, io.TextIOWrapper(decompressed) as stream:
            reader = csv.reader(stream)
            header = next(reader)
            fields = {
                name: header.index(name)
                for name in [
                    "Type",
                    "Timestamp",
                    "StatusMask",
                    "BidDepth",
                    "AskDepth",
                    "BidPrice1",
                    "BidPrice2",
                    "BidVol1",
                    "BidVol2",
                    "AskPrice1",
                    "AskPrice2",
                    "AskVol1",
                    "AskVol2",
                ]
            }
            for row in reader:
                original = int(row[fields["Timestamp"]])
                receive = max(original, previous_receive)
                previous_receive = receive
                if row[fields["Type"]] != "B" or not start <= receive < end:
                    continue
                result["scoped_books"] += 1
                result["receive_regressions_clamped"] += original < receive
                mask = int(row[fields["StatusMask"]], 16)
                result["trial_auction_books"] += bool(mask & 3)
                result["suspended_books"] += bool(mask & 4)
                continuous = not bool(mask & 7)
                result["continuous_status_books"] += continuous
                bid_depth = int(row[fields["BidDepth"]])
                ask_depth = int(row[fields["AskDepth"]])
                if not 1 <= bid_depth <= 5 or not 1 <= ask_depth <= 5:
                    continue
                bid = float(row[fields["BidPrice1"]])
                ask = float(row[fields["AskPrice1"]])
                bid_quantity = int(row[fields["BidVol1"]])
                ask_quantity = int(row[fields["AskVol1"]])
                if continuous:
                    if bid == 0:
                        bid = float(contract["limit_up"])
                        if bid_depth > 1 and abs(float(row[fields["BidPrice2"]]) - bid) <= 1e-8:
                            bid_quantity += int(row[fields["BidVol2"]])
                    if ask == 0:
                        ask = float(contract["limit_down"])
                        if ask_depth > 1 and abs(float(row[fields["AskPrice2"]]) - ask) <= 1e-8:
                            ask_quantity += int(row[fields["AskVol2"]])
                # All supported prices are finite when positive; reject infinity.
                valid_mid = 0 < bid < float("inf") and 0 < ask < float("inf") and bid_quantity > 0 and ask_quantity > 0
                result["native_mid_valid_books"] += valid_mid
                result["usable_continuous_books"] += continuous and valid_mid and ask - bid > 1e-8
        result["raw_sha256"] = hashing.hash.hexdigest()
    return result


def main():
    profile = yaml.safe_load(PROFILE.read_text())
    preregistration = plan(profile)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registration = OUTPUT / "preregistration.yaml"
    if registration.exists():
        raise RuntimeError("preregistration already exists; never overwrite or rerun this scan implicitly")
    registration.write_text(yaml.safe_dump(preregistration, sort_keys=False))
    (OUTPUT / "H10_NOTES.preregistered.md").write_bytes(NOTES.read_bytes())
    started = time.monotonic()
    partitions = []
    totals = {
        str(symbol): {"scoped_books": 0, "continuous_status_books": 0, "usable_continuous_books": 0, "native_mid_valid_books": 0, "observed_slots": 0}
        for symbol in preregistration["universe"]
    }
    for day in preregistration["eligibility_calendar"]:
        contract_path = CONTRACT_ROOT / (day + ".csv")
        if contract_path.exists():
            with contract_path.open() as handle:
                contracts = {row["symbol"]: row for row in csv.DictReader(handle)}
            contract_hash = digest(contract_path)
        else:
            contracts = {}
            contract_hash = None
        for symbol in preregistration["universe"]:
            path = RAW_ROOT / symbol / (day + ".csv.zst")
            if not path.exists() or symbol not in contracts:
                partitions.append({"day": day, "symbol": symbol, "status": "missing_skip_empty_calendar_slot"})
                continue
            result = scan_file(path, day, symbol, contracts[symbol])
            result["contract_sha256"] = contract_hash
            partitions.append(result)
            totals[symbol]["observed_slots"] += 1
            for field in ["scoped_books", "continuous_status_books", "usable_continuous_books", "native_mid_valid_books"]:
                totals[symbol][field] += result[field]
        print(day, "complete", "elapsed", round(time.monotonic() - started, 2), flush=True)
    qualified = []
    for symbol, total in totals.items():
        denominator = total["scoped_books"]
        total["continuous_status_fraction"] = total["continuous_status_books"] / denominator if denominator else None
        total["usable_continuous_fraction"] = total["usable_continuous_books"] / denominator if denominator else None
        total["native_mid_valid_fraction"] = total["native_mid_valid_books"] / denominator if denominator else None
        total["qualified"] = denominator > 0 and total["usable_continuous_books"] * 10 >= denominator * 9
        if total["qualified"]:
            qualified.append(symbol)
    receipt = {
        "schema": "h10-unlabeled-eligibility-receipt-v1",
        "preregistration_sha256": digest(registration),
        "elapsed_seconds": time.monotonic() - started,
        "compressed_bytes_read": sum(result.get("compressed_bytes", 0) for result in partitions),
        "partitions": partitions,
        "symbol_totals": totals,
        "qualified_symbols_sorted": sorted(qualified),
        "selected_symbols": sorted(qualified)[:2],
        "next_support_dates": preregistration["support_dates"],
        "support_counts_not_yet_computed": True,
    }
    (OUTPUT / "eligibility-receipt.yaml").write_text(yaml.safe_dump(receipt, sort_keys=False))
    print("selected", receipt["selected_symbols"], "elapsed", round(receipt["elapsed_seconds"], 2), flush=True)
    for symbol in sorted(totals):
        print(symbol, totals[symbol], flush=True)


if __name__ == "__main__":
    main()
