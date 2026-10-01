"""Frozen explanatory quote accounting on H2 intents; no candidate or fill simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

VIEWS = ["R", "D"]
HORIZONS = [1000000, 5000000, 30000000]
DELAYS = [50000, 250000]
CLASSES = ["both_agree", "both_oppose", "one_agrees", "one_opposes", "offset", "unchanged"]
METRICS = ["P", "B", "A", "M", "T", "Q", "W0", "Wh", "delta_width", "aggressive_gross_bps", "aggressive_net_bps", "passive_gross_bps", "passive_fee_bps", "passive_net_bps"]
PRIMARY = ["B", "A", "M", "T", "Q", "passive_net_bps"]
CASH_METRICS = ["passive_net_cash", "aggressive_net_cash"]
KEYS = ["day", "product", "SampleTime"]


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value, sort_keys=True))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def decompose_events(frame):
    result = frame.copy()
    active = result.active.to_numpy(bool)
    known = result.known.to_numpy(bool)
    side = np.sign(result.score.to_numpy(float))
    mid = (result.entry_bid + result.entry_ask).to_numpy(float) / 2
    scale = 10000 / mid
    bid = side * (result.exit_bid - result.entry_bid).to_numpy(float) * scale
    ask = side * (result.exit_ask - result.entry_ask).to_numpy(float) * scale
    movement = (bid + ask) / 2
    shared = np.where((bid * ask) > 0, np.sign(bid) * np.minimum(np.abs(bid), np.abs(ask)), 0)
    w0 = (result.entry_ask - result.entry_bid).to_numpy(float) * scale
    wh = (result.exit_ask - result.exit_bid).to_numpy(float) * scale
    passive_entry = np.where(side > 0, result.entry_bid, result.entry_ask)
    passive_exit = np.where(side > 0, result.exit_bid, result.exit_ask)
    fee_cash = 100 + 2000 * (passive_entry + passive_exit) * 0.00002
    passive_gross_cash = side * 2000 * (passive_exit - passive_entry)
    aggressive_entry = np.where(side > 0, result.entry_ask, result.entry_bid)
    aggressive_exit = passive_exit
    aggressive_fee = 100 + 2000 * (aggressive_entry + aggressive_exit) * 0.00002
    aggressive_gross = movement - (w0 + wh) / 2
    origin = result.OriginMidPrice.to_numpy(float)
    values = {
        "P": side * (mid - origin) * scale,
        "B": bid,
        "A": ask,
        "M": movement,
        "T": shared,
        "Q": movement - shared,
        "W0": w0,
        "Wh": wh,
        "delta_width": wh - w0,
        "aggressive_gross_bps": aggressive_gross,
        "aggressive_net_bps": aggressive_gross - aggressive_fee * scale / 2000,
        "passive_gross_bps": passive_gross_cash * scale / 2000,
        "passive_fee_bps": fee_cash * scale / 2000,
        "passive_net_bps": (passive_gross_cash - fee_cash) * scale / 2000,
        "passive_net_cash": passive_gross_cash - fee_cash,
        "aggressive_net_cash": side * 2000 * (aggressive_exit - aggressive_entry) - aggressive_fee,
    }
    values["P"][~np.isfinite(origin) | (origin <= 0)] = np.nan
    for name, value in values.items():
        result[name] = np.where(~active, 0.0, np.where(known, value, np.nan))
    result["origin_mid_known"] = np.isfinite(origin) & (origin > 0)
    result["movement_class"] = np.select(
        [
            ~active,
            ~known,
            (bid > 0) & (ask > 0),
            (bid < 0) & (ask < 0),
            ((bid > 0) & (ask == 0)) | ((ask > 0) & (bid == 0)),
            ((bid < 0) & (ask == 0)) | ((ask < 0) & (bid == 0)),
            bid * ask < 0,
        ],
        ["inactive", "unknown", *CLASSES[:-1]],
        default="unchanged",
    )
    return result


def daily_accounting(events, coverage):
    records = []
    groups = {(str(day), view, int(horizon), int(delay)): part for (day, view, horizon, delay), part in events.groupby(["day", "view", "horizon_us", "delay_us"], sort=False)}
    empty = events.iloc[:0]
    for row in coverage.itertuples():
        day = str(row.day)
        for view in VIEWS:
            for horizon in HORIZONS:
                for delay in DELAYS:
                    subset = groups.get((day, view, horizon, delay), empty)
                    active = subset.loc[subset.active]
                    known = active.loc[active.known]
                    observed = bool(row.observed)
                    record = {"day": day, "month": day[:6], "view": view, "horizon_us": horizon, "delay_us": delay, "observed": observed}
                    record.update(
                        {
                            "union_intents": len(subset),
                            "active": len(active),
                            "known": len(known),
                            "unknown": len(active) - len(known),
                            "origin_mid_known": int(known.origin_mid_known.sum()),
                        }
                        if observed
                        else dict.fromkeys(["union_intents", "active", "known", "unknown", "origin_mid_known"], math.nan)
                    )
                    for metric in METRICS:
                        record[metric] = float(known[metric].mean()) if observed and len(known) else 0.0 if observed and not len(active) else math.nan
                    for metric in CASH_METRICS:
                        record[metric] = float(known[metric].sum()) if observed and len(known) else 0.0 if observed and not len(active) else math.nan
                    records.append(record)
    return pd.DataFrame(records)


def paired_population(events, dimension, values):
    """Return a separately labeled known-active intersection; never alter inputs."""
    selected = events.loc[events.active & events.known & events[dimension].isin(values)].copy()
    keys = ["day", "view", "SampleTime", "horizon_us" if dimension == "delay_us" else "delay_us"]
    complete = selected.groupby(keys)[dimension].transform("nunique") == len(values)
    return selected.loc[complete].copy()


def verify_population(events, intents, coverage):
    if events[[*KEYS, "view", "horizon_us", "delay_us"]].duplicated().any() or intents[KEYS].duplicated().any():
        raise ValueError("Duplicate frozen event or intent identity")
    required = {*KEYS, "OriginMidPrice", "R_active", "D_active", "R_score", "D_score"}
    if not required.issubset(intents):
        raise ValueError("Missing frozen origin metadata")
    joined = events.merge(intents[list(required)], on=KEYS, how="left", validate="many_to_one", indicator=True)
    if (joined._merge != "both").any():
        raise ValueError("An economic row is outside the frozen intent population")
    for view in VIEWS:
        rows = joined.loc[joined.view == view]
        if not np.array_equal(rows.active, rows[view + "_active"]) or not np.array_equal(rows.score.to_numpy().view(np.uint64), rows[view + "_score"].to_numpy().view(np.uint64)):
            raise ValueError("Original direction or activity changed")
    for horizon in HORIZONS:
        for delay in DELAYS:
            part = events.loc[(events.horizon_us == horizon) & (events.delay_us == delay)]
            if len(part) != 2 * len(intents):
                raise ValueError("A horizon/delay omitted original intents")
    expected_days = set(coverage.loc[coverage.observed, "day"])
    if not set(events.day).issubset(expected_days):
        raise ValueError("Economic events outside observed calendar")
    return joined.drop(columns=["_merge", "R_active", "D_active", "R_score", "D_score"])


def input_tables(parent):
    analysis = parent / "analysis"
    conditions = [("product", "==", "CDF"), ("day", ">=", "20260701"), ("day", "<=", "20260826")]
    events = pd.read_parquet(analysis / "all-events.parquet", filters=conditions)
    intents = pd.read_parquet(analysis / "all-intents-before-outcomes.parquet", filters=conditions)
    coverage = pd.read_csv(analysis / "coverage.csv", dtype={"day": str})
    coverage = coverage.loc[(coverage["product"] == "CDF") & coverage.day.between("20260701", "20260826")].copy()
    return verify_population(events, intents, coverage), coverage


def frozen_trace(parent, day, directory):
    work = parent / "native" / (day + "-CDF")
    path = work / directory / "keys.parquet"
    if path.is_file():
        frame = pd.read_parquet(path, columns=["SampleTime"])
        return frame.SampleTime.to_numpy(np.int64)
    status = read_yaml(work / "native-status" / (day + ".yaml"))
    summary = read_yaml(work / directory / "sample_summary.yaml")
    if (
        status.get("status") != "completed"
        or status.get("fatal_error")
        or summary.get("schema_version") != 2
        or any(summary.get(k) != 0 for k in ["sampled_rows", "emitted_rows", "pending_rows", "emitted_unresolved_rows", "unresolved_label_cells"])
    ):
        raise ValueError("Unproven missing native flip trace")
    return np.array([], dtype=np.int64)


def add_flip_context(events, parent):
    events = events.copy()
    for day, indices in events.groupby("day").groups.items():
        times = events.loc[indices, "SampleTime"].to_numpy(np.int64)
        lasts = []
        for name in ["target-flips", "reference-flips"]:
            trace = frozen_trace(parent, str(day), name)
            if np.any(np.diff(trace) < 0):
                raise ValueError("Unordered native flip trace")
            at = np.searchsorted(trace, times, side="left") - 1
            last = np.zeros(len(times), dtype=np.int64)
            good = at >= 0
            last[good] = trace[at[good]]
            lasts.append(last)
        target, reference = lasts
        events.loc[indices, "last_flip_order"] = np.select(
            [target == 0, reference == 0, target == reference, reference > target], ["both_absent", "reference_absent", "tied", "reference_later"], default="target_later"
        )
        events.loc[indices[(target == 0) & (reference > 0)], "last_flip_order"] = "target_absent"
    return events


def verify_parent_receipt(parent):
    receipt = read_yaml(parent / "analysis-receipt.yaml")
    if receipt.get("status") != "completed" or receipt.get("schema_version") != 1:
        raise ValueError("Unproven parent analysis completion")
    for key, path in [("original_freeze_sha256", parent / "freeze.yaml"), ("amendment_sha256", parent / "amendments/001-empty-flip-traces.yaml")]:
        if receipt.get(key) != digest(path):
            raise ValueError("Parent analysis provenance changed: " + key)
    for mapping in ["files", "native_receipts"]:
        if not receipt.get(mapping):
            raise ValueError("Missing parent artifact identity: " + mapping)
        for name, expected in receipt[mapping].items():
            if digest(parent / name) != expected:
                raise ValueError("Parent analysis artifact changed: " + name)
    if receipt.get("summary") != read_yaml(parent / "analysis/summary.yaml"):
        raise ValueError("Parent analysis summary differs from receipt")


def prepare(parent, output, protocol):
    if read_yaml(parent / "status.yaml").get("state") != "analysis_complete":
        raise ValueError("H2 analysis is not complete")
    verify_parent_receipt(parent)
    events, coverage = input_tables(parent)  # Identity/coverage only; no new outcome accounting.
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copy2(__file__, source / "basis_touch_diagnostic.py")
    shutil.copy2(protocol, output / "registration.md")
    tests = Path(__file__).resolve().parents[1] / "tests/test_basis_touch_diagnostic.py"
    if not tests.is_file():
        raise ValueError("Independent tests must exist before freezing")
    shutil.copy2(tests, source / tests.name)
    paths = [parent / name for name in ["registration.yaml", "freeze.yaml", "analysis-receipt.yaml", "amendments/001-empty-flip-traces.yaml"]]
    paths.extend(parent / "analysis" / name for name in ["all-events.parquet", "all-intents-before-outcomes.parquet", "coverage.csv", "daily.csv", "summary.yaml"])
    for day in coverage.loc[coverage.observed, "day"]:
        work = parent / "native" / (day + "-CDF")
        receipt = read_yaml(work / "receipt.yaml")
        for name, expected in receipt["files"].items():
            if digest(work / name) != expected:
                raise ValueError("Changed H2 native artifact: " + str(work / name))
        paths.append(work / "receipt.yaml")
        for directory in ["target-flips", "reference-flips"]:
            paths.extend(path for path in (work / directory).iterdir() if path.is_file())
        status = work / "native-status" / (day + ".yaml")
        if status.is_file():
            paths.append(status)
    freeze = {
        "schema_version": 1,
        "family": "h2-touch-accounting-followup-v1",
        "parent": str(parent),
        "source": {str(p.relative_to(output)): digest(p) for p in [output / "registration.md", *source.iterdir()]},
        "inputs": {str(path): digest(path) for path in paths},
        "population": {
            "event_rows": len(events),
            "planned_days": len(coverage),
            "observed_days": int(coverage.observed.sum()),
            "original_known_rows": int(events.known.sum()),
            "original_active_rows": int(events.active.sum()),
        },
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": version("pyarrow"), "PyYAML": version("PyYAML")},
    }
    write_yaml(output / "freeze.yaml", freeze)
    write_yaml(output / "status.yaml", {"state": "frozen_no_new_diagnostics_computed"})
    print(json.dumps(freeze["population"], indent=2))


def bootstrap_tables(daily):
    draws, intervals = [], []
    for delay in DELAYS:
        panel = daily.loc[(daily.view == "D") & (daily.horizon_us == 5000000) & (daily.delay_us == delay) & daily.observed].sort_values("day")
        rng = np.random.default_rng(1729)
        monthly_draws = {}
        for month, group in panel.groupby("month", sort=True):
            data = group[PRIMARY].to_numpy(float)
            starts = rng.integers(0, len(group), size=(4000, math.ceil(len(group) / 3)))
            ids = ((starts[:, :, None] + np.arange(3)) % len(group)).reshape(4000, -1)[:, : len(group)]
            sampled = data[ids].mean(axis=1)
            sampled[:, ~np.isfinite(data).all(axis=0)] = np.nan
            monthly_draws[month] = (len(group), sampled)
        monthly_draws["pooled"] = (len(panel), sum(n * data for n, data in monthly_draws.values()) / len(panel))
        for month, (count, sampled) in monthly_draws.items():
            for i, metric in enumerate(PRIMARY):
                finite = np.isfinite(sampled[:, i]).all()
                interval = np.quantile(sampled[:, i], [0.025, 0.975]) if finite else [math.nan, math.nan]
                subset = panel if month == "pooled" else panel.loc[panel.month == month]
                intervals.append(
                    {
                        "month": month,
                        "delay_us": delay,
                        "metric": metric,
                        "days": count,
                        "mean": float(subset[metric].mean()) if np.isfinite(subset[metric]).all() else math.nan,
                        "lower_2_5": float(interval[0]),
                        "upper_97_5": float(interval[1]),
                        "role": "descriptive_not_promotion",
                    }
                )
            draws.extend({"month": month, "delay_us": delay, "draw": i, **dict(zip(PRIMARY, row, strict=True))} for i, row in enumerate(sampled))
    return pd.DataFrame(intervals), pd.DataFrame(draws)


def summarize(daily, events, analysis):
    observed = daily.loc[daily.observed]
    months = observed.groupby(["month", "view", "horizon_us", "delay_us"])
    monthly = months[METRICS].agg(lambda values: float(values.mean()) if np.isfinite(values).all() else math.nan).reset_index()
    support = months[["active", "known", "unknown", "origin_mid_known"]].sum().reset_index()
    monthly = monthly.merge(support, on=["month", "view", "horizon_us", "delay_us"])
    monthly["known_fraction"] = monthly.known / monthly.active
    cash = months[CASH_METRICS].agg(lambda values: float(values.sum()) if np.isfinite(values).all() else math.nan).reset_index()
    days = months.size().rename("observed_days").reset_index()
    known_days = months[METRICS].count().rename(columns={name: name + "_known_days" for name in METRICS}).reset_index()
    monthly.merge(cash).merge(days).merge(known_days).to_csv(analysis / "monthly.csv", index=False)
    concentration = []
    for keys, group in months:
        for metric in [*PRIMARY, "aggressive_net_bps", *CASH_METRICS]:
            finite = np.isfinite(group[metric]).all()
            values = group[metric].to_numpy()
            positive = np.maximum(values, 0)
            concentration.append(
                {
                    **dict(zip(["month", "view", "horizon_us", "delay_us"], keys, strict=True)),
                    "metric": metric,
                    "positive_days": int((values > 0).sum()),
                    "negative_days": int((values < 0).sum()),
                    "zero_days": int((values == 0).sum()),
                    "unknown_days": int((~np.isfinite(values)).sum()),
                    "leave_best_day_out_mean": float((values.sum() - values.max()) / (len(values) - 1)) if finite and len(values) > 1 else math.nan,
                    "max_positive_day_share": float(positive.max() / positive.sum()) if finite and positive.sum() > 0 else math.nan,
                }
            )
    pd.DataFrame(concentration).to_csv(analysis / "day-concentration.csv", index=False)
    rows = []
    for keys, group in events.groupby(["day", "view", "horizon_us", "delay_us"], sort=False):
        known = group.loc[group.active & group.known]
        for category in CLASSES:
            part = known.loc[known.movement_class == category]
            record = {**dict(zip(["day", "view", "horizon_us", "delay_us"], keys, strict=True)), "class": category, "known_active": len(known), "class_count": len(part)}
            for metric in ["B", "A", "M", "T", "Q", "delta_width", "passive_net_bps"]:
                record["contribution_" + metric] = float(part[metric].sum() / len(known)) if len(known) else 0.0 if not group.active.any() else math.nan
                record["conditional_" + metric] = float(part[metric].mean()) if len(part) else math.nan
            rows.append(record)
    classes = pd.DataFrame(rows)
    classes["month"] = classes.day.str[:6]
    classes.to_csv(analysis / "class-daily.csv", index=False)
    class_keys = ["month", "view", "horizon_us", "delay_us", "class"]
    contributions = [name for name in classes if name.startswith("contribution_")]
    class_monthly = classes.groupby(class_keys)[contributions].agg(lambda values: float(values.mean()) if np.isfinite(values).all() else math.nan)
    class_monthly = class_monthly.join(classes.groupby(class_keys)[["class_count", "known_active"]].sum())
    class_monthly["fraction_known_active"] = class_monthly.class_count / class_monthly.known_active
    class_monthly.to_csv(analysis / "class-monthly-contributions.csv")
    active = events.loc[events.active & events.known].copy()
    for context, cells in [
        ("spread_cell", [0, 1]),
        ("quiet_target_after_reference_flip", [False, True]),
        ("last_flip_order", ["both_absent", "target_absent", "reference_absent", "tied", "reference_later", "target_later"]),
    ]:
        day = active.groupby(["day", "month", "view", "horizon_us", "delay_us", context])[METRICS].mean().reset_index()
        counts = active.groupby(["month", "view", "horizon_us", "delay_us", context]).size().rename("events")
        context_keys = ["month", "view", "horizon_us", "delay_us", context]
        complete = pd.MultiIndex.from_product([["202607", "202608"], VIEWS, HORIZONS, DELAYS, cells], names=context_keys)
        days = day.groupby(context_keys).size().rename("observed_cell_days")
        day.groupby(context_keys)[METRICS].mean().join(counts).join(days).reindex(complete).fillna({"events": 0, "observed_cell_days": 0}).to_csv(
            analysis / ("context-" + context + ".csv")
        )


def run(output):
    freeze = read_yaml(output / "freeze.yaml")
    if Path(__file__).resolve() != output / "source/basis_touch_diagnostic.py":
        raise ValueError("Execute the frozen study source")
    for name, expected in freeze["source"].items():
        if digest(output / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    for name, expected in freeze["inputs"].items():
        if digest(name) != expected:
            raise ValueError("Frozen input changed: " + name)
    events, coverage = input_tables(Path(freeze["parent"]))
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=False)
    original = events[[*KEYS, "view", "horizon_us", "delay_us", "active", "known", "policy_known", "reason"]].copy()
    events = decompose_events(add_flip_context(events, Path(freeze["parent"])))
    if not original.equals(events[original.columns]):
        raise ValueError("Original coverage or intent identity changed")
    check = events.active & events.known
    for computed, original_name in [("aggressive_gross_bps", "gross_bps"), ("aggressive_net_bps", "net_bps")]:
        if not np.allclose(events.loc[check, computed], events.loc[check, original_name], rtol=1e-10, atol=1e-10):
            raise ValueError("Original H2 economic accounting mismatch")
    if not np.allclose(events.loc[check, "T"] + events.loc[check, "Q"], events.loc[check, "M"]):
        raise ValueError("Shared/residual accounting mismatch")
    coverage.to_csv(analysis / "coverage.csv", index=False)
    events.to_parquet(analysis / "events.parquet", index=False)
    daily = daily_accounting(events, coverage)
    daily.to_csv(analysis / "daily.csv", index=False)
    summarize(daily, events, analysis)
    intervals, draws = bootstrap_tables(daily)
    intervals.to_csv(analysis / "primary-descriptive-intervals.csv", index=False)
    draws.to_parquet(analysis / "primary-descriptive-draws.parquet", index=False)
    pairs = {"delay-5s": paired_population(events.loc[events.horizon_us == 5000000], "delay_us", DELAYS), "checkpoint-horizons": paired_population(events, "horizon_us", HORIZONS)}
    pair_coverage = []
    for name, paired in pairs.items():
        paired.to_parquet(analysis / ("paired-" + name + ".parquet"), index=False)
        paired_daily = paired.groupby(["day", "view", "horizon_us", "delay_us"])[METRICS].mean().reset_index()
        paired_daily["month"] = paired_daily.day.str[:6]
        paired_daily.to_csv(analysis / ("paired-" + name + "-daily.csv"), index=False)
        paired_daily.groupby(["month", "view", "horizon_us", "delay_us"])[METRICS].mean().to_csv(analysis / ("paired-" + name + "-monthly.csv"))
        keys = ["month", "view", "horizon_us", "delay_us"]
        original_counts = events.loc[events.active].groupby(keys).size().rename("original_active")
        known_counts = events.loc[events.active & events.known].groupby(keys).size().rename("original_known")
        counts = paired.groupby(keys).size().rename("paired_known")
        table = pd.concat([original_counts, known_counts, counts], axis=1).fillna({"paired_known": 0})
        table["intersection"] = name
        pair_coverage.append(table.reset_index())
        if name == "checkpoint-horizons":
            patterns = paired.pivot(index=["day", "view", "SampleTime", "delay_us"], columns="horizon_us", values="movement_class").reset_index()
            names = ["class_1s", "class_5s", "class_30s"]
            patterns = patterns.rename(columns=dict(zip(HORIZONS, names, strict=True)))
            patterns["month"] = patterns.day.str[:6]
            patterns.to_parquet(analysis / "paired-checkpoint-pattern-events.parquet", index=False)
            patterns.groupby(["month", "view", "delay_us", *names]).size().rename("count").to_csv(analysis / "paired-checkpoint-patterns.csv")
    pd.concat(pair_coverage).to_csv(analysis / "paired-coverage.csv", index=False)
    write_yaml(
        analysis / "summary.yaml",
        {
            "family": freeze["family"],
            "evidence_role": "descriptive_research_validation_no_candidate_or_fill_claim",
            "H2_decisions_unchanged": True,
            "population_unchanged": True,
            "population": freeze["population"],
            "bootstrap": "4000 circular3day withinmonth seed1729 descriptive95percent",
            "origin_limit": "Native origin midpoint only; no pre-arrival touch/spread attribution",
            "passive_limit": "Hypothetical instantaneous entry; no fill, queue or adverse-selection proof and no selective-policy bound",
        },
    )
    write_yaml(output / "results-manifest.yaml", {str(p.relative_to(output)): digest(p) for p in analysis.iterdir() if p.is_file()})
    write_yaml(output / "status.yaml", {"state": "analysis_complete_no_candidate_promotion"})
    print("Completed fixed quote-accounting diagnostic; no candidate or execution claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.protocol is None:
            parser.error("prepare requires --protocol")
        prepare(args.parent.resolve(), args.output.resolve(), args.protocol.resolve())
    else:
        if str(args.parent.resolve()) != read_yaml(args.output / "freeze.yaml")["parent"]:
            raise ValueError("Requested parent differs from frozen study")
        run(args.output.resolve())


if __name__ == "__main__":
    main()
