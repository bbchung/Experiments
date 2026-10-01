"""Descriptive TXF report completion over existing tables; no market/model execution."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = (
    "actual_mid_gross",
    "actual_gross",
    "actual_spread_cost",
    "actual_fees",
    "actual_tax",
    "actual_net",
    "actual_stress",
    "long_net",
    "short_net",
    "expected50_net",
    "directional_increment",
)


MONTHS = {"202607", "202608"}


VIEWS = {"overall": (), "side": ("side",), "session": ("session",), "delivery": ("contract",), "side_session_delivery": ("side", "session", "contract")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_status(summary):
    """Support determines evaluability; a supported negative bound remains evaluated."""
    result = {}
    for arm, record in summary["arms"].items():
        gates, bounds = record.get("gates", {}), record.get("block_bounds", {})
        monthly = record.get("monthly", [])

        def numeric(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

        checks = {
            "months_complete": gates.get("months_complete") is True and len(monthly) == 2 and {m.get("month") for m in monthly} == MONTHS,
            "side_lineage": gates.get("side_lineage") is True,
            "monthly_support": gates.get("monthly_support") is True
            and len(monthly) == 2
            and all(
                numeric(m.get(field)) and m[field] >= minimum
                for m in monthly
                for field, minimum in [("score_origin_availability", 0.9), ("endpoint_support", 0.9), ("resolved", 100)]
            ),
            "bootstrap_supported": bounds.get("supported") is True,
            "known_day_means": numeric(bounds.get("known_day_means")) and bounds["known_day_means"] >= 20,
            **{
                f"{name}_defined_replicates": numeric(bounds.get(name, {}).get("undefined_fraction")) and 0 <= bounds[name]["undefined_fraction"] <= 0.01
                for name in ("event", "equal_day")
            },
        }
        reasons = [name + "_not_established" for name in ("months_complete", "side_lineage", "monthly_support", "bootstrap_supported") if not checks[name]]
        if not checks["known_day_means"]:
            reasons.append("known_day_means_missing_or_nonfinite" if not numeric(bounds.get("known_day_means")) else "fewer_than_20_known_validation_day_means")
        for name in ("event", "equal_day"):
            if not checks[name + "_defined_replicates"]:
                value = bounds.get(name, {}).get("undefined_fraction")
                reasons.append(name + ("_undefined_fraction_missing_or_nonfinite" if not numeric(value) else "_undefined_fraction_outside_0_to_0.01"))
        result[arm] = {
            "source_status": record.get("status"),
            "status": "evaluated" if all(checks.values()) else "inconclusive",
            "source_advances": record.get("advances"),
            "primary": record.get("primary"),
            "support_checks": checks,
            "reasons": reasons,
            "economic_gates_unchanged": {k: v for k, v in gates.items() if k not in ("months_complete", "side_lineage", "monthly_support")},
        }
    return {
        "schema": "txf-report-status-supplement-v1",
        "arms": result,
        "scope": "Descriptive support correction only; original advancement decisions and frozen files remain unchanged.",
    }


def validate_inputs(origins, events, daily):
    require(not origins.duplicated(["arm", "day", "SampleTime"]).any(), "Duplicate master origins")
    require(not events.duplicated(["arm", "day", "SampleTime", "delay_ms"]).any(), "Duplicate accepted intents")
    require(not daily.duplicated(["arm", "day", "delay_ms"]).any(), "Duplicate daily records")
    require(origins.available.isin([True, False]).all(), "Unknown availability encoding")
    require(events.side.isin([-1, 1]).all(), "Unknown accepted side")
    require(events.delay_ms.isin([50, 250]).all() and daily.delay_ms.isin([50, 250]).all(), "Unexpected delay")
    require(origins[["arm", "day", "session", "contract"]].notna().all().all(), "Unknown planned membership; cannot invent a delivery/session")
    require(np.isfinite(events.loc[events.status.eq("resolved"), list(FIELDS)].to_numpy(dtype=float)).all(), "Resolved event has unknown economic cell")
    accepted = origins.loc[origins.decision.eq("accepted"), ["arm", "day", "SampleTime", "session", "contract", "side"]]
    for delay in (50, 250):
        joined = accepted.merge(events.loc[events.delay_ms.eq(delay), accepted.columns], on=list(accepted.columns), how="outer", indicator=True, validate="one_to_one")
        require(joined._merge.eq("both").all(), "Accepted event membership differs from origins")
    grid = origins.groupby(["arm", "day"], observed=True).available.agg(planned_origins="size", available_origins="sum").reset_index()
    expected = grid.merge(pd.DataFrame({"delay_ms": [50, 250]}), how="cross")
    counts = events.assign(resolved=events.status.eq("resolved")).groupby(["arm", "day", "delay_ms"], observed=True).resolved.agg(accepted="size", resolved="sum").reset_index()
    expected = expected.merge(counts, on=["arm", "day", "delay_ms"], how="left").fillna({"accepted": 0, "resolved": 0})
    keys = ["arm", "day", "delay_ms"]
    counts = ["planned_origins", "available_origins", "accepted", "resolved"]
    joined = expected.merge(daily[keys + counts], on=keys, how="outer", suffixes=("_expected", "_source"), indicator=True, validate="one_to_one")
    require(joined._merge.eq("both").all() and all(joined[k + "_expected"].eq(joined[k + "_source"]).all() for k in counts), "Planned daily support differs from exported ledger")


def daily_view(origins, events, dimensions):
    origin_keys = ["arm", "day", *[key for key in dimensions if key != "side"]]
    keys = ["arm", "day", "delay_ms", *dimensions]
    grid = origins.groupby(origin_keys, observed=True, dropna=False).available.agg(planned_origins="size", available_origins="sum").reset_index()
    grid = grid.merge(pd.DataFrame({"delay_ms": [50, 250]}), how="cross")
    if "side" in dimensions:
        grid = grid.merge(pd.DataFrame({"side": [-1, 1]}), how="cross")
    known = events.status.eq("resolved")
    counts = events.assign(_known=known).groupby(keys, observed=True, dropna=False)._known.agg(accepted="size", resolved="sum")
    sums = events.loc[known].groupby(keys, observed=True, dropna=False)[list(FIELDS)].sum(min_count=1).add_suffix("_sum")
    rows = grid.merge(counts, on=keys, how="left", validate="one_to_one").merge(sums, on=keys, how="left", validate="one_to_one")
    rows[["accepted", "resolved"]] = rows[["accepted", "resolved"]].fillna(0).astype("int64")
    rows["fully_origin_observed"] = rows.available_origins.eq(rows.planned_origins)
    rows["known_zero"] = rows.accepted.eq(0) & rows.fully_origin_observed
    rows["unknown_intents"] = rows.accepted - rows.resolved
    rows["unconditional_daily_known"] = rows.fully_origin_observed & rows.unknown_intents.eq(0)
    rows["conditional_daily_known"] = rows.resolved.gt(0) | rows.known_zero
    for field in FIELDS:
        rows.loc[rows.known_zero, field + "_sum"] = 0.0
        rows[field + "_mean"] = rows[field + "_sum"] / rows.resolved.replace(0, np.nan)
        rows.loc[rows.known_zero, field + "_mean"] = 0.0
    rows["month"] = rows.day.str[:6]
    return rows.sort_values(keys).reset_index(drop=True)


def aggregate_view(daily, dimensions):
    keys = ["arm", "delay_ms", "period", *dimensions]
    monthly = daily.assign(period=daily.month)
    pooled = daily.loc[daily.month.isin(MONTHS)].assign(period="validation_pooled")
    rows = []
    for key, group in pd.concat([monthly, pooled], ignore_index=True).groupby(keys, observed=True, dropna=False, sort=True):
        row = dict(zip(keys, key, strict=True))
        row.update(
            planned_member_days=len(group),
            known_day_means=int(group.conditional_daily_known.sum()),
            unknown_day_means=int((~group.conditional_daily_known).sum()),
            known_zero_days=int(group.known_zero.sum()),
            unconditional_known_days=int(group.unconditional_daily_known.sum()),
            planned_origins=int(group.planned_origins.sum()),
            available_origins=int(group.available_origins.sum()),
            accepted=int(group.accepted.sum()),
            resolved=int(group.resolved.sum()),
            unknown_intents=int(group.unknown_intents.sum()),
        )
        for field in FIELDS:
            total = group[field + "_sum"].sum(min_count=1)
            row[field + "_conditional_sum"] = total
            row[field + "_equal_event_mean"] = total / row["resolved"] if row["resolved"] else np.nan
            row[field + "_equal_day_mean"] = group[field + "_mean"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def reporting_supplement(analysis, origins, events, days, summary):
    """Write the fixed equal-day/marginal reports without altering scientific gates.

    Membership uses all supplied planned origins. Missing source tails must be
    represented there before this helper is called; it cannot invent a calendar.
    """
    validate_inputs(origins, events, days)
    output = Path(analysis)
    paths = [output / "report-status-support.json"]
    paths += [output / f"report-{view}-{suffix}.csv" for view in VIEWS for suffix in ["daily", "summary"]]
    require(not any(path.exists() for path in paths), "Preserve existing report supplement")
    status = normalize_status(summary)
    output.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    for view, dimensions in VIEWS.items():
        table = daily_view(origins, events, dimensions)
        table.to_csv(output / f"report-{view}-daily.csv", index=False)
        aggregate_view(table, dimensions).to_csv(output / f"report-{view}-summary.csv", index=False)
    return {"files": [path.name for path in paths], "scope": "Conditional descriptive tables; original status, advancement and economics are unchanged."}
