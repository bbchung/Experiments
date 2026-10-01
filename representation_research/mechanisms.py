"""Causal, sampled-origin mechanisms for the frozen endpoint research contract.

These transformations describe the declared origin grid, not continuous-time
occupation or full-session statistics. They use observed mids and no labels.
Missing endpoints, dates and peers remain missing; arithmetic never adds epsilon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MICROS = 1_000_000
DAY_MICROS = 86_400 * MICROS
HORIZONS = (60, 300)
GRAPH_SCHEMA = "sampled-origin-mechanisms-v1"


def _inputs(rows, mid, tick_bps=None):
    required = ["day", "symbol", "SampleTime"]
    if any(name not in rows for name in required):
        raise ValueError("rows require day, symbol and SampleTime")
    keys = rows[required].copy()
    if keys.isna().any().any():
        raise ValueError("origin keys must not be missing")
    keys["day"] = keys.day.astype(str)
    keys["symbol"] = keys.symbol.astype(str)
    if not keys.day.str.fullmatch(r"\d{8}").all() or (keys.day < "20260101").any():
        raise ValueError("origin days must be YYYYMMDD on or after 20260101")
    times = keys.SampleTime.to_numpy()
    if not np.issubdtype(times.dtype, np.integer):
        raise ValueError("SampleTime must contain integer native microseconds")
    if keys.duplicated(required).any():
        raise ValueError("duplicate origin keys")
    mid = np.asarray(mid, dtype=np.float64)
    if mid.shape != (len(keys),):
        raise ValueError("mid must align one to one with rows")
    if tick_bps is not None:
        tick_bps = np.asarray(tick_bps, dtype=np.float64)
        if tick_bps.shape != mid.shape:
            raise ValueError("tick_bps must align one to one with rows")
    groups = []
    for positions in keys.groupby(["day", "symbol"], sort=True).indices.values():
        positions = np.asarray(positions)
        groups.append(positions[np.argsort(times[positions], kind="stable")])
    return keys, mid, tick_bps, groups


def _log_move(end, start):
    end, start = np.broadcast_arrays(np.asarray(end, dtype=np.float64), np.asarray(start, dtype=np.float64))
    result = np.full(end.shape, np.nan)
    valid = np.isfinite(end) & np.isfinite(start) & (end > 0.0) & (start > 0.0)
    # Equal represented prices produce exactly zero, before any normalization.
    result[valid] = np.log1p((end[valid] - start[valid]) / start[valid]) * 10_000.0
    return result


def _past_returns(keys, mid, groups):
    result = {horizon: np.full(len(keys), np.nan) for horizon in HORIZONS}
    times = keys.SampleTime.to_numpy()
    for positions in groups:
        group_times = times[positions]
        for horizon in HORIZONS:
            sought = group_times - horizon * MICROS
            previous = np.searchsorted(group_times, sought)
            exact = (previous < len(positions)) & (group_times[np.minimum(previous, len(positions) - 1)] == sought)
            result[horizon][positions[exact]] = _log_move(mid[positions[exact]], mid[positions[previous[exact]]])
    return result


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def fit_graph(rows, mid, *, topology_train_days, peer_count=3):
    """Fit positive peers stable across two train-day halves, without labels.

    Correlations and beta use same-day, exact-origin 60 s log moves. Scales are
    empirical median absolute past moves at each declared horizon, not a
    square-root-time extrapolation. Missing requested days are recorded/skipped.
    """
    keys, mid, _, _ = _inputs(rows, mid)
    requested = sorted({str(day) for day in topology_train_days})
    if not requested or any(len(day) != 8 or not day.isdigit() or day < "20260101" for day in requested):
        raise ValueError("declare nonempty 2026 topology_train_days")
    if not isinstance(peer_count, int) or peer_count < 1:
        raise ValueError("peer_count must be positive")
    train = keys.day.isin(requested).to_numpy()
    actual = sorted(set(keys.day[train]))
    if len(actual) < 2:
        raise ValueError("graph fitting requires at least two observed train days")
    # Slice before computing any fitted quantity. Other days cannot influence
    # either topology or a normalizer, even if they are present in the input.
    selected = keys.loc[train].copy()
    train_mid = mid[train]
    selected, train_mid, _, train_groups = _inputs(selected, train_mid)
    returns = _past_returns(selected, train_mid, train_groups)
    observed_steps = [int(np.min(np.diff(selected.SampleTime.to_numpy()[positions]))) for positions in train_groups if len(positions) > 1]
    panel = selected.assign(move=returns[60]).pivot(index=["day", "SampleTime"], columns="symbol", values="move")
    halfway = len(actual) // 2
    first = panel.loc[panel.index.get_level_values("day").isin(actual[:halfway])]
    second = panel.loc[panel.index.get_level_values("day").isin(actual[halfway:])]
    minimum_pairs = 32
    first_corr = first.corr(min_periods=minimum_pairs)
    second_corr = second.corr(min_periods=minimum_pairs)
    records = {}
    for symbol in panel.columns:
        stable = []
        for peer in panel.columns:
            if peer == symbol:
                continue
            left, right = first_corr.loc[symbol, peer], second_corr.loc[symbol, peer]
            if np.isfinite(left) and np.isfinite(right) and left > 0.0 and right > 0.0:
                stable.append((min(left, right), str(peer), float(left), float(right)))
        stable.sort(key=lambda record: (-record[0], record[1]))
        peers = [record[1] for record in stable[:peer_count]]
        beta = None
        paired_rows = 0
        if peers:
            complete = panel[[symbol, *peers]].dropna()
            paired_rows = len(complete)
            if paired_rows >= minimum_pairs:
                target = complete[symbol].to_numpy()
                basket = complete[peers].to_numpy().mean(axis=1)
                centered = basket - basket.mean()
                square = float(centered @ centered)
                if square > 0.0:
                    beta = _finite_number(float(centered @ (target - target.mean())) / square)
        symbol_positions = selected.symbol.eq(symbol).to_numpy()
        scales = {}
        for horizon in HORIZONS:
            observed = returns[horizon][symbol_positions]
            observed = observed[np.isfinite(observed)]
            scales[str(horizon)] = _finite_number(np.median(np.abs(observed))) if len(observed) else None
        records[str(symbol)] = {
            "peers": peers,
            "peer_stability": [{"symbol": record[1], "first_half_correlation": record[2], "second_half_correlation": record[3]} for record in stable[:peer_count]],
            "beta": beta,
            "paired_rows": paired_rows,
            "variation_bps": scales,
        }
    return {
        "schema": GRAPH_SCHEMA,
        "requested_train_days": requested,
        "observed_train_days": actual,
        "missing_train_days": sorted(set(requested) - set(actual)),
        "train_halves": [actual[:halfway], actual[halfway:]],
        "peer_count": peer_count,
        "minimum_pairs_per_half": minimum_pairs,
        "resolution_floor": "one_current_target_tick_bps",
        "origin_interval_micros": min(observed_steps) if observed_steps else 10 * MICROS,
        "origin_anchor_clock_micros": int(np.min(selected.SampleTime.to_numpy() % DAY_MICROS)),
        "symbols": records,
    }


def _denominator(scale, ticks):
    ticks = np.asarray(ticks, dtype=np.float64)
    scale = np.broadcast_to(np.asarray(np.nan if scale is None else scale, dtype=np.float64), ticks.shape)
    valid = np.isfinite(scale) & (scale >= 0.0) & np.isfinite(ticks) & (ticks > 0.0)
    return np.where(valid, np.maximum(scale, ticks), np.nan)


def _graph_features(keys, mid, ticks, groups, returns, graph):
    columns = [f"{name}_{horizon}s" for horizon in HORIZONS for name in ("common_innovation", "target_residual", "beta_gap", "peer_coverage")]
    columns.append("sampled_target_held_age_fraction")
    values = {column: np.full(len(keys), np.nan) for column in columns}
    times = keys.SampleTime.to_numpy()
    held = np.full(len(keys), np.nan)
    for positions in groups:
        group_mid = mid[positions]
        valid = np.isfinite(group_mid) & (group_mid > 0.0)
        gaps = np.diff(times[positions]) != graph["origin_interval_micros"]
        change = valid & np.r_[True, (~valid[:-1]) | gaps | (group_mid[1:] != group_mid[:-1])]
        # An invalid sample breaks the observed chain; the next sample anchors it.
        anchors = np.maximum.accumulate(np.where(change, np.arange(len(positions)), -1))
        usable = valid & (anchors >= 0)
        held[positions[usable]] = times[positions[anchors[usable]]]
        values["sampled_target_held_age_fraction"][positions[usable]] = np.minimum((times[positions[usable]] - held[positions[usable]]) / (300 * MICROS), 1.0)
    for day_positions in keys.groupby("day", sort=True).indices.values():
        day_positions = np.asarray(day_positions)
        day_keys = keys.iloc[day_positions]
        symbols = sorted(set(day_keys.symbol))
        symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
        day_times = np.unique(times[day_positions])
        time_index = np.searchsorted(day_times, times[day_positions])
        column_index = np.array([symbol_index[symbol] for symbol in day_keys.symbol])
        price_panel = np.full((len(day_times), len(symbols)), np.nan)
        price_panel[time_index, column_index] = mid[day_positions]
        return_panels = {}
        for horizon in HORIZONS:
            panel = np.full_like(price_panel, np.nan)
            panel[time_index, column_index] = returns[horizon][day_positions]
            return_panels[horizon] = panel
        for symbol, selected_positions in day_keys.groupby("symbol", sort=True).indices.items():
            positions = day_positions[np.asarray(selected_positions)]
            record = graph["symbols"].get(symbol)
            if record is None or record["beta"] is None or not record["peers"]:
                continue
            peers = record["peers"]
            rows_at = np.searchsorted(day_times, times[positions])
            peer_columns = [symbol_index.get(peer) for peer in peers]
            for horizon in HORIZONS:
                peer_moves = np.column_stack([return_panels[horizon][rows_at, column] if column is not None else np.full(len(positions), np.nan) for column in peer_columns])
                available = np.isfinite(peer_moves)
                values[f"peer_coverage_{horizon}s"][positions] = available.mean(axis=1)
                complete = available.all(axis=1)
                denominator = _denominator(record["variation_bps"][str(horizon)], ticks[positions])
                common = record["beta"] * peer_moves.mean(axis=1)
                common[~complete] = np.nan
                values[f"common_innovation_{horizon}s"][positions] = common / denominator
                values[f"target_residual_{horizon}s"][positions] = (returns[horizon][positions] - common) / denominator
                anchor_times = np.maximum(held[positions], times[positions] - horizon * MICROS)
                anchors = np.searchsorted(day_times, anchor_times)
                exact = (anchors < len(day_times)) & np.isfinite(anchor_times)
                exact &= day_times[np.minimum(anchors, len(day_times) - 1)] == anchor_times
                gap_moves = np.column_stack(
                    [
                        _log_move(price_panel[rows_at, column], price_panel[np.minimum(anchors, len(day_times) - 1), column])
                        if column is not None
                        else np.full(len(positions), np.nan)
                        for column in peer_columns
                    ]
                )
                gap_complete = np.isfinite(gap_moves).all(axis=1) & exact
                gap = record["beta"] * gap_moves.mean(axis=1)
                gap[~gap_complete] = np.nan
                values[f"beta_gap_{horizon}s"][positions] = gap / denominator
    return pd.DataFrame(values, index=keys.index)


def _session_features(keys, mid, ticks, groups, returns, graph, config):
    history_days = int(config.get("history_days", 20))
    minimum_days = int(config.get("minimum_days", 5))
    step = int(config.get("sample_interval_seconds", 10)) * MICROS
    if history_days < minimum_days or minimum_days < 1 or step <= 0 or 60 * MICROS % step:
        raise ValueError("invalid prior history/minimum days or sampled interval")
    observed_days = sorted(set(keys.day))
    calendar_days = sorted({str(day) for day in config.get("calendar_days", observed_days)})
    if not set(observed_days).issubset(calendar_days):
        raise ValueError("prior calendar must cover every observed day")
    if any(len(day) != 8 or not day.isdigit() or day < "20260101" for day in calendar_days):
        raise ValueError("prior calendar days must start at 20260101")
    clock = keys.SampleTime.to_numpy() % DAY_MICROS
    current = np.abs(returns[60])
    displacement = np.full(len(keys), np.nan)
    anchor_clock = int(config.get("origin_anchor_clock_micros", graph["origin_anchor_clock_micros"]))
    for positions in groups:
        anchors = positions[clock[positions] == anchor_clock]
        if len(anchors):
            later = positions[keys.SampleTime.to_numpy()[positions] >= keys.SampleTime.to_numpy()[anchors[0]]]
            displacement[later] = _log_move(mid[later], mid[anchors[0]])
    reference = np.full(len(keys), np.nan)
    displacement_reference = np.full(len(keys), np.nan)
    coverage = np.full(len(keys), np.nan)
    for symbol_positions in keys.groupby("symbol", sort=True).indices.values():
        positions = np.asarray(symbol_positions)
        symbol_keys = keys.iloc[positions]
        lookup = pd.MultiIndex.from_arrays([symbol_keys.day, clock[positions]])
        for source, destination, is_activity in ((current, reference, True), (np.abs(displacement), displacement_reference, False)):
            panel = pd.DataFrame({"day": symbol_keys.day.to_numpy(), "clock": clock[positions], "value": source[positions]}).pivot(index="day", columns="clock", values="value")
            panel = panel.reindex(calendar_days)
            rolling = panel.shift(1).rolling(history_days, min_periods=minimum_days)
            prior = rolling.median()
            stacked = prior.stack(future_stack=True)
            destination[positions] = stacked.reindex(lookup).to_numpy()
            if is_activity:
                counts = panel.shift(1).rolling(history_days, min_periods=1).count()
                slots = np.minimum(np.arange(len(calendar_days)), history_days)
                fraction = counts.div(pd.Series(np.where(slots > 0, slots, np.nan), index=counts.index), axis=0)
                coverage[positions] = fraction.stack(future_stack=True).reindex(lookup).to_numpy()
    activity = current / _denominator(reference, ticks)
    # Persist excess intensity rather than counting brittle ratio > 1 flags.
    # Equal current/reference activity is exact zero before normalization;
    # a tiny rounding residual remains a tiny contribution to the average.
    excess_activity = np.clip((current - reference) / _denominator(reference, ticks), 0.0, 1.0)
    signed_displacement = displacement / _denominator(displacement_reference, ticks)
    persistence = np.full(len(keys), np.nan)
    count = 60 * MICROS // step
    times = keys.SampleTime.to_numpy()
    for positions in groups:
        if len(positions) < count:
            continue
        history = np.lib.stride_tricks.sliding_window_view(excess_activity[positions], count)
        group_times = times[positions]
        valid_grid = np.diff(group_times) == step
        cumulative = np.r_[0, np.cumsum(~valid_grid)]
        ends = np.arange(count - 1, len(positions))
        starts = ends - count + 1
        complete = np.isfinite(history).all(axis=1) & (cumulative[ends] == cumulative[starts])
        persistence[positions[ends[complete]]] = history[complete].mean(axis=1)
    return pd.DataFrame(
        {
            "same_clock_activity_surprise": activity,
            "persistent_excess_activity": persistence,
            "sampled_day_displacement_surprise": signed_displacement,
            "prior_activity_scale_ticks": reference / np.where(np.isfinite(ticks) & (ticks > 0.0), ticks, np.nan),
            "prior_day_slot_coverage": coverage,
        },
        index=keys.index,
    )


def _occupation_features(keys, mid, ticks, groups, graph, config):
    step = int(config.get("sample_interval_seconds", 10)) * MICROS
    block = 120 * MICROS
    if step <= 0 or block % step:
        raise ValueError("120 s occupation blocks require a divisible sample interval")
    count = block // step
    columns = ["quantile_transport", "resolution_weighted_quantile_coherence", "current_accepted_median_gap", "old_range_resolution_support", "accepted_dispersion_ratio"]
    values = {column: np.full(len(keys), np.nan) for column in columns}
    times = keys.SampleTime.to_numpy()
    for positions in groups:
        if len(positions) <= 2 * count:
            continue
        prices = mid[positions]
        log_prices = np.full(len(prices), np.nan)
        valid_price = np.isfinite(prices) & (prices > 0.0)
        log_prices[valid_price] = np.log(prices[valid_price])
        # Each block has count sampled points. Both end strictly before the
        # current origin; current price has its own, separately named output.
        windows = np.lib.stride_tricks.sliding_window_view(log_prices[:-1], 2 * count)
        current_positions = np.arange(2 * count, len(positions))
        group_times = times[positions]
        cumulative = np.r_[0, np.cumsum(np.diff(group_times) != step)]
        complete = np.isfinite(windows).all(axis=1) & valid_price[current_positions]
        complete &= cumulative[current_positions] == cumulative[current_positions - 2 * count]
        if not complete.any():
            continue
        destinations = positions[current_positions[complete]]
        selected = windows[complete]
        old, recent = selected[:, :count], selected[:, count:]
        old_quantiles = np.quantile(old, [0.25, 0.5, 0.75], axis=1).T
        new_quantiles = np.quantile(recent, [0.25, 0.5, 0.75], axis=1).T
        difference = (new_quantiles - old_quantiles) * 10_000.0
        record = graph["symbols"].get(str(keys.symbol.iloc[positions[0]]))
        scale = record["variation_bps"]["300"] if record is not None else None
        denominator = _denominator(scale, ticks[destinations])
        values["quantile_transport"][destinations] = difference.mean(axis=1) / denominator
        # A representational rounding residual must not become a full ±1 sign.
        # Each quantile votes in proportion to its shift up to one physical
        # target tick, after which its contribution saturates at ±1. Invalid
        # resolution stays missing rather than granting an arbitrary sign.
        resolution = _denominator(0.0, ticks[destinations])[:, None]
        values["resolution_weighted_quantile_coherence"][destinations] = (difference / np.maximum(resolution, np.abs(difference))).mean(axis=1)
        values["current_accepted_median_gap"][destinations] = (log_prices[current_positions[complete]] - new_quantiles[:, 1]) * 10_000.0 / denominator
        # Range membership also needs continuous resolution semantics: a tiny
        # midpoint rounding error outside a closed range is not lost support.
        outside_bps = np.maximum(np.maximum(old_quantiles[:, 0, None] - recent, recent - old_quantiles[:, 2, None]), 0.0) * 10_000.0
        values["old_range_resolution_support"][destinations] = np.clip(1.0 - outside_bps / resolution, 0.0, 1.0).mean(axis=1)
        old_width = (old_quantiles[:, 2] - old_quantiles[:, 0]) * 10_000.0
        new_width = (new_quantiles[:, 2] - new_quantiles[:, 0]) * 10_000.0
        values["accepted_dispersion_ratio"][destinations] = new_width / _denominator(old_width, ticks[destinations])
    return pd.DataFrame(values, index=keys.index)


def transform(rows, mid, tick_bps, graph, prior_config=None):
    """Return 19 mechanism fields in three groups, aligned to the input index.

    Freeze graph and prior_config before comparisons. A future append cannot
    change earlier fields. Same-clock priors shift whole date slots before
    estimating their reference; no current-day observation enters its baseline.
    """
    if graph.get("schema") != GRAPH_SCHEMA or graph.get("resolution_floor") != "one_current_target_tick_bps":
        raise ValueError("unsupported frozen graph or normalization contract")
    config = {} if prior_config is None else dict(prior_config)
    keys, mid, ticks, groups = _inputs(rows, mid, tick_bps)
    returns = _past_returns(keys, mid, groups)
    return {
        "graph": _graph_features(keys, mid, ticks, groups, returns, graph),
        "session": _session_features(keys, mid, ticks, groups, returns, graph, config),
        "occupation": _occupation_features(keys, mid, ticks, groups, graph, config),
    }
