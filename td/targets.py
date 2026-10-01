"""Pure projections of native future truth; no feature access or row filtering."""

from __future__ import annotations

from itertools import product

import numpy as np

from ...io import ContractError, digest


def column(name, horizon):
    return f"truth.{name}[{horizon}s]"


def candidates(cfg):
    result = []
    for family, h in product(cfg["targets"]["families"], cfg["truth"]["horizons_seconds"]):
        parameters = [{}]
        if family == "threshold":
            parameters = [{"ticks": b} for b in cfg["targets"]["threshold_ticks"]]
        elif family in {"triple_barrier", "first_passage"}:
            parameters = [{"up_ticks": u, "down_ticks": d} for u, d in product(cfg["truth"]["up_ticks"], cfg["truth"]["down_ticks"])]
        elif family == "excursion":
            parameters = [{"lambda": v} for v in cfg["targets"]["lambdas"]]
        for params in parameters:
            if len(result) >= cfg["budget"]["max_candidates"]:
                raise ContractError("TD candidate budget exceeded before fitting")
            definition = {"family": family, "horizon_seconds": h, **params}
            result.append(
                {
                    **definition,
                    "id": digest(definition)[:16],
                    "task": "binary_pair" if family in {"threshold", "triple_barrier", "first_passage"} else "regression",
                    "complexity": 1 + len(params),
                }
            )
    if len(result) > cfg["budget"]["max_candidates"]:
        raise ContractError("TD candidate budget exceeded before fitting")
    return sorted(result, key=lambda c: c["id"])


def project(frame, target, truth):
    h, family = target["horizon_seconds"], target["family"]

    def get(name):
        values = frame[column(name, h)].to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, np.nan)

    if family == "fixed_return":
        return get("return")[:, None]
    if family in {"mfe", "mae"}:
        return get(family)[:, None]
    if family == "excursion":
        return (get("mfe") - target["lambda"] * np.abs(get("mae")))[:, None]
    if family == "threshold":
        ret = get("return")
        return np.where(np.isfinite(ret)[:, None], np.column_stack([ret > target["ticks"], ret < -target["ticks"]]).astype(float), np.nan)
    u, d = [truth[key].index(target[key]) for key in ["up_ticks", "down_ticks"]]
    up, down = get(f"up_order.{u}"), get(f"down_order.{d}")
    # First passage asks whether each barrier is ever hit by H, irrespective of
    # the opposing hit. TB asks which is first. Never condition rows on a hit.
    if family == "first_passage":
        return np.column_stack([np.where(np.isfinite(up), up >= 0, np.nan), np.where(np.isfinite(down), down >= 0, np.nan)])
    if family == "triple_barrier":
        labels = np.column_stack([(up >= 0) & ((down < 0) | (up < down)), (down >= 0) & ((up < 0) | (down < up))]).astype(float)
        return np.where((np.isfinite(up) & np.isfinite(down))[:, None], labels, np.nan)
    raise ContractError("Unsupported target projection")


def truth_columns(truth):
    names = ["return", "mfe", "mae", "mfe_time", "mae_time"]
    names += [f"{side}_{kind}.{i}" for side in ["up", "down"] for i in range(len(truth[side + "_ticks"])) for kind in ["time", "order"]]
    return [column(name, h) for h in truth["horizons_seconds"] for name in names]
