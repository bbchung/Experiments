"""python3.13 -m AstraResearch.Experiments.representation_research.runner --profile PROFILE COMMAND.

The versioned profile owns every comparison choice; commands do not override
targets, role boundaries, model budgets, feature arms, or the acceptance rule.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import pyarrow
import scipy
from catboost import CatBoostClassifier, FeaturesData, Pool

from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from . import contract, data, evaluation, mechanisms


def output(profile):
    return Path(profile["paths"]["output"])


def core(matrix, names, rows):
    """Small semantic control; tick units match the fixed native endpoint target."""
    book = [
        f"BookStructure.0.{f}.0"
        for f in (
            "imbalance_touch",
            "imbalance_near",
            "imbalance_total",
            "gap_front_imbalance",
            "gap_width_imbalance",
            "book_profile_hhi_gap",
            "fair_price_micro_offset_spreads",
            "fair_price_depth_offset_spreads",
        )
    ]
    tick = [n for n in names if n.startswith("TickRegime.")]
    bounded = [f"RawDirectionPath.0.{f}.0" for f in ("trade_imbalance", "book_imbalance", "touch_volume_imbalance", "deep_volume_imbalance")]
    ticks = ["RawDirectionPath.0.sticky_move_ticks.0", "StickyPriceRealizedVolatility.0.realized_vol.4"]
    selected = book + tick + bounded + ticks
    if not set(selected).issubset(names):
        raise ContractError(f"Core control native inputs missing: {set(selected) - set(names)}")
    positions = {name: i for i, name in enumerate(names)}
    result = pd.DataFrame(np.asarray(matrix[:, [positions[n] for n in selected]]), columns=selected)
    # Price displacement and realized variation are in the user's common five-
    # tick event unit, rather than dollars or raw volume. Preserve all sentinels.
    for n in ticks:
        result[n] = result[n] / 5.0
    local_seconds = (rows.SampleTime.to_numpy(dtype=np.int64) // 1_000_000 + 8 * 3600) % 86400
    result["Origin.session_fraction"] = (local_seconds - 9 * 3600) / (4.5 * 3600)
    result["Origin.book_age_seconds"] = (rows.SampleTime.to_numpy() - rows.SampleBookTime.to_numpy()) / 1_000_000
    return result


def prepare_features(profile):
    root = output(profile)
    parent = Path(profile["paths"]["parent_study"])
    sets, role_rows, mids, counts = [], [], [], []
    for role in ("train", "tune", "forward"):
        matrix, rows, names, _ = data.load_projection(parent / role)
        sets.append(core(matrix, names, rows))
        role_rows.append(rows)
        mids.append(np.load(root / "shared" / f"mid-{role}.npy", mmap_mode="r"))
        counts.append(len(rows))
    rows = pd.concat(role_rows, ignore_index=True)
    mid = np.concatenate(mids)
    all_core = pd.concat(sets, ignore_index=True)
    # Piecewise ladder fractions are causal native observations. A conservative
    # one-tick physical resolution floor is explicit, never a numerical epsilon.
    up = all_core["TickRegime.0.tick_regime_up_fraction.0"].to_numpy(dtype=float)
    down = all_core["TickRegime.0.tick_regime_down_fraction.0"].to_numpy(dtype=float)
    tick_bps = np.maximum(up, down) * 10000.0
    tick_bps[~np.isfinite(tick_bps) | (tick_bps <= 0)] = np.nan
    graph = mechanisms.fit_graph(
        rows.loc[rows.day.isin(profile["splits"]["train"])].reset_index(drop=True),
        mid[rows.day.isin(profile["splits"]["train"])],
        topology_train_days=profile["splits"]["train"],
        peer_count=profile["hypotheses"]["peer_count"],
    )
    write_yaml(root / "shared" / "graph.yaml", graph)
    values = mechanisms.transform(rows, mid, tick_bps, graph, prior_config=profile["hypotheses"]["session_prior"])
    values["core"] = all_core
    offsets = np.r_[0, np.cumsum(counts)]
    for role, start, stop in zip(("train", "tune", "forward"), offsets[:-1], offsets[1:], strict=True):
        for family, frame in values.items():
            path = root / "shared" / f"{role}-{family}.parquet"
            frame.iloc[start:stop].reset_index(drop=True).to_parquet(path, index=False)
    write_yaml(
        root / "shared" / "features.yaml",
        {
            "columns": {name: frame.columns.tolist() for name, frame in values.items()},
            "rows": dict(zip(("train", "tune", "forward"), counts, strict=True)),
            "source": "native_float64_origin_mid_on_original_10s_grid",
            "labels_used_in_features": False,
            "graph_fit_role": "train",
        },
    )


def source_files():
    package = Path(__file__).parent
    return sorted(package.glob("*.py")) + [
        package.parents[1] / "io.py",
        package.parent / "signal" / "metrics.py",
        package.parents[1] / "statistics.py",
        package.parents[1] / "material.py",
        package.parents[1] / "store.py",
        package.parents[1] / "contracts.py",
        package.parent / "signal" / "__main__.py",
        package.parent / "signal" / "study.yaml",
        package.parent.parent / "Experiments" / "information_state_20260930" / "PLAN.md",
        package.parent.parent / "Experiments" / "information_state_20260930" / "METHOD_VALIDATION.md",
    ]


def freeze(profile):
    root = output(profile)
    parent = Path(profile["paths"]["parent_study"])
    preflight_receipt = read_yaml(root / "preflight.yaml")
    if not preflight_receipt.get("passed") or preflight_receipt.get("catboost_fits") != 0 or not preflight_receipt.get("all_arms_identical_keys_roles_labels"):
        raise ContractError("The scientific method requires a successful no-fit comparison preflight")
    inputs = {str(p): p for p in (root / "shared").iterdir() if p.is_file()}
    inputs[str(root / "preflight.yaml")] = root / "preflight.yaml"
    inputs[str(parent / "protocol.yaml")] = parent / "protocol.yaml"
    for role in ("train", "tune", "forward"):
        for name in ("projection.json", "x.npy", "rows.parquet", "nominal.parquet", "nominal-projection.yaml"):
            path = parent / role / name
            inputs[str(path)] = path
    versions = {m.__name__: m.__version__ for m in (catboost, np, pd, pyarrow, scipy)}
    path = root / "environment.yaml"
    write_yaml(path, versions)
    inputs[str(path)] = path
    return contract.freeze(profile, root / "frozen-contract.yaml", source_files(), inputs)


def load_role(profile, role, arm):
    root = output(profile)
    native_role = "train" if role == "train" else "forward" if role == "development" else "tune"
    parent = Path(profile["paths"]["parent_study"]) / native_role
    matrix, rows, names, _ = data.load_projection(parent)
    numeric, nominal, _ = data.baseline_scope(profile)
    # Labels of the declared held-symbol group never enter fitting or nomination.
    mask = rows.day.isin(profile["splits"][role]).to_numpy() & data.sampled_mask(rows, profile["sampling"]["interval_seconds"])
    for horizon in profile["horizons"]:
        _, complete = data.native_classes(rows, horizon)
        mask &= complete
    if role in ("train", "es", "calibration"):
        mask &= ~rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).to_numpy()
    indices = np.flatnonzero(mask)
    recipe = profile["arms"][arm]
    if recipe["base"] == "current_nominal":
        positions = {n: i for i, n in enumerate(names)}
        x = np.asarray(matrix[np.ix_(indices, [positions[n] for n in numeric])], dtype=np.float32, order="C")
        feature_names = numeric
    elif recipe["base"] == "core":
        frame = pd.read_parquet(root / "shared" / f"{native_role}-core.parquet").iloc[indices]
        x = np.asarray(frame, dtype=np.float32, order="C")
        feature_names = frame.columns.tolist()
    else:
        raise ContractError("Unknown frozen representation base")
    extras = []
    for family in recipe["families"]:
        frame = pd.read_parquet(root / "shared" / f"{native_role}-{family}.parquet").iloc[indices]
        extras.append(np.asarray(frame, dtype=np.float32))
        feature_names += frame.columns.tolist()
    if recipe.get("permuted_control"):
        joined = np.column_stack(extras)
        finite = np.isfinite(joined).all(axis=1)
        selected_rows = rows.iloc[indices].reset_index(drop=True)
        for (day, symbol), group in selected_rows.groupby(["day", "symbol"], sort=True).indices.items():
            usable = group[finite[group]]
            seed = int.from_bytes(hashlib.sha256(f"information-state-v1:{day}:{symbol}".encode()).digest()[:8], "little")
            joined[usable] = joined[np.random.default_rng(seed).permutation(usable)]
        extras = [joined]
    if extras:
        x = np.asarray(np.column_stack([x, *extras]), dtype=np.float32, order="C")
    nominal_frame = pd.read_parquet(parent / "nominal.parquet")
    keys = ["day", "symbol", "SampleTime", "SampleBookTime", "SampleBookSeq"]
    if not nominal_frame[keys].equals(rows[keys]):
        raise ContractError("Native nominal and numeric complete row anchors disagree")
    cat = nominal_frame.iloc[indices][nominal].astype(object).fillna("NaN").to_numpy(dtype=object)
    return rows.iloc[indices].reset_index(drop=True), x, cat, feature_names, nominal


def signature(profile, rows, horizon, role):
    keys = list(rows[["day", "symbol", "SampleTime", "SampleBookTime", "SampleBookSeq"]].itertuples(index=False, name=None))
    labels = {h: rows[f"mid_return_ticks[{h}s]"].tolist() for h in profile["horizons"]}
    return contract.comparison_signature(profile, keys, labels, [role] * len(rows))


def predict(model, x, cat, numeric_names, nominal_names):
    pieces = []
    for start in range(0, len(x), 16384):
        features = FeaturesData(
            num_feature_data=x[start : start + 16384], cat_feature_data=cat[start : start + 16384], num_feature_names=numeric_names, cat_feature_names=nominal_names
        )
        pieces.append(model.predict_proba(features, thread_count=8)[:, [1, 2]])
    return np.concatenate(pieces)


def fit_one(profile, arm, horizon):
    root = output(profile)
    stem = f"{arm}-{horizon}"
    fit_path = root / "fits" / f"{stem}.yaml"
    model_path = root / "fits" / f"{stem}.cbm"
    frozen = read_yaml(root / "frozen-contract.yaml")
    if fit_path.exists():
        receipt = read_yaml(fit_path)
        if receipt["contract_digest"] != digest(frozen) or receipt["model_sha256"] != file_hash(model_path):
            raise ContractError("Resumed model belongs to another frozen comparison")
        model = CatBoostClassifier().load_model(str(model_path))
        expected_features = receipt["numeric_features"] + receipt["nominal_features"]
        if (
            list(model.classes_) != [0, 1, 2]
            or model.feature_names_ != expected_features
            or model.get_cat_feature_indices() != list(range(len(receipt["numeric_features"]), len(expected_features)))
        ):
            raise ContractError("Resumed model changed its endpoint classes, feature order or nominal schema")
        return model
    tr, tx, tc, names, cats = load_role(profile, "train", arm)
    es, ex, ec, enames, ecats = load_role(profile, "es", arm)
    if names != enames or cats != ecats:
        raise ContractError("Feature order changed across roles")
    train_signature = signature(profile, tr, horizon, "train")
    es_signature = signature(profile, es, horizon, "es")
    target, known = data.native_classes(tr, horizon)
    early, valid = data.native_classes(es, horizon)
    if set(target[known]) != {0, 1, 2} or set(early[valid]) != {0, 1, 2}:
        raise ContractError("Three-class endpoint support is insufficient")
    groups = tr.loc[known, ["day", "symbol"]]
    counts = groups.groupby(["day", "symbol"])["day"].transform("size").to_numpy()
    age = (pd.to_datetime(groups.day.max()) - pd.to_datetime(groups.day)).dt.days.to_numpy()
    weights = np.exp2(-age / profile["training"]["recency_half_life_days"]) / counts
    weights /= weights.mean()
    train_pool = Pool(FeaturesData(num_feature_data=tx[known], cat_feature_data=tc[known], num_feature_names=names, cat_feature_names=cats), label=target[known], weight=weights)
    early_pool = Pool(FeaturesData(num_feature_data=ex[valid], cat_feature_data=ec[valid], num_feature_names=names, cat_feature_names=cats), label=early[valid])
    params = profile["training"]["params"]
    model = CatBoostClassifier(**params)
    started = time.monotonic()
    with contract.gpu_lock(root / "frozen-contract.yaml", check_device=True):
        print(f"FIT {arm} {horizon}s rows={known.sum()} inputs={len(names) + len(cats)}", flush=True)
        model.fit(train_pool, eval_set=early_pool, early_stopping_rounds=profile["training"]["early_stopping_rounds"], use_best_model=True)
    if list(model.classes_) != [0, 1, 2]:
        raise ContractError("Fitted model changed the native endpoint codebook")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.cbm")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    write_yaml(
        fit_path,
        {
            "contract_digest": digest(frozen),
            "model_sha256": file_hash(model_path),
            "arm": arm,
            "horizon": horizon,
            "numeric_features": names,
            "nominal_features": cats,
            "train_comparison": train_signature,
            "es_comparison": es_signature,
            "trees": model.tree_count_,
            "seconds": time.monotonic() - started,
            "weights": "equal_symbol_day_recency",
            "rows": int(known.sum()),
        },
    )
    return model


def score(profile, arm, horizon, role, model):
    rows, x, cat, names, cats = load_role(profile, role, arm)
    scores = predict(model, x, cat, names, cats)
    root = output(profile)
    np.save(root / "fits" / f"{arm}-{horizon}-{role}-scores.npy", scores)
    result = evaluation.evaluate(rows, scores, horizon)
    held = rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).to_numpy()
    groups = {"seen": ~held, "label_held_out": held}
    result["symbol_groups"] = {group: evaluation.evaluate(rows.loc[mask].reset_index(drop=True), scores[mask], horizon) for group, mask in groups.items() if mask.any()}
    days = evaluation.daily(rows, scores, horizon)
    result["comparison_signature"] = signature(profile, rows, horizon, role)
    days.to_csv(root / "fits" / f"{arm}-{horizon}-{role}-daily.csv", index=False)
    write_yaml(root / "fits" / f"{arm}-{horizon}-{role}-metrics.yaml", result)
    print(f"SCORE {arm} {horizon}s {role}: jointAP={result['mean_ap']:.6f} bigAP={result['big_ap']:.6f} dirAUC={result['conditional_direction_auc']:.6f}", flush=True)
    return result, days


def run(profile):
    root = output(profile)
    contract.verify(root / "frozen-contract.yaml")
    expected_environment = read_yaml(root / "environment.yaml")
    if expected_environment != {m.__name__: m.__version__ for m in (catboost, np, pd, pyarrow, scipy)}:
        raise ContractError("Scientific package versions changed from the frozen environment")
    primary = profile["primary_horizon"]
    arms = list(profile["arms"])
    baseline = profile["baseline"]
    calibrated, models = {}, {}
    for arm in arms:
        model = fit_one(profile, arm, primary)
        calibrated[arm], _ = score(profile, arm, primary, "calibration", model)
        models[arm] = model
        gc.collect()
    decisions = {arm: evaluation.decision(profile, calibrated[baseline], calibrated[arm]) for arm in arms if arm != baseline and not profile["arms"][arm].get("permuted_control")}
    admitted = [arm for arm in decisions if decisions[arm]["supported"]]
    nominated = max(admitted, key=lambda arm: calibrated[arm]["mean_ap"]) if admitted else None
    write_yaml(
        root / "nomination.yaml",
        {"baseline": baseline, "challenger": nominated, "qualifying": admitted, "decisions": decisions, "selection_role": "calibration", "held_symbol_labels_used": False},
    )
    # The development labels were already exposed historically. Every arm is
    # reported, but only calibration nomination can receive a positive decision.
    developed, daily = {}, {}
    for arm in arms:
        developed[arm], daily[arm] = score(profile, arm, primary, "development", models[arm])
        contract.assert_same_comparison(developed[baseline]["comparison_signature"], developed[arm]["comparison_signature"])
    final = {}
    for arm in arms:
        if arm == baseline:
            continue
        final[arm] = evaluation.decision(profile, developed[baseline], developed[arm], daily[baseline], daily[arm], stage="development")
        final[arm]["checks"]["nominated_before_development_comparison"] = arm == nominated
        for group in ("seen", "label_held_out"):
            a, b = developed[baseline]["symbol_groups"][group], developed[arm]["symbol_groups"][group]
            final[arm]["checks"][f"{group}_magnitude_generalization"] = b["big_ap"] >= a["big_ap"]
            final[arm]["checks"][f"{group}_direction_generalization"] = b["conditional_direction_auc"] >= max(
                a["conditional_direction_auc"] - 0.01, profile["acceptance"]["minimum_group_direction_auc"]
            )
        final[arm]["supported"] = all(final[arm]["checks"].values())
    write_yaml(root / "development-decisions.yaml", final)
    # Horizon is a research variable, with a bounded two-arm follow-up. Selection
    # is calibration-only; a failed nomination uses the predeclared core control.
    secondary_arm = nominated or "core"
    for horizon in profile["horizons"]:
        if horizon == primary:
            continue
        for arm in (baseline, secondary_arm):
            model = fit_one(profile, arm, horizon)
            score(profile, arm, horizon, "calibration", model)
            score(profile, arm, horizon, "development", model)
            del model
            gc.collect()
    write_yaml(
        root / "completed.yaml",
        {
            "primary_completed": True,
            "pristine_oos": False,
            "sealed_scored": False,
            "nominated": nominated,
            "supported": bool(nominated and final[nominated]["supported"]),
            "contract": digest(read_yaml(root / "frozen-contract.yaml")),
        },
    )


def preflight(profile):
    """Exercise schemas and paired populations without creating any model."""
    reference = {}
    entries = []
    for arm in profile["arms"]:
        for role in ("train", "es", "calibration", "development"):
            rows, x, cat, names, cats = load_role(profile, role, arm)
            comparison = signature(profile, rows, profile["primary_horizon"], role)
            if role in reference:
                contract.assert_same_comparison(reference[role], comparison)
            else:
                reference[role] = comparison
            if not len(rows) or len(set(names + cats)) != len(names) + len(cats) or x.shape != (len(rows), len(names)) or cat.shape != (len(rows), len(cats)):
                raise ContractError("Preflight found invalid feature population or duplicate columns")
            Pool(FeaturesData(num_feature_data=x[:8], cat_feature_data=cat[:8], num_feature_names=names, cat_feature_names=cats), label=np.zeros(min(8, len(rows)), dtype=int))
            entries.append({"arm": arm, "role": role, "rows": len(rows), "numeric_columns": len(names), "nominal_columns": len(cats), "comparison": comparison})
            print(f"PREFLIGHT {arm} {role} rows={len(rows)} columns={len(names) + len(cats)}", flush=True)
            del rows, x, cat
            gc.collect()
    receipt = {"passed": True, "catboost_fits": 0, "all_arms_identical_keys_roles_labels": True, "entries": entries}
    write_yaml(output(profile) / "preflight.yaml", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("command", choices=("prepare", "features", "freeze", "verify", "run"))
    args = parser.parse_args()
    profile = contract.load_profile(args.profile)
    if args.command == "prepare":
        data.prepare(profile)
    elif args.command == "features":
        prepare_features(profile)
    elif args.command == "freeze":
        freeze(profile)
    elif args.command == "verify":
        print(contract.verify(output(profile) / "frozen-contract.yaml"))
    elif args.command == "run":
        run(profile)


if __name__ == "__main__":
    main()
