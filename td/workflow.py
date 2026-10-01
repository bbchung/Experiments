"""Nested target selection. Outer results never enter candidate ranking."""

from __future__ import annotations

import shutil
from pathlib import Path

import catboost
import numpy as np
import pandas as pd

from ...code_guard import source_manifest, verify_sources
from ...io import ContractError, digest, file_hash, write_yaml
from .discovery import staged_search
from .evaluation import date_view, fold_evaluate, null_features, summarize
from .native import population
from .splits import design
from .targets import candidates


def pareto(leaderboard):
    # Predictability is task-specific: dominance uses skill only within a task.
    eligible = [r for r in leaderboard if r["status"] == "measured"]
    result = []
    for row in eligible:
        vector = [row["skill"], row["economic_value"], row["stability"], row["coverage"], -row["target"]["complexity"]]
        dominated = False
        for other in eligible:
            if other["target"]["task"] != row["target"]["task"]:
                continue
            comparison = [other["skill"], other["economic_value"], other["stability"], other["coverage"], -other["target"]["complexity"]]
            if all(a >= b for a, b in zip(comparison, vector, strict=True)) and any(a > b for a, b in zip(comparison, vector, strict=True)):
                dominated = True
                break
        if not dominated:
            result.append(row["target"]["id"])
    return result


def shortlist(leaderboard, cfg):
    available = [r for r in leaderboard if r["qualified"]]
    selected = []
    while available and len(selected) < cfg["budget"]["shortlist"]:
        best = max(r["economic_value"] for r in available)
        near = [r for r in available if r["economic_value"] >= best - cfg["economics"]["equivalence_ticks"]]
        chosen = min(near, key=lambda r: (r["target"]["complexity"], -r["stability"], -r["economic_value"], r["target"]["id"]))
        selected.append(chosen["target"])
        available.remove(chosen)
    return selected


def search(frame, features, targets, splits, cfg, output, *, predictions=True):
    output.mkdir(parents=True, exist_ok=True)
    board = []
    fits = 0
    for target in targets:
        folds = []
        for i, split in enumerate(splits):
            result, rows = fold_evaluate(frame, features, target, split, cfg)
            folds.append(result)
            if rows is not None and predictions:
                folder = output / target["id"]
                folder.mkdir(exist_ok=True)
                rows.to_parquet(folder / f"fold-{i}.parquet", index=False)
        fits += sum(f["fits"] for f in folds)
        write_yaml(output / (target["id"] + ".yaml"), {"target": target, "folds": folds})
        board.append(summarize(target, folds, cfg))
    return board, fits


def search_procedure(frame, features, targets, splits, cfg, output, *, predictions):
    if cfg["discovery"]["enabled"]:
        board, audit = staged_search(frame, features, targets, splits, cfg, output, search, predictions=predictions)
        audit["fits"] = audit["screen_fits"] + audit["confirmation_fits"]
        audit["trees"] = audit["screen_trees"] + audit["confirmation_trees"]
        return board, audit
    board, fits = search(frame, features, targets, splits, cfg, output, predictions=predictions)
    return board, {"fits": fits, "trees": sum(r["trees"] for r in board)}


def select(frame, features, targets, splits, cfg, output):
    board, discovery = search_procedure(frame, features, targets, splits, cfg, output / "observed", predictions=True)
    fits, trees = discovery["fits"], discovery["trees"]
    nulls = []
    needs_null = any(row["qualified"] for row in board)
    status = "completed" if needs_null else "skipped_no_qualified_candidate"
    for repeat in range(cfg["budget"]["null_repeats"] if needs_null else 0):
        seed = cfg["proxy"]["seed"] + 100000 + repeat
        try:
            shuffled, offsets = null_features(frame, features, seed, max(cfg["truth"]["horizons_seconds"]))
        except ContractError as error:
            status = "insufficient_null_support"
            nulls.append({"repeat": repeat, "seed": seed, "status": status, "reason": str(error)})
            break
        # Repeat the entire candidate search and calibration, using one joint
        # feature permutation for every candidate in this null experiment.
        null_board, null_discovery = search_procedure(shuffled, features, targets, splits, cfg, output / "null" / str(repeat), predictions=False)
        del shuffled
        fits += null_discovery["fits"]
        trees += null_discovery["trees"]
        best = max([0.0, *[r["economic_value"] for r in null_board if r["qualified"]]])
        nulls.append(
            {
                "repeat": repeat,
                "seed": seed,
                "status": "completed",
                "offsets": offsets,
                "best_qualified_economic_value": best,
                "leaderboard": null_board,
                "discovery": null_discovery,
            }
        )
        print(f"TD null {repeat + 1}/{cfg['budget']['null_repeats']}", flush=True)
    complete = [n for n in nulls if n["status"] == "completed"]
    enough = status == "completed" and 1 / (len(complete) + 1) <= cfg["gates"]["null_alpha"]
    for row in board:
        p = (1 + sum(n["best_qualified_economic_value"] >= row.get("economic_value", 0.0) for n in complete)) / (len(complete) + 1) if enough else None
        row.update(search_adjusted_null_p=p, null_supported=enough)
        row["qualified"] = row["qualified"] and enough and p <= cfg["gates"]["null_alpha"]
    selected = shortlist(board, cfg)
    assessed = board or discovery.get("screen_leaderboard", [])
    insufficient = (bool(assessed) and not any(row["status"] == "measured" for row in assessed)) or discovery.get("insufficient_observed_support", False)
    result = {
        "status": "qualified" if selected else "inconclusive" if (needs_null and not enough) or insufficient else "no_qualified_target",
        "primary": selected[0] if selected else None,
        "alternatives": selected[1:],
        "pareto": pareto(board),
        "leaderboard": board,
        "null_results": nulls,
        "null_status": status,
        "discovery": discovery,
        "fits": fits,
        "trees": trees,
        "selection_population_digest": digest(frame.sample_id.tolist()),
        "feature_manifest_digest": digest(features),
        "selection_rule": "support/predictability/economic/stability gates; max-search null; economic equivalence then simplicity; no cross-task raw metric ranking",
    }
    write_yaml(output / "selection.yaml", result)
    return result


def fit_budget(cfg, targets, folds):
    heads = sum(2 if t["task"] == "binary_pair" else 1 for t in targets)
    if cfg["discovery"]["enabled"]:
        heads += sum(sorted((2 if t["task"] == "binary_pair" else 1 for t in targets), reverse=True)[: cfg["discovery"]["max_promoted"]])
    planned = len(folds) * (cfg["splits"]["inner_folds"] * heads * (1 + cfg["budget"]["null_repeats"]) + 2 * cfg["budget"]["shortlist"])
    if planned > cfg["budget"]["max_fits"]:
        raise ContractError(f"TD maximum fits {planned} exceeds budget before native export or training")
    return planned


def tree_budget(cfg, targets, folds):
    rounds = len(folds) * cfg["splits"]["inner_folds"] * (1 + cfg["budget"]["null_repeats"])
    per_search = len(targets) * cfg["proxy"]["iterations"]
    if cfg["discovery"]["enabled"]:
        per_search = len(targets) * cfg["discovery"]["iterations"] + min(len(targets), cfg["discovery"]["max_promoted"]) * cfg["proxy"]["iterations"]
    return rounds * per_search + len(folds) * cfg["budget"]["shortlist"] * cfg["proxy"]["iterations"]


def run(frame, cfg, output, provenance=None):
    """In-memory entry used by native CLI and deterministic engineering tests.

    The CLI is the research entrypoint: it supplies native receipts. Direct
    callers without them are explicitly marked engineering-only.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    features = cfg["data"]["features"]
    targets = candidates(cfg)
    folds = design(cfg["data"]["dates"], cfg)
    planned = fit_budget(cfg, targets, folds)
    source_root = Path(__file__).resolve().parents[2]
    sources = source_manifest(source_root)
    for name in sources:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / name, destination)
    shutil.copy2(source_root / "requirements.txt", output / "source/requirements.txt")
    verify_sources(source_root, sources)
    write_yaml(output / "configuration.yaml", cfg)
    write_yaml(
        output / "protocol.yaml",
        {
            "sources": sources,
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "catboost": catboost.__version__},
            "features": features,
            "row_weight": "uniform",
            "population": "all_native_captured_origins_except_predeclared_partitions",
            "label_support": "per_head_finite_training_labels_and_explicit_conditional_predictive_metrics",
            "economic_qualification": "requires_available_policy_and_known_returns_for_every_triggered_intent",
            "tree_budget_per_candidate": cfg["proxy"]["iterations"],
            "discovery": cfg["discovery"],
            "early_stopping": False,
            "execution": "single_thread_cpu",
            "provenance": provenance,
            "engineering_only": provenance is None,
        },
    )
    write_yaml(output / "candidates.yaml", targets)
    write_yaml(output / "splits.yaml", folds)
    frame, audit = population(frame, cfg)
    audit.to_parquet(output / "population.parquet", index=False)
    outputs, fits, trees = [], 0, 0
    for fold in folds:
        verify_sources(source_root, sources)
        work = output / f"outer-{fold['id']}"
        selection_frame = date_view(frame, fold["selection_days"])
        print(f"TD outer {fold['id'] + 1}/{len(folds)}: inner search", flush=True)
        chosen = select(selection_frame, features, targets, fold["inner"], cfg, work / "inner")
        fits += chosen["fits"]
        trees += chosen["trees"]
        frozen = {
            "outer_fold": fold["id"],
            "split_digest": fold["digest"],
            "selection_days": fold["selection_days"],
            "primary": chosen["primary"],
            "alternatives": chosen["alternatives"],
            "status": chosen["status"],
            "selection_hash": file_hash(work / "inner/selection.yaml"),
        }
        # Freeze target choices using inner rows only; native outer truth is
        # already materialized but cannot filter decision origins or select targets.
        write_yaml(work / "frozen-targets.yaml", frozen)
        frozen_hash = file_hash(work / "frozen-targets.yaml")
        validation = []
        for target in [t for t in [chosen["primary"], *chosen["alternatives"]] if t]:
            result, rows = fold_evaluate(frame, features, target, fold["refit"], cfg, save=work / "models" / target["id"])
            fits += result["fits"]
            trees += result.get("trees", 0)
            validation.append({"target": target, "result": result})
            if rows is not None:
                rows.to_parquet(work / (target["id"] + "-outer.parquet"), index=False)
        write_yaml(work / "outer-results.yaml", {"frozen_hash": frozen_hash, "results": validation})
        outputs.append(
            {
                "fold": fold["id"],
                "frozen": frozen,
                "frozen_hash": frozen_hash,
                "outer": validation,
                "pareto": chosen["pareto"],
                "leaderboard": chosen["leaderboard"],
                "discovery": chosen["discovery"],
            }
        )
    primary_results = [o["outer"][0]["result"] for o in outputs if o["outer"]]
    measured = [r for r in primary_results if r["status"] == "measured"]
    economic_complete = [r for r in measured if r["economics"]["evaluation_complete"]]
    # No outer-derived re-ranking. The latest shortlist is exactly the one
    # frozen before its outer window; aggregate reports the selection procedure.
    latest = outputs[-1]["frozen"]
    status = "qualified" if latest["primary"] else latest["status"]
    procedure = {
        "outer_folds": len(folds),
        "nominated_folds": len(primary_results),
        "measured_folds": len(measured),
        "economic_complete_folds": len(economic_complete),
        "positive_economic_folds": sum(r["economics"]["net_ticks_per_opportunity"] > 0 for r in economic_complete),
        "mean_net_ticks_per_opportunity": float(np.mean([r["economics"]["net_ticks_per_opportunity"] for r in measured]))
        if measured and len(economic_complete) == len(measured)
        else None,
    }
    assessed = summarize({"family": "frozen_selection_procedure"}, [o["outer"][0]["result"] if o["outer"] else {"status": "not_nominated", "fits": 0} for o in outputs], cfg)
    procedure["assessment"] = "supported" if assessed["qualified"] else "inconclusive" if assessed["status"] == "inconclusive" else "not_supported"
    procedure["checks"] = assessed.get("checks", {})
    procedure["selection_frequency"] = {t["id"]: sum(o["frozen"]["primary"] is not None and o["frozen"]["primary"]["id"] == t["id"] for o in outputs) for t in targets}
    summary = {
        "status": status,
        "qualification_basis": "inner_only_shortlist_not_production_authority",
        "primary": latest["primary"],
        "alternatives": latest["alternatives"],
        "pareto": outputs[-1]["pareto"],
        "outer_procedure": procedure,
        "folds": outputs,
        "budget": {
            "candidates": len(targets),
            "null_repeats_per_outer": cfg["budget"]["null_repeats"],
            "planned_max_fits": planned,
            "planned_max_trees": tree_budget(cfg, targets, folds),
            "actual_fits": fits,
            "actual_trees": trees,
        },
        "population_digest": digest(audit.to_dict("records")),
        "engineering_only": provenance is None,
        "release_authority": False,
    }
    write_yaml(output / "selected-targets.yaml", {k: summary[k] for k in ["status", "qualification_basis", "primary", "alternatives", "pareto", "release_authority"]})
    if cfg["discovery"]["enabled"]:
        write_yaml(
            output / "discovered-targets.yaml",
            {
                "basis": "screening_hypotheses_not_economic_or_null_qualification",
                "latest_promoted": outputs[-1]["discovery"]["promoted"],
                "folds": [{"fold": o["fold"], "promoted": o["discovery"]["promoted"]} for o in outputs],
                "release_authority": False,
            },
        )
    write_yaml(output / "summary.yaml", summary)
    rows = [
        {
            "outer_fold": o["fold"],
            "target_id": r["target"]["id"],
            "family": r["target"]["family"],
            "task": r["target"]["task"],
            **{k: r.get(k) for k in ["qualified", "skill", "economic_value", "stability", "coverage", "search_adjusted_null_p"]},
        }
        for o in outputs
        for r in o["leaderboard"]
    ]
    pd.DataFrame(rows).to_csv(output / "leaderboard.csv", index=False)
    report = [
        "# Target Discovery",
        "",
        f"Status: {status}. Engineering-only input: {provenance is None}.",
        f"Candidates: {len(targets)}; actual fits: {fits}; maximum budgeted fits: {planned}.",
        f"Staged discovery: {cfg['discovery']}. Screening nominations are hypotheses, not qualified targets.",
        f"Latest pre-outer primary: {latest['primary']}; alternatives: {latest['alternatives']}.",
        f"Outer selection-procedure results: {procedure}.",
        "",
        "The shortlist uses inner evidence only. Outer results evaluate the selection procedure and do not re-rank targets.",
        "Economic values are fixed-holding sticky-price cost proxies, not executable fills or native Fuxi PnL.",
        "No feature selection, model HPO, or production release is performed. Proxy performance is protocol-specific, not a model ceiling.",
        "A null result is valid. A target with insufficient support is reported rather than silently dropping rows.",
        "All captured decision origins are retained regardless of future truth. Training and prediction metrics declare their finite-label masks separately.",
        "Calibration score bins use all origins; unavailable bin policies and unknown triggered returns block economic qualification, and observed-return diagnostics remain conditional.",
        "Null searches are skipped when confirmation has no qualified candidate; skipped is never a null pass. Otherwise each null repeats screening and promotion.",
        "See configuration.yaml, population.parquet, splits.yaml, and outer-*/inner/ for complete decisions, metrics and null searches.",
    ]
    (output / "REPORT.md").write_text("\n\n".join(report) + "\n")
    verify_sources(source_root, sources)
    files = {str(p.relative_to(output)): file_hash(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_yaml(output / "manifest.yaml", {"files": files, "configuration_digest": digest(cfg)})
    return summary
