"""python3.13 -m AstraResearch.Experiments.signal <command> --profile Experiments/signal/study.yaml

  plan                          print the date plan
  screen                        freeze the universe from the screen window
  regime  --roles dev,sealed    continuous-trading grids (auction-regime exclusion)
  stats                         daily market-data statistics for DailyMdStatsInfo
  material --set S --roles R    native replays of feature set S
  cache   --set S --roles R     feature matrices (base set defines rows)
  fit     --tag T --select SPEC [--train train --es es --eval test]
  compare A B [--eval test]     paired day-block bootstrap of two fits
  representation --stage S    frozen representation research from the profile pointer

SPEC is a comma list of <set>[@<run>:<k>]: all numeric columns of a set, or the
top k of that set by total importance in an earlier run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from ...io import ContractError
from . import dataset, material, model, regime, universe
from .profile import date_plan, load, work

ROLE_STRIDE = {"train": 3}


def role_days(plan, role):
    return plan["dev"] if role == "dev" else plan[role]


def hygienic(profile, base_set, columns, sample=200000):
    data = dataset.open_cache(profile, base_set, "train")
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(data["X"].shape[0], size=min(sample, data["X"].shape[0]), replace=False))
    index = {c: j for j, c in enumerate(data["columns"])}
    keep = []
    for c in columns:
        x = np.asarray(data["X"][rows, index[c]])
        fin = np.isfinite(x)
        if fin.mean() >= 0.5 and np.unique(x[fin]).size >= 3:
            keep.append(c)
    return keep


def parse_select(profile, spec, runs: Path):
    base_set = profile["features"]["base"]
    selection = {}
    for item in spec.split(","):
        name, _, ranked = item.partition("@")
        cache = dataset.open_cache(profile, name, "train")
        if ranked:
            run, _, k = ranked.partition(":")
            result = json.loads((runs / f"{run}.json").read_text())
            score = {}
            for r in result["horizons"].values():
                for column, value in r["importance"]:
                    set_name, _, col = column.partition(":")
                    if set_name == name:
                        score[col] = score.get(col, 0.0) + value
            selection[name] = sorted(score, key=lambda c: -score[c])[: int(k)]
        else:
            selection[name] = hygienic(profile, base_set, cache["columns"]) if name == base_set else None
    return selection


def concat(parts):
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "Y", "day", "symbol", "time")}
    out["columns"], out["horizons"] = parts[0]["columns"], parts[0]["horizons"]
    out["ok"] = {h: np.concatenate([p["ok"][h] for p in parts]) for h in out["horizons"]}
    return out


def representation_profile(path):
    """Resolve only the new recipe pointer; legacy signal fields are not inputs."""
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    pointer = document.get("representation_research") if isinstance(document, dict) else None
    recipe = pointer.get("profile") if isinstance(pointer, dict) else None
    if not isinstance(recipe, str) or not recipe.strip():
        raise ContractError("Signal profile requires representation_research.profile for frozen representation research")
    recipe_path = Path(recipe)
    if not recipe_path.is_absolute():
        recipe_path = path.parent / recipe_path
    from ..representation_research.contract import load_profile

    return load_profile(recipe_path.resolve())


def representation_stage(profile, stage):
    """Load the research runner only after the frozen recipe is validated."""
    from ..representation_research import runner

    if stage == "verify":
        print(runner.contract.verify(runner.output(profile) / "frozen-contract.yaml"))
    else:
        dispatch = {"prepare": runner.data.prepare, "features": runner.prepare_features, "freeze": runner.freeze, "run": runner.run}
        dispatch[stage](profile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "screen", "regime", "stats", "material", "cache", "fit", "compare", "representation"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--profile", type=Path, default=Path(__file__).with_name("study.yaml"))
    ap.add_argument("--stage", choices=["prepare", "features", "freeze", "verify", "run"], default=argparse.SUPPRESS, help="stage of frozen representation research")
    legacy_defaults = {"set": None, "roles": "dev", "tag": None, "select": None, "train": "train", "es": "es", "eval": "test", "workers": 24, "head": None}
    for option in ("set", "roles", "tag", "select", "train", "es", "eval"):
        ap.add_argument(f"--{option}", default=argparse.SUPPRESS)
    ap.add_argument("--workers", type=int, default=argparse.SUPPRESS)
    ap.add_argument("--head", choices=["multiclass", "hybrid"], default=argparse.SUPPRESS, help="override model.head for fit")
    args = ap.parse_args(argv)
    if args.command == "representation":
        overrides = sorted(set(vars(args)) & set(legacy_defaults))
        if overrides or args.args:
            ap.error("representation parameters are fixed by its profile; legacy overrides and positional arguments are not permitted")
        if not hasattr(args, "stage"):
            ap.error("representation requires --stage")
        representation_stage(representation_profile(args.profile), args.stage)
        return
    if hasattr(args, "stage"):
        ap.error("--stage is only valid with representation")
    for option, default in legacy_defaults.items():
        if not hasattr(args, option):
            setattr(args, option, default)
    profile = load(args.profile)
    if args.head:
        profile["model"]["head"] = args.head
    plan = date_plan(profile)
    runs = work(profile) / "runs"

    if args.command == "plan":
        print(yaml.safe_dump({k: [v[0], v[-1], len(v)] if v else [] for k, v in plan.items()}, sort_keys=False))
    elif args.command == "screen":
        print(universe.screen(profile, plan["screen"]))
    elif args.command == "regime":
        days = [d for role in args.roles.split(",") for d in role_days(plan, role)]
        regime.build(profile, days, universe.symbols(profile))
    elif args.command == "stats":
        links = work(profile) / "stats_input" / "tse" / "kgi" / "trade_book"
        links.mkdir(parents=True, exist_ok=True)
        for s in universe.symbols(profile):
            target = links / s
            if not target.exists():
                target.symlink_to(Path(profile["paths"]["market_data"]) / s)
        days = sorted(d for role in ("dev", "sealed") for d in role_days(plan, role))
        first = profile["paths"].get("stats_start", days[0])
        subprocess.run(
            [
                "python3.13",
                profile["paths"]["daily_stats_script"],
                "-i",
                str(links),
                "-o",
                str(work(profile) / "daily_md_stats"),
                "--start",
                str(first),
                "--last",
                days[-1],
                "-j",
                "4",
            ],
            check=True,
            env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
        )
    elif args.command == "material":
        days = sorted(d for role in args.roles.split(",") for d in role_days(plan, role))
        print(material.generate(profile, args.set, days, universe.symbols(profile), workers=args.workers))
    elif args.command == "cache":
        base = profile["features"]["base"]
        for role in args.roles.split(","):
            if args.set == base:
                print(role, dataset.build_base(profile, base, role, role_days(plan, role), ROLE_STRIDE.get(role, 1)))
            else:
                print(role, "missing", dataset.build_aligned(profile, args.set, role, base)[:5])
    elif args.command == "fit":
        selection = parse_select(profile, args.select, runs)
        base = profile["features"]["base"]
        load_ = lambda roles: concat([dataset.load_role(profile, r, base, selection) for r in roles.split(",")])
        train, es, test = load_(args.train), load_(args.es), load_(args.eval)
        model.run(profile, args.tag, train, es, test, runs)
        (runs / f"{args.tag}.selection.json").write_text(json.dumps({"select": args.select, "train": args.train, "es": args.es, "eval": args.eval, "columns": train["columns"]}))
    elif args.command == "compare":
        a, b = args.args
        base = profile["features"]["base"]
        test = dataset.load_role(profile, args.eval, base, {base: []})
        model.compare(profile, test, runs, a, b)


if __name__ == "__main__":
    main()
