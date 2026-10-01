"""python3.13 -m AstraResearch.Experiments.fe <command> --profile <study>/profile.yaml

roles                          print the chronological role plan
screen                         freeze the universe from the pre-study window
stats                          DailyMdStats CSVs (past-day contract stats) for the universe
regime                         continuous-trading grids for every material day
material SET [--scope S]       native replays of a feature set (scope: all|dev|holdout)
cohort                         shared evaluation rows from the anchor set
cache SET                      aligned float32 matrices of SET at the cohort rows
fit ...                        see AstraResearch.Experiments.fe.train
"""

from __future__ import annotations

import argparse
import subprocess

from . import profile as P


def material_jobs(profile, scope):
    roles = P.roles(profile)
    dev_days = sorted({d for r in ("dev_train", "dev_es", "dev_val", "final_train", "final_es", "oos") for d in roles[r]})
    jobs = []
    if scope in ("all", "dev"):
        jobs += [(d, s) for d in dev_days for s in P.symbols(profile, "development")]
    if scope in ("all", "holdout"):
        jobs += [(d, s) for d in roles["oos"] for s in P.symbols(profile, "holdout")]
    return sorted(jobs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--scope", default="all")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    profile = P.load(args.profile)
    if args.command == "roles":
        for name, days in P.roles(profile).items():
            print(f"{name:12s} {days[0]} {days[-1]} {len(days)}")
    elif args.command == "screen":
        from . import screen

        r = screen.screen(profile)
        print(r["candidates"], r["eligible"], len(r["development"]), len(r["holdout"]))
    elif args.command == "stats":
        universe = P.symbols(profile)
        links = P.work(profile) / "stats_input" / "tse" / "kgi" / "trade_book"
        links.mkdir(parents=True, exist_ok=True)
        for s in universe:
            target = links / s
            if not target.exists():
                target.symlink_to(f"{profile['paths']['market_data']}/{s}")
        days = P.observed_days(profile, str(profile["data_floor"]), str(profile["dates"]["last"]))
        subprocess.run(
            [
                "python3.13",
                profile["paths"]["daily_stats_script"],
                "-i",
                str(links),
                "-o",
                str(P.work(profile) / "daily_md_stats"),
                "--start",
                days[0],
                "--last",
                days[-1],
                "-j",
                "8",
            ],
            check=True,
            env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"},
        )
    elif args.command == "regime":
        from ..signal import regime

        days = sorted({d for d, _ in material_jobs(profile, "all")})
        regime.build(profile, days, P.symbols(profile))
    elif args.command == "material":
        from . import material

        (set_name,) = args.args
        print(material.generate(profile, set_name, material_jobs(profile, args.scope), P.symbols(profile), workers=args.workers))
    elif args.command == "cohort":
        from . import dataset

        print(dataset.build_cohort(profile, material_jobs(profile, "all")))
    elif args.command == "cache":
        from . import dataset

        (set_name,) = args.args
        print(dataset.build_cache(profile, set_name))
    elif args.command in ("fit", "score"):
        from . import compare, freeze

        (round_name,) = args.args
        spec = profile["rounds"][round_name]
        if not spec.get("method_study"):
            spec["_judge_identity"] = freeze.verify(profile)
        if args.command == "fit":
            compare.fit_round(profile, round_name, horizons=[int(h) for h in args.horizons.split(",")] if args.horizons else None, arms=args.arms.split(",") if args.arms else None)
        else:
            compare.score_round(profile, round_name)
    elif args.command == "census":
        from . import census

        (set_name,) = args.args
        print(census.run(profile, set_name))
    elif args.command == "method":
        from . import method_study

        method_study.run(profile)
    elif args.command == "models":
        # Diagnostic model comparison: arms with `model: lightgbm` use the LightGBM
        # kernel, others the CatBoost kernel; scoring is the unchanged judge.
        from ...io import write_yaml
        from . import compare, econ, train, train_lgbm

        (round_name,) = args.args
        spec = profile["rounds"][round_name]
        for name, arm in spec["arms"].items():
            fit = train_lgbm.fit_arm if arm.get("model") == "lightgbm" else train.fit_arm
            fit(profile, round_name, name, arm, spec["stage"], spec["horizons"], seeds=arm.get("seeds"))
        compare.score_round(profile, round_name)
        write_yaml(P.work(profile) / "fits" / round_name / "economics.yaml", econ.tail(profile, round_name))
    elif args.command == "summary":
        from . import report

        (round_name,) = args.args
        print(report.summary(profile, round_name))
    elif args.command == "freeze":
        from . import freeze

        print(freeze.freeze(profile, args.note))
    elif args.command == "verify":
        from . import freeze

        print(freeze.verify(profile))
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
