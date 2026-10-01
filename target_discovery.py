"""Independent TD command: python3.13 -m AstraResearch.Experiments.target_discovery --help."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..io import read_yaml
from .td.config import protocol, validate_data
from .td.native import materialize
from .td.splits import design
from .td.targets import candidates
from .td.workflow import fit_budget, run


def main():
    parser = argparse.ArgumentParser(description="Bounded native Target Discovery, separate from PMQ/HPO")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(__file__).parents[1] / "runs/td-truth-cache")
    parser.add_argument("--check", action="store_true", help="Validate protocol, candidates and split feasibility without replay or fitting")
    args = parser.parse_args()
    cfg = protocol(read_yaml(args.config))
    validate_data(cfg.get("data"))
    targets = candidates(cfg)
    folds = design(cfg["data"]["dates"], cfg)
    fits = fit_budget(cfg, targets, folds)
    if args.check:
        print(f"TD valid: {len(targets)} candidates, {len(folds)} outer folds, fixed {len(cfg['data']['features'])} features, at most {fits} fits")
        return
    if args.output.exists():
        parser.error("TD output already exists; use a new directory to preserve run evidence")
    frame, provenance = materialize(cfg, args.config.resolve().parent, args.cache.resolve(), exposure_owner=f"target-discovery:{args.output.resolve()}")
    result = run(frame, cfg, args.output.resolve(), provenance)
    print(f"TD {result['status']}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
