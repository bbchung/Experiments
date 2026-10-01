"""Invoke the existing exact train-only PMQ over canonical material references."""

from __future__ import annotations

import argparse
from pathlib import Path

from AstraResearch.io import read_yaml, write_yaml
from AstraResearch.kernels.pmq import METHOD, PMQProtocolKernel, PMQSelectionKernel, PMQStudyKernel, QualityOnlySelectionKernel
from AstraResearch.runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = read_yaml(args.material / "profile.yaml")
    profile["resources"]["cpu_threads"] = min(profile["resources"]["cpu_threads"], 8)
    profile["resources"]["engine_workers"] = min(profile["resources"]["engine_workers"], 8)
    profile["resources"]["build_jobs"] = min(profile["resources"]["build_jobs"], 8)
    bindings = read_yaml(args.material / "baseline-state.yaml")["bindings"]
    runtime = Runtime(profile, args.output, [PMQProtocolKernel(), PMQStudyKernel(), PMQSelectionKernel(), QualityOnlySelectionKernel()])
    inputs = {name: bindings[name] for name in ("train", "plan")}
    from AstraResearch.contracts import ArtifactRef

    inputs = {name: ArtifactRef(**ref) for name, ref in inputs.items()}
    protocol = runtime.execute("pmq_protocol", inputs)
    study = runtime.execute("pmq_study", {**inputs, "protocol": protocol})
    selection = runtime.execute("pmq_selection", {**inputs, "study": study})
    quality = runtime.execute("quality_only_selection", {**inputs, "study": study})
    write_yaml(args.output / "selected.yaml", runtime.store.resolve(selection).metadata)
    write_yaml(args.output / "quality-only.yaml", runtime.store.resolve(quality).metadata)
    write_yaml(
        args.output / "completed.yaml",
        {
            "protocol": protocol.document(),
            "study": study.document(),
            "selection": selection.document(),
            "quality": quality.document(),
            "operation": f"existing_{METHOD}_no_threshold_relaxation",
        },
    )


if __name__ == "__main__":
    main()
