"""Reuse exact Release compile/link recipes without mutating its frozen binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build/Release"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--sources", nargs="+", default=["cross_return_context.cpp", "cross_return_context_test.cpp"])
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    replacements, receipt = {}, {"original_build": str(build), "compiled_sources": {}, "targets": {}}
    for target in args.targets:
        commands = subprocess.check_output(["ninja", "-C", str(build), "-t", "commands", target], text=True).splitlines()
        for command in commands:
            if " -c " not in command:
                continue
            values = shlex.split(command)
            source = Path(values[values.index("-c") + 1])
            if source.name not in args.sources:
                continue
            original = values[values.index("-o") + 1]
            if original in replacements:
                continue
            destination = output / (source.name + ".o")
            replacements[original] = str(destination)
            for key, value in (("-o", str(destination)), ("-MF", str(destination.with_suffix(".d"))), ("-MT", str(destination))):
                if key in values:
                    values[values.index(key) + 1] = value
            subprocess.run(values, cwd=build, check=True)
            receipt["compiled_sources"][str(source)] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "argv": values}
        link = shlex.split(commands[-1])
        if link[:2] != [":", "&&"] or link[-2:] != ["&&", ":"]:
            raise ValueError("Unexpected CMake link recipe; do not execute shell text")
        link = [replacements.get(value, value) for value in link[2:-2]]
        destination = output / target
        link[link.index("-o") + 1] = str(destination)
        link = [f"-Wl,--dependency-file={output / (target + '.link.d')}" if value.startswith("-Wl,--dependency-file=") else value for value in link]
        subprocess.run(link, cwd=build, check=True)
        receipt["targets"][target] = {"path": str(destination), "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "link_argv": link}
        (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("isolated Release target", target, flush=True)


if __name__ == "__main__":
    main()
