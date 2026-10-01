"""Persistently supervise frozen material, exact PMQ and sequential GPU studies."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from AstraResearch.io import ContractError, read_yaml, write_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    args = parser.parse_args()
    root = args.root.resolve()
    here = Path(__file__).resolve().parent
    deadline = time.monotonic() + 22 * 3600

    def state(phase, **values):
        record = {"phase": phase, "utc": dt.datetime.now(dt.UTC).isoformat(), **values}
        write_yaml(root / "research-status.yaml", record)
        print(json.dumps(record), flush=True)

    def await_bindings(tag, names):
        material = root / f"material-{tag}"
        while time.monotonic() < deadline:
            path = material / "baseline-state.yaml"
            if path.exists():
                record = read_yaml(path)
                if set(names).issubset(record["bindings"]):
                    return material
                if record.get("phase") == "complete":
                    raise ContractError(f"{tag} material ended without {names}")
            heartbeat = material / "heartbeat.yaml"
            if heartbeat.exists() and read_yaml(heartbeat).get("terminal") not in (None, "completed"):
                raise ContractError(f"{tag} material failed; retain its native attempt diagnostics")
            state("waiting_material", product=tag, required=names)
            time.sleep(30)
        raise TimeoutError("Research supervision exceeded its 22-hour initial-result budget")

    def run(script, arguments, log_name):
        with (root / log_name).open("ab") as log:
            child = subprocess.Popen([sys.executable, str(here / script), *map(str, arguments)], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            state("running", script=script, pid=child.pid, log=str(root / log_name))
            result = child.wait()
        if result:
            raise ContractError(f"{script} failed with {result}; see {root / log_name}")

    try:
        stock = await_bindings("stock", ["train", "plan"])
        if not (root / "pmq-stock/completed.yaml").exists():
            run("run_pmq.py", ["--material", stock, "--output", root / "pmq-stock"], "pmq-stock-launcher.log")
        await_bindings("stock", ["train", "tune", "forward"])
        if not (root / "study-stock/completed.yaml").exists():
            run("study.py", ["--material", stock, "--output", root / "study-stock", "--pmq-selection", root / "pmq-stock/selected.yaml"], "study-stock-launcher.log")
        for product in ("txf", "exf"):
            material = await_bindings(product, ["train", "tune", "forward"])
            output = root / f"study-{product}"
            if not (output / "completed.yaml").exists():
                run("study.py", ["--material", material, "--output", output], f"study-{product}-launcher.log")
        state("complete", products=["stock", "txf", "exf"])
    except BaseException as error:
        state("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
