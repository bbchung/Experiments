"""Keep this experiment's futures workers off the stock runtime's CPU slots."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import psutil

from AstraResearch.io import read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    parser.add_argument("--hours", type=float, default=22)
    args = parser.parse_args()
    if not {16, 17, 18, 19}.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Expected dedicated experiment CPU slots are unavailable")
    root = args.root.resolve()
    assigned = {}
    deadline = time.monotonic() + args.hours * 3600
    with (root / "affinity-adjustments.jsonl").open("a") as log:
        while time.monotonic() < deadline:
            running = False
            for tag, base in (("txf", 16), ("exf", 18)):
                heartbeat = read_yaml(root / f"material-{tag}/heartbeat.yaml")
                if heartbeat.get("terminal") is not None:
                    continue
                running = True
                parent = psutil.Process(heartbeat["pid"])
                descendants = {process.pid for process in parent.children(recursive=True)}
                for record in heartbeat.get("processes", []):
                    if record["name"] != "native-day" or record["pid"] not in descendants:
                        continue
                    try:
                        worker = psutil.Process(record["pid"])
                        if "AstraResearch.native_day_worker" not in worker.cmdline():
                            raise RuntimeError("Heartbeat worker identity changed")
                        key = (tag, worker.pid, worker.create_time())
                        if key not in assigned:
                            original = worker.cpu_affinity()
                            if original not in ([0], [1]):
                                raise RuntimeError(f"Unexpected worker CPU assignment: {original}")
                            assigned[key] = base + original[0]
                        target = [assigned[key]]
                        for process in [worker, *worker.children(recursive=True)]:
                            for thread in process.threads():
                                try:
                                    previous = sorted(os.sched_getaffinity(thread.id))
                                    if previous != target:
                                        os.sched_setaffinity(thread.id, target)
                                        log.write(
                                            json.dumps(
                                                {
                                                    "utc": dt.datetime.now(dt.UTC).isoformat(),
                                                    "product": tag,
                                                    "worker": worker.pid,
                                                    "thread": thread.id,
                                                    "previous": previous,
                                                    "assigned": target,
                                                }
                                            )
                                            + "\n"
                                        )
                                except ProcessLookupError:
                                    pass
                        log.flush()
                    except psutil.NoSuchProcess:
                        continue
            if not running:
                break
            time.sleep(2)


if __name__ == "__main__":
    main()
