"""Day-constancy of native columns on dev_train (never dev_val/oos).

constancy = 1 - (mean within symbol-day variance) / (total variance) over
finite values. A column near one is (almost) constant inside a symbol-day: it
can identify the day, so a model may fit day-level label idiosyncrasies with
it. Symmetry comes from the native registry (S0 magnitude, S1 antisymmetric,
S2 other). This is a representation diagnostic, not a selection by outcomes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ...io import read_yaml
from . import dataset
from . import profile as P
from .train import role_rows


def symmetry_map(guide_path) -> dict[tuple[str, str], str]:
    out = {}
    for module in read_yaml(Path(guide_path))["module_types"]:
        for family in module.get("feature_families", []):
            out[module["type"], family["name"].rstrip("*")] = family.get("mathematical_symmetry", "")
    return out


def family_symmetry(sym, column):
    module, _, rest = column.partition(".")
    family = rest.split(".", 1)[1].rsplit(".", 1)[0] if rest.count(".") >= 2 else rest
    if (module, family) in sym:
        return sym[module, family]
    # Pattern families (e.g. ref_move_ticks*) are registered by prefix.
    for (m, f), s in sym.items():
        if m == module and family.startswith(f):
            return s
    return ""


def run(profile, set_name, guide_path, sample=400000, seed=0):
    cohort = dataset.load_cohort(profile)
    cache = dataset.open_cache(profile, set_name)
    rows = role_rows(profile, cohort, "dev_train", "development")
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(rows, size=min(sample, len(rows)), replace=False))
    group = np.unique(np.char.add(cohort["day"][rows], cohort["symbol"][rows]), return_inverse=True)[1]
    X = np.asarray(cache["X"][rows], dtype=np.float64)
    sym = symmetry_map(guide_path)
    out = []
    for j, name in enumerate(cache["numeric"]):
        x = X[:, j]
        fin = np.isfinite(x)
        rec = {"column": name, "symmetry": family_symmetry(sym, name), "constancy": np.nan}
        if fin.sum() > 1000 and np.var(x[fin]) > 0:
            g = group[fin]
            v = x[fin]
            counts = np.bincount(g)
            means = np.bincount(g, weights=v) / np.maximum(counts, 1)
            within = ((v - means[g]) ** 2).sum() / len(v)
            rec["constancy"] = float(1 - within / np.var(v))
        out.append(rec)
    target = P.work(profile) / "census" / f"{set_name}-constancy.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["column", "symmetry", "constancy"])
        writer.writeheader()
        writer.writerows(out)
    return target
