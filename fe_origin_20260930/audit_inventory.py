"""Reconcile every live declaration, canonical variant and observed native column."""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatchcase
from pathlib import Path

import pyarrow.parquet as pq

from AstraResearch.expansion import compile_expansion
from AstraResearch.io import file_hash, read_yaml, write_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/fe_origin_20260930"))
    args = parser.parse_args()
    root = args.root.resolve()
    guide_path = root / "validation-2/feature-guide.yaml"
    guide = read_yaml(guide_path)
    declarations = {module["type"]: module for module in guide["module_types"]}
    variants, sources, exclusions = compile_expansion(Path.cwd(), declarations)
    pool = {(module["type"], family) for module in variants for family in module["exports"]}
    sample = root / "validation-2/full/data/20260302/2330/values.parquet"
    schema = pq.read_schema(sample)
    semantics = read_yaml(sample.parent / "feature_semantic_manifest.yaml")
    active = {entry["descriptor"]: entry for entry in semantics}
    repo = Path.cwd().parent
    records = []
    for module in guide["module_types"]:
        source = repo / module["source"]
        for family in module["feature_families"]:
            names = [name for name in active if name.split(".")[0] == module["type"] and fnmatchcase(name.split(".")[2], family["name"])]
            records.append(
                {
                    "type": module["type"],
                    "family": family["name"],
                    "source": module["source"],
                    "source_sha256": file_hash(source),
                    "abstraction": family["abstraction_level"],
                    "symmetry": family["mathematical_symmetry"],
                    "history_scope": family["history_scope"],
                    "attributes": ";".join(family["attributes"]),
                    "parameter_status": module["parameter_space"]["status"],
                    "template_status": module["config_template"]["status"],
                    "canonical_pool": (module["type"], family["name"]) in pool,
                    "observed_columns": len(names),
                    "observed_types": ";".join(sorted({str(schema.field(name).type) for name in names})),
                }
            )
    with (root / "feature-inventory-current.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "native_guide_sha256": file_hash(guide_path),
        "module_types": len(declarations),
        "registered_families": len(records),
        "canonical_variants": len(variants),
        "canonical_exported_type_family_pairs": len(pool),
        "canonical_pool_types": len({module["type"] for module in variants}),
        "observed_semantic_columns": len(active),
        "complete_parameter_types": sum(module["parameter_space"]["status"] == "complete" for module in guide["module_types"]),
        "undeclared_parameter_types": sum(module["parameter_space"]["status"] != "complete" for module in guide["module_types"]),
        "explicit_template_exclusions": exclusions,
        "expansion_policy_sources": sources,
        "scientific_limit": "Registration, construction and schema coverage are not formula correctness or predictive validation",
    }
    write_yaml(root / "inventory-summary.yaml", summary)
    print({key: value for key, value in summary.items() if not isinstance(value, list)}, flush=True)


if __name__ == "__main__":
    main()
