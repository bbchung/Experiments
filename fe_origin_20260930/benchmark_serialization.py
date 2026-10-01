"""Read-only byte comparison of a published canonical YAML manifest and C emitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

from AstraResearch.io import file_hash, read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Keep earlier serialization evidence")
    path = args.artifact.resolve()
    original_hash = file_hash(path)
    if path.name != "artifact.yaml" or original_hash != path.parent.name:
        raise RuntimeError("Expected a canonical manifest published under its byte identity")
    started = time.monotonic()
    value = read_yaml(path)
    parsed = time.monotonic() - started
    print("parsed canonical manifest", path.stat().st_size, "bytes", parsed, "seconds", flush=True)
    started = time.monotonic()
    alternate = yaml.dump(value, Dumper=yaml.CSafeDumper, sort_keys=True, allow_unicode=True, width=160).encode("utf-8")
    emitted = time.monotonic() - started
    matches = alternate == path.read_bytes()
    result = {
        "source": str(path),
        "original_sha256": original_hash,
        "alternative_sha256": hashlib.sha256(alternate).hexdigest(),
        "bytes": len(alternate),
        "byte_equal": matches,
        "parse_seconds": parsed,
        "c_emitter_seconds": emitted,
        "pyyaml_version": yaml.__version__,
        "diagnostic_only": True,
        "active_code_or_native_artifacts_modified": False,
        "scope": "one_existing_manifest_not_universal_canonicalization_proof",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
