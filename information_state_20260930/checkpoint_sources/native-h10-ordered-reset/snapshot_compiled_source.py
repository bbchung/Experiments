import hashlib
import shutil
from pathlib import Path

import yaml

r = Path(__file__).resolve().parent
repo = r.parents[3]
d = yaml.safe_load((r / "producer-source-build.yaml").read_text())
records = []
for node in d["sources"]:
    p = Path(node["path"])
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != node["sha256"]:
        raise SystemExit(f"Compiled source changed: {p}")
    target = r / "compiled-source" / p.relative_to(repo)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != actual:
            raise SystemExit(f"Snapshot changed: {target}")
    else:
        shutil.copy2(p, target)
    records.append({"path": str(target), "sha256": actual})
receipt = {
    "schema": "compiled-source-snapshot-v1",
    "compiled_binary": d["binary"],
    "compiled_binary_sha256": d["binary_sha256"],
    "original_build_receipt": {"path": str(r / "producer-source-build.yaml"), "sha256": hashlib.sha256((r / "producer-source-build.yaml").read_bytes()).hexdigest()},
    "sources": records,
    "no_source_or_binary_modification": True,
}
p = r / "compiled-source-snapshot.yaml"
if p.exists():
    if yaml.safe_load(p.read_text()) != receipt:
        raise SystemExit("Existing receipt differs")
else:
    p.write_text(yaml.safe_dump(receipt, sort_keys=True))
print("SNAPSHOT", len(records), "sources")
