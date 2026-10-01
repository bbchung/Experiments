"""Root-only preparation, preflight, freeze and serial fits in one hash-cache lifetime."""

from pathlib import Path

from astra.representation_research import contract
from astra.representation_research_v3 import runner

profile = contract.load_profile(Path("experiments/information_state_20260930/profile-v3.yaml"))
print("FEATURE_PREPARATION_START", flush=True)
prepared = runner.prepare_features(profile)
assert prepared["passed"] and prepared["catboost_fits"] == 0 and prepared["labels_read"] is False
print("FEATURE_PREPARATION_PASSED", len(prepared["native_files"]), flush=True)
print("PREFLIGHT_START", flush=True)
preflight = runner.preflight(profile)
assert preflight["passed"] and preflight["catboost_fits"] == 0
print("PREFLIGHT_PASSED", len(preflight["entries"]), flush=True)
frozen = runner.freeze(profile)
print("FE_CONTRACT_FROZEN", frozen["identity"], flush=True)
contract.verify(runner.output(profile) / "frozen-contract.yaml")
print("SERIAL_THREE_FITS_START", flush=True)
runner.run(profile)
print("V3_COMPLETED", flush=True)
