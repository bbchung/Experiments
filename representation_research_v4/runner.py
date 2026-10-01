"""Four-arm native H15 study under the unchanged frozen primary300 judge."""

from __future__ import annotations

import argparse
import gc
import re
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from catboost import CatBoostClassifier, FeaturesData, Pool

from ...io import ContractError, digest, file_hash, read_yaml, write_yaml
from ..representation_research import contract, evaluation
from ..representation_research import data as original_data
from ..representation_research import runner as inherited
from ..representation_research_v2.runner import JUDGE_FIELDS, assert_same_cohort
from . import data

output = inherited.output
signature = inherited.signature
predict = inherited.predict
ARMS = {
    "baseline": {"base": "current_nominal", "families": []},
    "compact": {"base": "compact_native", "families": []},
    "peer_k": {"base": "compact_native", "families": ["peer_k"]},
    "peer_j": {"base": "compact_native", "families": ["peer_j"]},
}
BINARY_SHA256 = "613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b"
PAIR_PRODUCER_ID = "00540736c9defcabf2eb9e577bee50840dd3eedd62e00b3fdd981572118a8b61"
PAIR_SUPPORT_ID = "0430f280dc5f18d71050507e878952c6e46cb11360006d59a5b0d8882d8614a5"
SYMBOLS = ("2330", "2317", "2454", "2308", "2382", "3231", "2603", "2609", "2615", "3481", "2409", "2344", "2337", "2481", "3037", "3711")
MAPPING = {symbol: "2317" if symbol == "2330" else "2330" for symbol in SYMBOLS}
EVIDENCE_ROLES = {
    "pair_support": (
        "h15-native-source-support-validation-v2",
        "h15-native-support-evaluation-method-v2",
        "method_identity",
        ("native_artifact_integrity_passed", "support_gate.passed"),
    ),
    "prefix": (
        "h15-real-receive-prefix-validation-v1",
        "h15-real-receive-prefix-comparison-method-v1",
        "comparison_method_identity",
        ("passed", "genuine_straddle_found", "common_origin_present", "all_common_origin_bits_exact"),
    ),
    "mapping_support": (
        "h15-fixed-mapping-source-coverage-validation-v1",
        "h15-fixed-mapping-evaluation-method-v1",
        "method_identity",
        ("native_artifact_integrity_passed", "coverage_gate.passed"),
    ),
    "full_material": ("h15-full-material-validation-v1", "h15-full-material-validation-method-v1", "method_identity", ("passed", "all_original_cells_mid_clock_state_validated")),
}


def require(condition, message):
    if not bool(condition):
        raise ContractError(message)


def parent_contract(profile):
    parent = contract.verify(Path(profile["inheritance"]["frozen_contract"]))
    require(parent["identity"] == profile["inheritance"]["identity"], "V4 parent scientific identity changed")
    for name in JUDGE_FIELDS:
        require(digest(profile.get(name)) == digest(parent["profile"].get(name)), f"V4 changes frozen judge field {name}; research methodology separately first")
    require(profile["paths"]["parent_study"] == parent["profile"]["paths"]["parent_study"], "V4 historical projection changed")
    require(profile["arms"] == ARMS and list(profile["arms"]) == list(ARMS) and profile["baseline"] == "baseline", "V4 four prospective arms changed")
    require(
        type(profile["compute"]["primary_fits"]) is int
        and profile["compute"]["primary_fits"] == 4
        and type(profile["compute"]["secondary_fits_maximum"]) is int
        and profile["compute"]["secondary_fits_maximum"] == 0
        and profile["compute"]["no_hpo"] is True,
        "V4 fit budget changed",
    )
    require(
        profile["compact"]["numeric_fields"] == list(data.C_NUMERIC) and profile["compact"]["categorical_fields"] == list(data.C_CATEGORICAL), "V4 prospectively audited C changed"
    )
    require(profile["h15"]["numeric_fields"] == list(data.NUMERIC) and profile["h15"]["categorical_fields"] == list(data.CATEGORICAL), "V4 native H15 field semantics changed")
    require(profile["h15"]["diagnostics_predictors"] is False, "Native diagnostic fields cannot enter predictors")
    return parent


def artifact_record(path):
    path = Path(path).resolve()
    require(path.is_file(), f"Missing bound artifact: {path}")
    return {"path": str(path), "sha256": file_hash(path)}


def checked_record(node, absent=()):
    require(isinstance(node, dict) and isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str), "Bound artifact requires an exact path and SHA256")
    path = Path(node["path"])
    require(path.is_absolute() and re.fullmatch("[a-f0-9]{64}", node["sha256"]) is not None, "Bound artifact path/SHA256 is malformed")
    if str(path.resolve()) in absent:
        require(not path.exists() and not path.is_symlink(), "A frozen absent input appeared; no replacement or new support is allowed")
    else:
        require(path.is_file() and file_hash(path) == node["sha256"], f"Bound artifact changed or disappeared: {path}")
    return path


def require_member(record, records):
    require(any(item.get("path") == record["path"] and item.get("sha256") == record["sha256"] for item in records), "Frozen method omitted its explicitly bound artifact")


def canonical_method(path, expected_schema, identity=None):
    path = Path(path)
    require(path.is_file() and path.with_suffix(path.suffix + ".identity").is_file(), "Native method/identity anchor missing")
    method = read_yaml(path)
    anchor = path.with_suffix(path.suffix + ".identity").read_text().strip()
    require(isinstance(method, dict) and method.get("schema") == expected_schema and method.get("status") == "frozen", "Wrong native method role/schema")
    require(re.fullmatch("[a-f0-9]{64}", anchor) is not None and anchor == method.get("identity") and (identity is None or anchor == identity), "Native method identity changed")
    require(digest({k: v for k, v in method.items() if k != "identity"}) == anchor, "Native method canonical identity changed")
    require(all(isinstance(method.get(scope), list) and method[scope] for scope in ("sources", "inputs")), "Native proof source/input closure must be nonempty")
    return method


def canonical_evidence(node, role):
    require(role in EVIDENCE_ROLES, "Unknown native proof role")
    receipt_schema, method_schema, identity_field, mandatory = EVIDENCE_ROLES[role]
    require(
        node.get("schema") == receipt_schema and node.get("receipt_identity_field", identity_field) == identity_field,
        "Native proof schema/identity field differs from its scientific role",
    )
    requested = node.get("required_true")
    require(
        isinstance(requested, list) and requested and len(set(requested)) == len(requested) and set(mandatory).issubset(requested),
        "Native proof cannot omit source-level mandatory successful prerequisites",
    )
    path = Path(node["path"])
    checked_record({"path": str(path), "sha256": node["sha256"]})
    receipt = read_yaml(path)
    require(receipt.get("schema") == receipt_schema, "Native evidence receipt schema changed")
    for name in requested:
        value = receipt
        for part in name.split("."):
            require(isinstance(value, dict) and part in value, "Native evidence prerequisite is missing")
            value = value[part]
        require(value is True, f"Native evidence prerequisite failed: {name}")
    require(receipt.get("labels_read") is False and type(receipt.get("model_fits")) is int and receipt["model_fits"] == 0, "Source prerequisite cannot use labels or models")
    method_path = Path(node["method_path"])
    method = canonical_method(method_path, method_schema, node["method_identity"])
    require(receipt.get(identity_field) == method["identity"], "Native receipt does not belong to its prospective method")
    binding = method.get("plan" if role == "prefix" else "profile")
    require(isinstance(binding, dict), "Native method lacks its bound source profile/plan")
    checked_record(binding)
    require_member(binding, method["inputs"])
    bound = read_yaml(Path(binding["path"]))
    if role != "prefix":
        require(method.get("profile_identity") == digest(bound) and receipt.get("profile_identity") == digest(bound), "Native evidence source profile binding changed")
    else:
        require(bound.get("schema") == "h15-real-receive-prefix-bound-comparison-v1", "Prefix method bound an unrelated plan")
        require(bound.get("selection_identity") == receipt.get("selection_method_identity"), "Prefix selection lineage changed")
    absent = set()
    for source_profile in node.get("source_profiles", []):
        checked_record(source_profile)
        require_member(source_profile, method["inputs"])
        source = read_yaml(Path(source_profile["path"]))
        declared = source.get("required_absent_inputs")
        require(
            isinstance(declared, list) and all(isinstance(value, str) and Path(value).is_absolute() for value in declared),
            "Frozen absence rules require a hash-bound source profile",
        )
        absent.update(str(Path(value).resolve()) for value in declared)
    input_paths = {str(Path(item["path"]).resolve()) for item in method["inputs"]}
    require(absent.issubset(input_paths), "Source profile declares absence outside the frozen input closure")
    for scope in ("sources", "inputs"):
        seen = set()
        for item in method[scope]:
            require(item["path"] not in seen, "Native dependency closure has duplicate paths")
            seen.add(item["path"])
            checked_record(item, absent if scope == "inputs" else ())
    if role == "pair_support":
        require(
            method["identity"] == PAIR_SUPPORT_ID and receipt.get("producer_identity") == bound.get("producer_identity") == PAIR_PRODUCER_ID,
            "Fixed native pair producer/support lineage changed",
        )
    elif role == "prefix":
        require(receipt.get("producer_identity") == PAIR_PRODUCER_ID and receipt.get("binary_sha256") == BINARY_SHA256, "Prefix proof changed its immutable pair producer")
    elif role == "mapping_support":
        require(digest(receipt.get("mapping")) == digest(MAPPING), "Native mapping source coverage changed its fixed sixteen-reference map")
        producer_node = bound.get("producer")
        require(isinstance(producer_node, dict), "Mapping evaluation lacks its canonical producer lineage")
        checked_record(producer_node)
        require_member(producer_node, method["inputs"])
        producer = canonical_method(producer_node["path"], "h15-fixed-mapping-producer-method-v1", bound.get("producer_identity"))
        preparation_node = producer.get("preparation")
        checked_record(preparation_node)
        require_member(preparation_node, producer["inputs"])
        preparation = read_yaml(Path(preparation_node["path"]))
        require(producer.get("preparation_identity") == digest(preparation), "Mapping producer preparation changed")
        require(
            preparation.get("shared_pair_producer_identity") == PAIR_PRODUCER_ID and preparation.get("shared_pair_support_identity") == PAIR_SUPPORT_ID,
            "Mapping lost the fixed pair lineage",
        )
        require(preparation.get("binary", {}).get("sha256") == BINARY_SHA256, "Mapping changed the immutable native binary")
        source_profile = preparation.get("profile")
        checked_record(source_profile)
        require_member(source_profile, producer["inputs"])
        coverage = read_yaml(Path(source_profile["path"]))
        require(
            producer.get("profile_identity") == digest(coverage) and digest(coverage.get("mapping")) == digest(MAPPING) and coverage.get("symbols") == list(SYMBOLS),
            "Mapping producer source profile changed its sixteen target/reference identities",
        )
        require(receipt.get("producer_identity") == producer["identity"], "Coverage receipt changed its mapping producer")
    else:
        require(receipt.get("binary_sha256") == BINARY_SHA256 and digest(receipt.get("mapping")) == digest(MAPPING), "Full native material changed producer or fixed mapping")
        require(bound.get("pair_producer_identity") == PAIR_PRODUCER_ID and bound.get("pair_support_identity") == PAIR_SUPPORT_ID, "Full material lost fixed native pair lineage")
    # This also closes the evidence receipt to the same checked dependencies.
    for scope in ("sources", "inputs"):
        require(isinstance(receipt.get(scope), list), "Native receipt omitted its source/input closure")
        for item in method[scope]:
            require_member(item, receipt[scope])
        require(len({item.get("path") for item in receipt[scope]}) == len(receipt[scope]), "Native receipt dependency closure has duplicate paths")
        for item in receipt[scope]:
            checked_record(item, absent if scope == "inputs" else ())
    return method, receipt


def native_evidence(profile):
    result = [canonical_evidence(profile["h15"][name], name) for name in EVIDENCE_ROLES]
    binary = Path(profile["h15"]["binary"])
    require(profile["h15"]["binary_sha256"] == BINARY_SHA256 and file_hash(binary) == BINARY_SHA256, "H15 immutable producer changed")
    node = artifact_record(binary)
    require(
        all(any(n.get("path") == node["path"] and n.get("sha256") == node["sha256"] for n in method["inputs"]) for method, _ in result),
        "Native proof omitted the immutable binary dependency",
    )
    return result


def study_sources(profile, parent, evidence):
    experiment = Path(profile["_profile_path"]).parent
    records = [*parent["sources"], *[node for _, receipt in evidence for node in receipt["sources"]]]
    paths = {record["path"]: Path(record["path"]) for record in records}
    for path in [
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).parents[2] / "feature_values.py",
        Path(original_data.__file__),
        experiment / "H15_REPRESENTATION_COMPARISON_PLAN.md",
        experiment / "H15_COMPACT_C_SOURCE_AUDIT.md",
        Path(__file__).parents[2] / "tests/test_representation_v4_data.py",
        Path(__file__).parents[2] / "tests/test_representation_v4.py",
    ]:
        paths[str(path.resolve())] = path.resolve()
    return [artifact_record(path) for _, path in sorted(paths.items())]


def preparation(profile):
    root = output(profile)
    path = root / "shared/h15-preparation.yaml"
    receipt = read_yaml(path)
    require(
        receipt.get("schema") == "h15-native-feature-preparation-v1"
        and receipt.get("passed") is True
        and receipt.get("labels_read") is False
        and type(receipt.get("model_fits")) is int
        and receipt["model_fits"] == 0
        and receipt.get("all_original_parent_keys_mid_bits_validated") is True,
        "Native exact join preparation has no complete typed parent key/mid validation",
    )
    caches = receipt.get("caches")
    require(
        isinstance(caches, list) and len(caches) == 3 and [node.get("role") for node in caches] == ["train", "tune", "forward"],
        "Exactly three canonical full-parent native role caches are required",
    )
    for node in caches:
        role = node["role"]
        expected = root / "shared" / f"{role}-h15.parquet"
        require(node.get("path") == str(expected.resolve()), "Native cache was swapped or replaced by another role/path")
        checked_record(node)
        require(type(node.get("rows")) is int and node["rows"] > 0 and node["rows"] == pq.ParquetFile(expected).metadata.num_rows, "Native cache row metadata changed")
        require(
            node.get("numeric") == list(data.CANONICAL[: len(data.NUMERIC)]) and node.get("categorical") == list(data.CANONICAL[len(data.NUMERIC) :]),
            "Native cache canonical field order changed",
        )
        require(
            node.get("parent_keys_sha256") == file_hash(Path(profile["paths"]["parent_study"]) / role / "rows.parquet"), "Native cache parent observation-file identity changed"
        )
        require(node.get("parent_mid_sha256") == file_hash(root / "shared" / f"mid-{role}.npy"), "Native cache parent float64-mid identity changed")
    native_files = receipt.get("native_files")
    require(
        isinstance(native_files, list) and native_files and len({node.get("path") for node in native_files}) == len(native_files),
        "Native material input closure must be complete and unique",
    )
    for node in native_files:
        checked_record(node)
    return receipt


def preflight_binding(profile, parent, evidence):
    prepared = preparation(profile)
    nodes = []
    for role in EVIDENCE_ROLES:
        node = profile["h15"][role]
        nodes.extend(artifact_record(path) for path in (node["path"], node["method_path"], node["method_path"] + ".identity"))
    return {
        "profile_identity": digest(contract.scientific_profile(profile)),
        "preparation": artifact_record(output(profile) / "shared/h15-preparation.yaml"),
        "caches": prepared["caches"],
        "native_evidence": nodes,
        "sources": study_sources(profile, parent, evidence),
        "inherited_inputs": parent["inputs"],
    }


def arm_names(profile, arm):
    if arm == "baseline":
        numeric, nominal, _ = original_data.baseline_scope(profile)
        return list(numeric), list(nominal)
    numeric, nominal = list(data.C_NUMERIC), list(data.C_CATEGORICAL)
    if arm in ("peer_k", "peer_j"):
        numeric += list(data.CANONICAL[: len(data.NUMERIC)])
        offset = len(data.NUMERIC) + (2 if arm == "peer_k" else 0)
        nominal += list(data.CANONICAL[offset : offset + 2])
    return numeric, nominal


def validate_preflight(profile, receipt, binding):
    require(
        receipt.get("schema") == "h15-representation-preflight-v1"
        and all(
            receipt.get(k) is True
            for k in ("passed", "all_arms_identical_keys_roles_labels", "inherited_judge_identical", "inherited_native_cohort_identical", "shared_h15_numeric_bits_identical")
        ),
        "V4 preflight is incomplete or uses untyped successful proof flags",
    )
    require(type(receipt.get("catboost_fits")) is int and receipt["catboost_fits"] == 0, "V4 preflight cannot fit")
    require(digest(receipt.get("binding")) == digest(binding), "V4 source/profile/native-cache identity drifted after preflight")
    entries = receipt.get("entries")
    expected = [(role, arm) for role in contract.ROLES for arm in ARMS]
    require(
        isinstance(entries, list) and [(entry.get("role"), entry.get("arm")) for entry in entries] == expected,
        "V4 preflight must cover exactly four ordered arms by four ordered roles",
    )
    for entry in entries:
        names, cats = arm_names(profile, entry["arm"])
        require(entry.get("numeric_features") == names and entry.get("nominal_features") == cats, "Preflight feature-role descriptor order changed")
        require(
            type(entry.get("rows")) is int
            and entry["rows"] > 0
            and type(entry.get("numeric_columns")) is int
            and entry["numeric_columns"] == len(names)
            and type(entry.get("nominal_columns")) is int
            and entry["nominal_columns"] == len(cats),
            "Preflight shape/row metadata changed",
        )
    for role in contract.ROLES:
        selected = [entry for entry in entries if entry["role"] == role]
        for entry in selected:
            require(entry["rows"] == selected[0]["rows"], "Preflight arm row counts differ")
            contract.assert_same_comparison(selected[0]["comparison"], entry["comparison"])


def preflight(profile):
    root = output(profile)
    require(not (root / "frozen-contract.yaml").exists(), "Frozen V4 preflight cannot be overwritten")
    (root / "preflight.yaml").unlink(missing_ok=True)
    parent = parent_contract(profile)
    evidence = native_evidence(profile)
    binding = preflight_binding(profile, parent, evidence)
    prior = read_yaml(Path(parent["profile"]["paths"]["output"]) / "preflight.yaml")
    references = {entry["role"]: entry["comparison"] for entry in prior["entries"] if entry["arm"] == "baseline"}
    entries = []
    for role in contract.ROLES:
        baseline = data.load_role(profile, role, "baseline")
        expected = signature(profile, baseline[0], 300, role)
        assert_same_cohort(references[role], expected)
        compact = None
        peer_numeric = None
        for arm in ARMS:
            current = baseline if arm == "baseline" else data.load_role(profile, role, arm)
            rows, x, cat, names, cats = current
            comparison = signature(profile, rows, 300, role)
            contract.assert_same_comparison(expected, comparison)
            require(x.shape == (len(rows), len(names)) and cat.shape == (len(rows), len(cats)) and len(set(names + cats)) == len(names) + len(cats), "V4 schema/shape invalid")
            require(names == arm_names(profile, arm)[0] and cats == arm_names(profile, arm)[1], "V4 arm feature-role capacity changed")
            require(x.dtype == np.float32 and cat.dtype == object and len(rows) > 0, "V4 consumer array storage or row support changed")
            if arm == "compact":
                compact = (x.copy(), cat.copy())
            if arm in ("peer_k", "peer_j"):
                require(np.array_equal(x[:, : len(data.C_NUMERIC)].view(np.uint32), compact[0].view(np.uint32)), "C bits differ between arms")
                require(np.array_equal(cat[:, : len(data.C_CATEGORICAL)], compact[1]), "C nominal coding differs between arms")
                if arm == "peer_k":
                    peer_numeric = x[:, len(data.C_NUMERIC) :].copy()
                else:
                    require(np.array_equal(x[:, len(data.C_NUMERIC) :].view(np.uint32), peer_numeric.view(np.uint32)), "K/J native numeric bits differ at identical observations")
            Pool(FeaturesData(num_feature_data=x[:8], cat_feature_data=cat[:8], num_feature_names=names, cat_feature_names=cats), label=np.zeros(min(8, len(rows)), dtype=int))
            entries.append(
                {
                    "role": role,
                    "arm": arm,
                    "comparison": comparison,
                    "rows": len(rows),
                    "numeric_columns": len(names),
                    "nominal_columns": len(cats),
                    "numeric_features": names,
                    "nominal_features": cats,
                }
            )
            if current is not baseline:
                del current, rows, x, cat
        del baseline, compact, peer_numeric
        gc.collect()
    receipt = {
        "schema": "h15-representation-preflight-v1",
        "passed": True,
        "catboost_fits": 0,
        "all_arms_identical_keys_roles_labels": True,
        "inherited_judge_identical": True,
        "inherited_native_cohort_identical": True,
        "shared_h15_numeric_bits_identical": True,
        "binding": binding,
        "entries": entries,
    }
    require(digest(binding) == digest(preflight_binding(profile, parent, native_evidence(profile))), "Source/profile/native artifacts changed during preflight")
    validate_preflight(profile, receipt, binding)
    write_yaml(root / "preflight.yaml", receipt)
    return receipt


def freeze(profile):
    root = output(profile)
    require(not (root / "frozen-contract.yaml").exists(), "Frozen V4 cannot be overwritten")
    parent = parent_contract(profile)
    evidence = native_evidence(profile)
    receipt = read_yaml(root / "preflight.yaml")
    binding = preflight_binding(profile, parent, evidence)
    validate_preflight(profile, receipt, binding)
    prepared = preparation(profile)
    sources = {n["path"]: Path(n["path"]) for n in binding["sources"]}
    inputs = {n["path"]: Path(n["path"]) for n in parent["inputs"]}
    for method, native in evidence:
        for n in native["sources"]:
            sources[n["path"]] = Path(n["path"])
        for n in native["inputs"]:
            # Missing source inputs have already passed hash-bound absence rules.
            # Their profile/method fact remains frozen; run rechecks appearance.
            if Path(n["path"]).is_file():
                inputs[n["path"]] = Path(n["path"])
    for path in [
        Path(profile["inheritance"]["frozen_contract"]),
        Path(profile["inheritance"]["frozen_contract"] + ".identity"),
        root / "preflight.yaml",
        *root.joinpath("shared").iterdir(),
    ]:
        if path.is_file():
            inputs[str(path)] = path
    for name in ("pair_support", "prefix", "mapping_support", "full_material"):
        node = profile["h15"][name]
        for path in (Path(node["path"]), Path(node["method_path"]), Path(node["method_path"] + ".identity")):
            inputs[str(path)] = path
    for n in prepared["native_files"]:
        path = Path(n["path"])
        require(file_hash(path) == n["sha256"], "Prepared native H15 input changed")
        inputs[str(path)] = path
    versions = {m.__name__: m.__version__ for m in (catboost, np, pd, pyarrow, scipy)}
    write_yaml(root / "environment.yaml", versions)
    inputs[str(root / "environment.yaml")] = root / "environment.yaml"
    return contract.freeze(profile, root / "frozen-contract.yaml", list(sources.values()), inputs)


def fit_one(profile, arm):
    root = output(profile)
    fit_path, model_path = (root / "fits" / f"{arm}-300.{ext}" for ext in ("yaml", "cbm"))
    frozen = read_yaml(root / "frozen-contract.yaml")
    if fit_path.exists():
        receipt = read_yaml(fit_path)
        require(receipt["contract_digest"] == digest(frozen) and receipt["model_sha256"] == file_hash(model_path), "Resumed fit changed contract")
        model = CatBoostClassifier().load_model(str(model_path))
        require(
            list(model.classes_) == [0, 1, 2] and model.feature_names_ == receipt["numeric_features"] + receipt["nominal_features"], "Resumed model feature/class schema changed"
        )
        require(model.get_cat_feature_indices() == list(range(len(receipt["numeric_features"]), len(model.feature_names_))), "Resumed nominal feature positions changed")
        return model
    tr, tx, tc, names, cats = data.load_role(profile, "train", arm)
    es, ex, ec, enames, ecats = data.load_role(profile, "es", arm)
    require(names == enames and cats == ecats, "Feature order changed across fitting roles")
    target, known = original_data.native_classes(tr, 300)
    early, valid = original_data.native_classes(es, 300)
    require(set(target[known]) == set(early[valid]) == {0, 1, 2}, "Native endpoint classes insufficient")
    groups = tr.loc[known, ["day", "symbol"]]
    counts = groups.groupby(["day", "symbol"])["day"].transform("size").to_numpy()
    age = (pd.to_datetime(groups.day.max()) - pd.to_datetime(groups.day)).dt.days.to_numpy()
    weights = np.exp2(-age / profile["training"]["recency_half_life_days"]) / counts
    weights /= weights.mean()
    train_pool = Pool(FeaturesData(num_feature_data=tx[known], cat_feature_data=tc[known], num_feature_names=names, cat_feature_names=cats), label=target[known], weight=weights)
    early_pool = Pool(FeaturesData(num_feature_data=ex[valid], cat_feature_data=ec[valid], num_feature_names=names, cat_feature_names=cats), label=early[valid])
    model = CatBoostClassifier(**profile["training"]["params"])
    started = time.monotonic()
    with contract.gpu_lock(root / "frozen-contract.yaml", check_device=True):
        print(f"FIT {arm}300s rows={int(known.sum())} inputs={len(names) + len(cats)}", flush=True)
        model.fit(train_pool, eval_set=early_pool, early_stopping_rounds=profile["training"]["early_stopping_rounds"], use_best_model=True)
    require(list(model.classes_) == [0, 1, 2], "Fitted model endpoint codebook changed")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".tmp.cbm")
    model.save_model(str(temporary))
    temporary.replace(model_path)
    write_yaml(
        fit_path,
        {
            "contract_digest": digest(frozen),
            "model_sha256": file_hash(model_path),
            "arm": arm,
            "horizon": 300,
            "numeric_features": names,
            "nominal_features": cats,
            "train_comparison": signature(profile, tr, 300, "train"),
            "es_comparison": signature(profile, es, 300, "es"),
            "trees": model.tree_count_,
            "seconds": time.monotonic() - started,
            "weights": "equal_symbol_day_recency",
            "rows": int(known.sum()),
        },
    )
    return model


def score(profile, arm, role, model):
    rows, x, cat, names, cats = data.load_role(profile, role, arm)
    scores = predict(model, x, cat, names, cats)
    root = output(profile)
    np.save(root / "fits" / f"{arm}-300-{role}-scores.npy", scores)
    result = evaluation.evaluate(rows, scores, 300)
    held = rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).to_numpy()
    result["symbol_groups"] = {
        group: evaluation.evaluate(rows.loc[mask].reset_index(drop=True), scores[mask], 300) for group, mask in {"seen": ~held, "label_held_out": held}.items() if mask.any()
    }
    result["comparison_signature"] = signature(profile, rows, 300, role)
    days = evaluation.daily(rows, scores, 300)
    days.to_csv(root / "fits" / f"{arm}-300-{role}-daily.csv", index=False)
    write_yaml(root / "fits" / f"{arm}-300-{role}-metrics.yaml", result)
    print(f"SCORE {arm} {role}: jointAP={result['mean_ap']:.6f} bigAP={result['big_ap']:.6f} dirAUC={result['conditional_direction_auc']:.6f}", flush=True)
    return result, days


def run(profile):
    root = output(profile)
    frozen = contract.verify(root / "frozen-contract.yaml")
    require(contract.scientific_profile(profile) == frozen["profile"], "V4 in-memory profile differs from freeze")
    parent_contract(profile)
    native_evidence(profile)
    require({m.__name__: m.__version__ for m in (catboost, np, pd, pyarrow, scipy)} == read_yaml(root / "environment.yaml"), "Scientific environment changed")
    calibrated, models = {}, {}
    for arm in ARMS:
        models[arm] = fit_one(profile, arm)
        calibrated[arm], _ = score(profile, arm, "calibration", models[arm])
        contract.assert_same_comparison(calibrated["baseline"]["comparison_signature"], calibrated[arm]["comparison_signature"])
        gc.collect()
    decisions = {arm: evaluation.decision(profile, calibrated["baseline"], calibrated[arm]) for arm in ARMS if arm != "baseline"}
    qualifying = [arm for arm, verdict in decisions.items() if verdict["supported"]]
    nominated = max(qualifying, key=lambda arm: calibrated[arm]["mean_ap"]) if qualifying else None
    write_yaml(
        root / "nomination.yaml",
        {"baseline": "baseline", "challenger": nominated, "qualifying": qualifying, "decisions": decisions, "selection_role": "calibration", "held_symbol_labels_used": False},
    )
    developed, daily = {}, {}
    for arm in ARMS:
        developed[arm], daily[arm] = score(profile, arm, "development", models[arm])
        contract.assert_same_comparison(developed["baseline"]["comparison_signature"], developed[arm]["comparison_signature"])
    final = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        verdict = evaluation.decision(profile, developed["baseline"], developed[arm], daily["baseline"], daily[arm], stage="development")
        verdict["checks"]["nominated_before_development_comparison"] = arm == nominated
        for group in ("seen", "label_held_out"):
            require(group in developed["baseline"]["symbol_groups"] and group in developed[arm]["symbol_groups"], "V4 development symbol group absent")
            a, b = (developed[name]["symbol_groups"][group] for name in ("baseline", arm))
            verdict["checks"][f"{group}_magnitude_generalization"] = b["big_ap"] >= a["big_ap"]
            verdict["checks"][f"{group}_direction_generalization"] = b["conditional_direction_auc"] >= max(
                a["conditional_direction_auc"] - 0.01, profile["acceptance"]["minimum_group_direction_auc"]
            )
        verdict["supported"] = all(verdict["checks"].values())
        final[arm] = verdict
    write_yaml(root / "development-decisions.yaml", final)
    contrasts = {}
    for baseline, challenger in (("compact", "peer_k"), ("peer_k", "peer_j")):
        contrasts[f"{challenger}-versus-{baseline}"] = {
            "calibration": evaluation.decision(profile, calibrated[baseline], calibrated[challenger]),
            "development": evaluation.decision(profile, developed[baseline], developed[challenger], daily[baseline], daily[challenger], stage="development"),
        }
    write_yaml(root / "predeclared-contrasts.yaml", contrasts)
    contract.verify(root / "frozen-contract.yaml")
    native_evidence(profile)
    write_yaml(
        root / "completed.yaml",
        {
            "primary_completed": True,
            "primary_fits": 4,
            "secondary_fits": 0,
            "pristine_oos": False,
            "sealed_scored": False,
            "nominated": nominated,
            "supported": bool(nominated and final[nominated]["supported"]),
            "contract": digest(frozen),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("stage", choices=("preflight", "freeze", "verify", "run"))
    args = parser.parse_args()
    profile = contract.load_profile(args.profile)
    if args.stage == "verify":
        print(contract.verify(output(profile) / "frozen-contract.yaml")["identity"])
    else:
        print({"preflight": preflight, "freeze": freeze, "run": run}[args.stage](profile))


if __name__ == "__main__":
    main()
