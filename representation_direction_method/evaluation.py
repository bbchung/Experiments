"""Typed fixed-magnitude score semantics and direction-claim evidence."""

from __future__ import annotations

import hashlib

import numpy as np

from ...io import ContractError
from ..representation_methodology.judgement import claim_decision, factorized_daily, factorized_metrics
from ..representation_research import contract
from ..representation_research.runner import signature


def head(values, length, *, magnitude=False):
    values = np.asarray(values)
    if values.dtype != np.float64 or values.shape != (length,) or not np.isfinite(values).all() or np.any(values > 1) or np.any(values < 0) or (magnitude and np.any(values <= 0)):
        raise ContractError("Typed heads must retain finite native float64 probability vectors; positive magnitude, no epsilon")
    return values


def assert_fixed_b(reference, candidate):
    reference, candidate = head(reference, len(reference), magnitude=True), head(candidate, len(reference), magnitude=True)
    if not np.array_equal(reference.view(np.uint64), candidate.view(np.uint64)):
        raise ContractError("Direction study changed the immutable magnitude head bits")


def original_heads(scores):
    scores = np.asarray(scores)
    if scores.dtype != np.float64 or scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ContractError("Original native up/down score artifact must retain finite float64 class probabilities")
    b = head(scores[:, 0] + scores[:, 1], len(scores), magnitude=True)
    q = head(scores[:, 0] / b, len(scores))
    return b, q


def report(profile, rows, b, q, role):
    if role not in ("calibration", "development"):
        raise ContractError("Method scores use separate frozen calibration/development roles only")
    b, q = head(b, len(rows), magnitude=True), head(q, len(rows))
    comparison = signature(profile, rows, 300, role)
    values = factorized_metrics(rows, b, q, 300)
    held = rows.symbol.isin(profile["universe"]["label_held_out_symbols"]).to_numpy()
    values["symbol_groups"] = {
        group: factorized_metrics(rows.loc[mask].reset_index(drop=True), b[mask], q[mask], 300) for group, mask in (("seen", ~held), ("label_held_out", held)) if mask.any()
    }
    values["comparison_signature"] = comparison
    values["magnitude_bits_sha256"] = hashlib.sha256(b.tobytes()).hexdigest()
    values["role"] = role
    return values, factorized_daily(rows, b, q, 300)


def decision(profile, baseline, candidate, baseline_daily=None, candidate_daily=None, *, stage="calibration"):
    if stage not in ("calibration", "development") or baseline.get("role") != stage or candidate.get("role") != stage:
        raise ContractError("Direction method cannot score one role as another or claim sealed OOS")
    contract.assert_same_comparison(baseline["comparison_signature"], candidate["comparison_signature"])
    if baseline.get("head_contract") != "explicit_p_big_and_q_v1" or candidate.get("head_contract") != baseline["head_contract"]:
        raise ContractError("Direction method requires explicit typed magnitude/conditional heads")
    if not baseline.get("magnitude_bits_sha256") or baseline["magnitude_bits_sha256"] != candidate.get("magnitude_bits_sha256"):
        raise ContractError("Direction comparison magnitude artifacts differ")
    result = claim_decision(profile, baseline, candidate, baseline_daily, candidate_daily, claim="direction", stage=stage)
    magnitude_fields = ("big_ap", "big_base", "equal_symbol_day_big_auc", "absolute_move_spearman")
    if any(np.float64(baseline[field]).view(np.uint64) != np.float64(candidate[field]).view(np.uint64) for field in magnitude_fields) or np.float64(
        baseline["nonoverlap"]["big_ap"]
    ).view(np.uint64) != np.float64(candidate["nonoverlap"]["big_ap"]).view(np.uint64):
        raise ContractError("Fixed magnitude head changed its exact evaluation semantics")
    if stage == "development" and (
        not baseline_daily.day.equals(candidate_daily.day)
        or not np.array_equal(baseline_daily.big_ap.to_numpy().view(np.uint64), candidate_daily.big_ap.to_numpy().view(np.uint64))
    ):
        raise ContractError("Fixed magnitude head changed its daily cohort or metric values")
    result["method"] = "fixed-magnitude-conditional-direction-v1"
    result["qualification"] = "Independent method study only; no FE admission, production adoption or pristine OOS evidence."
    return result


def mean_q(vectors):
    if len(vectors) not in (2, 3):
        raise ContractError("Only preregistered actual two/three-repeat score ensembles are defined")
    vectors = [head(vector, len(vectors[0])) for vector in vectors]
    return np.mean(np.stack(vectors), axis=0, dtype=np.float64)
