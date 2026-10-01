# Conditional-direction methodology proposal

This is an approved research outline, not a validated or adopted production
method. Frozen V1/V2/V3 and method1 judges remain unchanged. A separate prospective
study must freeze its own kernel/profile/source/input closure before any new fit.

MultiClass cross-entropy decomposes into a big-event binary loss plus a
big-event-only direction loss. Neutral dominance therefore does not establish
mathematical incompatibility. The falsifiable hypothesis concerns finite
shared-tree capacity and global early stopping, not an inherent loss defect.
Conditional training also changes CatBoost's learned quantization/target-statistic
population. The bounded formulation experiment cannot isolate capacity or
early stopping as the source of a gain, even with unchanged main parameters.

Hold one preselected immutable full-baseline `P(big)` artifact fixed. Compare the
original MultiClass conditional direction with a new CatBoost Logloss learner
trained only on native big events. The full TRAIN population defines the
original equal-symbol/day and recency weights; subset those weights to big
events without recomputing counts or renormalizing. ES big-event labels only
select iteration. No calibration learner, fusion weight or threshold uses ES.

Keep direct float64 `b` and `q`; compose joint up/down as `b*q` and `b*(1-q)`.
Never reconstruct magnitude from that joint sum: arithmetic ULPs can alter ties.
Preserve b's exact uint64 bits, zero/nonzero probability semantics, original
five-key/four-label cohorts, feature schema and role boundaries.

The prospective study budget is at most three same-seed CatBoost GPU fits,
serialized by root. First calibration meaningful-effect failure rejects early;
all calibration scoring precedes every development read. Reuse immutable MC
controls only with identical source-key/label/feature/training provenance.
Actual per-row q ensembles may be evaluated; scalar metric averages cannot
serve as predictive ensemble evidence. Null-repeat comparisons and stability
deltas are required, but three repeats do not estimate general seed variance
or false-admission rates.

The separately frozen typed direction claim retains joint AP +2%, direction
AUC +0.01 and the old direction floor, while magnitude uses strict zero
noninferiority and absolute power. Equal-symbol/day, paired chronological
bounds, nonoverlap support and symbol-group guards remain. Synthetic score
fixtures establish semantic conformance only; structural source/role/label
provenance enforces leakage boundaries. Metrics cannot detect a perfectly
future-dependent score supplied dishonestly.

Any empirical method benefit remains developmental on exposed dates. The full
legacy magnitude baseline remains an unadmitted raw-scale control; this study
cannot establish a complete normalized replacement or FE/OOS success. Adoption
requires a separate decision and a new formal frozen procedure before any fresh
native C++ FE comparison. DIRECTION_METHOD_PLAN.md records the executable study
contract, without changing existing judges or constructing Python features.
