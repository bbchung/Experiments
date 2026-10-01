# Evaluation-method validation before FE comparison

Validation completed on 2026-09-30 before any scientific fit in this recipe.
This changes the judge as a separate, declared methodology step, rather than
changing gates in response to a candidate's calibration or forward performance.
It validates target fidelity and experiment integrity; it does not claim this
method already produces a predictively superior model.

The old side-score difference mixes magnitude with conditional direction. A
controlled example retains exactly the same P(up | big) values and endpoint
classes while changing only P(big). The old difference-score direction AUC moves
from 1 to 0; the conditional-probability AUC stays exactly 1. The new method
reports the old joint score as an additional diagnostic and freezes conditional
direction separately. Magnitude AP and absolute-move ranking receive their own
metrics. Both information types must improve for a complete representation to
qualify; an AP increase with direction merely not degrading is insufficient.

Another controlled case shows `nextafter(5, 0)` remains below the native five-tick
boundary in float64 although casting it to float32 would produce 5. The evaluator
uses original float64 endpoints and original up/down indicators and rejects a
contradictory codebook. Equal-score AP returns the actual base rate; Top1% retains
the existing fractional boundary-tie procedure. Predetermined nonoverlap uses
time and symbol alone, independent of model scores or outcomes.

Thirteen contract tests validate chronological roles, 2026 history floor,
future-only sealed OOS, train-only selection, source/input/profile drift,
manifest rehash rejection, exact five-part row identity, all-four native label
hashes and cross-process GPU serialization. Twelve mechanism tests validate
future-prefix invariance, strictly prior session references, missing calendar
slots, missing peer/anchor behavior, no cross-day carry and physical-resolution
handling of exact zero and floating midpoint perturbations.

Six metric/data tests cover the direction invariance and native boundary cases,
tie AP, nonoverlap and time-only thinning. Eight signal CLI tests prove the
versioned profile is resolved before the runner is imported and legacy scientific
overrides cannot change the comparison. Six orchestration tests verify all seven
primary fits and six secondary fits, nomination before development scoring,
permutation exclusion, group guards and absence of completion after a failed
secondary step. These use mocked fits and do not spend GPU training budget.

All 45 new targeted tests pass in 2.677 seconds. The existing AstraResearch suite
plus the new tests discovered at its launch passes 783 tests in 181.517 seconds;
the six later-added orchestration tests are included in the targeted 45. Ruff
checks pass. No C++ source was changed in this round.

The actual no-fit preflight passes every arm/role combination. Seven arms have
identical ordered native sample keys, all-four endpoint labels and role masks:
564,264 train rows, 49,788 ES rows, 106,284 calibration rows and 221,280 development
rows. Baseline has 3,386 inputs; core has 159; complete information state and its
permutation diagnostic have 178. CatBoost Pool schema checks perform no fit.
Four fixed stocks' labels are absent from train, ES and nomination; development
group evidence is explicitly transductive and already exposed.

The scientific profile, source dependencies, parent projections, origin mids,
fitted graph, prepared state columns, preflight and package-version receipt are
hashed before training. Every fit verifies the same freeze and takes the single
global GPU0 lock. Candidate gates, label/sampling/split choices and uncertainty
parameters cannot change within this comparison. Source corrections or a new
judge require a new recipe and baseline comparison.

The prior signal study scored through 20260924. Existing material is therefore
development evidence. The reserved final OOS starts no earlier than 20261001
and still requires a nominated frozen candidate, protected future material,
thirty untouched trading days, identical baseline/candidate scoring and all
frozen gates. There is no final-OOS result yet.
