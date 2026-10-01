# V2: sampled price renewal, with the V1 judge unchanged

## Failure-driven question

V1's replacement arms failed calibration nomination and development acceptance.
The compact core already loses full-baseline information. Its graph addition
improves conditional direction on development, but not magnitude or joint AP.
Consequently the replacement result cannot determine whether the new state is
useless as complementary information. This round includes the missing control.

In the train-only graph census, median absolute 60s movement equals zero for
15/16 symbols; even 300s has zero medians for 5/16. A Gaussian volatility-like
scale collapses distinct questions: whether an observable price renewal will
arrive, how far it will move, and whether successive marks share direction.
H9 explicitly represents that marked-renewal mechanism. This is a sampled-mid
process, not an assertion about unobserved physical quote events.

## Predeclared arms and judge

Four 300s fits, in order: fresh full baseline; full baseline plus the original
19 graph/session/occupation states; full baseline plus compact H9; compact core
plus the same H9. All four keep the same 132 native categorical inputs.
Complement arms test added information without discarding context. The compact
replacement tests whether the new mechanism can replace that context.

Every scientific judge field is copied exactly from V1: original 30s origins,
all-four-horizon endpoint support, pure native float64 +/-5 tick labels, fixed
date roles and symbol label holdout, identical CatBoost Multiclass/weights/
budget, metrics, nomination, group guards and uncertainty acceptance. The
untouched V1 trainer and scorer implement this round. All arms are eligible;
selection uses calibration before any development comparison. Six inherited
secondary fits compare baseline and the nominee, or predeclared core+H9 on
failure, at 60/120/180s. No HPO or whole-day future permutation.

The new profile has a distinct comparison identity because the FE arms changed.
Its key/role/label population hashes must nevertheless equal the V1 preflight.
The old profile, manifest, identity anchor, scientific sources and native caches
are immutable inherited dependencies. Shared artifacts use local hardlinks,
not retargetable symlinks or physically duplicated datasets.

## Semantics and empirical rejection

H9 is built only from native OriginMidPrice, symbol and sampling timestamps.
Physical half-cent midpoint identity and exact piecewise tick coordinates avoid
turning floating-point closed-mid noise into false renewals, dwell resets or
signed episodes. This transforms inputs only; native endpoint labels and their
float64 threshold semantics remain unchanged. H9 is restricted to the declared
ordinary TWSE stock universe whose native ladder it uses.

Prior activity references use only completed earlier calendar slots and the
same clock bucket. Missing days do not extend the reference lookback. Renewal
arrival and marks have separate denominators; zero, missing support and censored
first dwell remain separate states. All new numerical inputs use bounded
probabilities, physical tick units or prior-only semantic reference units.

Retain a hypothesis only through the unchanged calibration and development
rules. A complement gain without replacement gain identifies information that
still needs semantic rebuilding; it does not admit raw legacy AlphaFactors as
normalized final inputs. Failure of all arms sends the next question back to
information source, cohort and mechanism, without weakening the judge.

All current validation dates have already been exposed. A nomination is
development evidence only. Future OOS remains reserved after 20260930, requires
at least 30 newly unobserved days and is not scored by this round.

## Reproduction

From AstraResearch, run the new helper with profile-v2.yaml for `features`,
`preflight`, then `freeze`. Training and verification use the existing official
CLI:

```
python3.13 -m astra.signal representation --profile experiments/information_state_20260930/signal-v2.yaml --stage run
python3.13 -m astra.signal representation --profile experiments/information_state_20260930/signal-v2.yaml --stage verify
```

The helper closes the new FE dependencies before the official runner fits.
Never run the old generic `features`/`freeze` stages on the V2 recipe.
