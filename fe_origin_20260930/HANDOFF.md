# FE origin research: committable stopping point

Paused at the user's request on 2026-09-30. All STOCK, TXF, EXF and H5 studies are complete. Closeout performs validation and documentation only; it starts no new hypothesis or fit. No Git commit has been created. Generated material, models, receipts and failure logs remain outside the proposed commit under `AstraResearch/runs/fe_origin_20260930/` and its referenced stores.

## Recorded result

The fixed target is native mid-price endpoint change at 60/120/180/300 seconds, with signed five-tick events and 300 seconds primary. All history is on or after 20260101. Missing symbol/day/contract inputs are skipped inside their original date roles, without fetching, filling or changing role boundaries. Admission is pure signal and excludes execution/cost rules. The first bounded panel contains 16 stocks plus TXF and EXF; it does not cover every possible symbol.

STOCK `current_nominal` passes all eight predeclared development-forward checks after calibration-only nomination. It uses 3,254 current numeric inputs plus 132 original native strings with the same three-class `{none: 0, up: 1, down: 2}` endpoint objective. The 30 forward days are 20260812–20260922, with 662,880 common scored origins.

| Primary 300-second metric | Numeric control | Current + original nominal inputs |
|---|---:|---:|
| Pooled side-mean AP | 0.123212 | 0.128588 |
| Direction AUC given big endpoint | 0.498958 | 0.522806 |
| Symbol-day AUC | 0.741589 | 0.745397 |
| Correct minus opposite Top1% | -0.003846 | 0.015611 |
| Non-overlap AP | 0.141834 | 0.147787 |

Daily-equal-weight paired AP difference: mean +0.004192516, predeclared three-day-block interval [+0.001241788, +0.010588278], 2,048 draws. Non-overlap support is 344 up and 278 down events across 30 days each. This is support for this recipe relative to its numeric control. Existing tools already have categorical support. Prior research exposure remains, so this is not pristine OOS, trading proof or a universal baseline improvement. Top1% selection is concentrated: the largest symbol contributes 29.6% of selection mass and the top three 73.1%.

Neither TXF nor EXF passes its primary comparison. The new strictly-prior FlowResponseSurprise representation has insufficient availability under existing PMQ thresholds and no admitted mechanism gain. H5 separates P(big) and P(up|big): pooled forward AP improves to 0.132759 and direction AUC to 0.530826, but calibration nomination, daily AP lower bound, net tail precision and non-overlap AP fail. Its original decision remains rejected.

Full tables are in the retained, Git-ignored [run report](../../runs/fe_origin_20260930/RESULTS.md); hypothesis motivation and frozen controls are in [PLAN.md](PLAN.md) and [FACTORIZED_PLAN.md](FACTORIZED_PLAN.md). Direct source findings and qualifications are in [SOURCE_AUDIT.md](SOURCE_AUDIT.md).

## Commit scope

- Pure material workflow: absolute history floor, explicit native multi-horizon endpoint bank, market-valid stock population, missing-input skips, explicit peer subscriptions and additional native declarations.
- Native FE: new FlowResponseSurprise producer and warmup contract tests; CrossReturnContext halt-boundary correction with sampling-invariance evidence. Revised producers require fresh material lineage before predictive use.
- Native inference: explicit three-class endpoint CBPredictor route with actual integer-class validation, small deterministic fixtures and existing binary/regression compatibility tests.
- Research infrastructure: tie-correct AP and fractional-boundary Top1% metrics; guarded faster canonical YAML emission that preserves the old byte identity.
- Experiment sources, profiles, cropped 2026 calendar, contract tests, source audit and this handoff. Generated reports, runs/stores and large outputs stay ignored. Unrelated `shioaji.log` remains untouched.

Suggested commit subject: `feat: add pure endpoint FE research and native classifier validation`.

## Final validation and preserved identities

- Release build succeeds; all 100 CTest targets pass in 9.14 seconds.
- `python3.13 -m unittest discover -s tests -v`: all 744 tests pass in 167.067 seconds.
- Experimental `test_*.py` discovery with `PYTHONPATH=.:experiments/fe_origin_20260930`: all 15 tests pass.
- The actual supported model scores natively on 2609/2337/2481, 20260203: all 3,386 selected inputs and all original labels are bitwise identical, maximum C++/Python probability difference is 0. The exact-model admission guard passes. Excluded, intentionally revised producer fields are recorded separately.
- The new core YAML emitter reproduces every byte of the 298,318,912-byte published stock train manifest in 12.48 seconds, retaining identity `081a1f0e901e6b270fd1830ced622b02f942a7cca69504baa698a13d17356892`.
- All frozen study source hashes match their protocols. Final test logs and runtime details are in `runs/fe_origin_20260930/final-validation/`.

Original scientific runtime fingerprint: `4700f38167e0a8256172a8bdcc3d262e157bebb1b7fe1ab6dd2596bf5945fd1a`; preserved at `runs/fe_origin_20260930/frozen-native-engine/coco` before rebuilding. Final Release runtime fingerprint: `8d9d2a5d2fcf3bc650894b51321e482b12c83ecfee74feb84476b770389d8033`. Existing frozen profiles retain their original paths and identities; changing their runtime/source recipe requires a new run, not a silent resume.

Stock tune identity: `708af24f00387338b3349c3763ade209956b576f36b39d060406d9b0e916ce06`. The exact input lists, material bindings, CBM bytes and original decisions remain in `study-stock/protocol.yaml`, `nomination.yaml`, `primary-decision.yaml` and the fitted-model receipts. Native parity is in `native-endpoint-stock-release-1/results.yaml`. Diagnostic plots and per-symbol concentration CSV remain under the run root.

## Unfinished work preserved for the next research direction

Full factory three-class training/model-artifact/DAG wiring and baseline packaging are not complete. The explicit native scoring route and this development admission are ready, but must not be described as completed factory integration. The user-requested pause takes precedence over continuing that integration.

The inventory covers 420 registered native types and 1,703 families; it does not prove every formula or predictively test every registered type. Documented unresolved representation problems include unusable daily/session priors in this recipe, vanishing event-clock denominators, floating-mid closed-path amplification and EXF tick-boundary arithmetic. The originals remain in frozen data. Any new corrections, daily-history producer, normalizer, changed objective or more aggressive hypothesis needs a new declared recipe and material lineage; previously read development-forward dates remain exposed.

The authorized old-material cleanup freed 137.3 GiB. Current stores/models and original market inputs are preserved. Exact deletions and the metadata-only archive are recorded in `disk-cleanup-20260930.json`; archived metadata is not reusable material.
