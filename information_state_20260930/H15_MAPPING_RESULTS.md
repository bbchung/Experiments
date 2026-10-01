# Fixed H15 reference mapping: rejected

The prospectively fixed sixteen-symbol mapping (peer2330, except target2330 uses peer2317) fails its source-support gate. The pair pilot remains positive source evidence, but it does not qualify this mapping for full-period materialization or training. No labels or models were read.

Producer: `63d23532f91ba6f4027ddb49b33de0747fc55e2f8ea0bc28ab912ae569a0960d`.
Evaluator: `6746c4d4957dba539b21a076940fc58e71acf524378810a00b117a673284ec66`.
Binary: `613260125eb524b3e2c5af77a8621083b28cacfedc488d421c06e7709961fd6b`.

All32 fixed symbol/day cells are present, and their native schema, original observation keys, raw mid bits, availability, missing states and counters pass integrity validation. Aggregate directional and ordered-response support passes. However, the unchanged per-cell requirements (five session-qualified marks and five distinct recent sampled marks) fail:

| Target | Day | Session marks | Distinct recent marks | Ordered-response origins |
| --- | --- | ---: | ---: | ---: |
| 3481 | Both | 0 | 0 | 0 |
| 2344 | Both | 0 | 0 | 0 |
| 2337 | Both | 0 | 0 | 0 |
| 2409 | 20260119 | 2 | 1 | 19 |

The three zero-support symbols are target-invalid at every original30s origin on both days, although the common peer remains quote/stale and has confirmed source marks. Their closed target availability stays zero and hard-censor counters rise. Direct Jan19 raw Book witnesses from the already-bound receive-prefix spool have StatusMask3 (`TRIAL|AUCTION`), legal prices and positive reported queues. The C++ module rejects noncontinuous status before price-grid tests; this is expected behavior, not an evidenced arithmetic bug. Finite CurrentBook mid/tick exports alone do not certify continuous matching.

The local daily disposition CSVs list all three symbols with prior publication dates and active intervals covering both days, and describe five-minute matching. This supports a separate mechanism/eligibility research question. It is not permission to change this mapping's gate or select its successful targets. The BasicInfo day_trade permission is not a pure-signal regime rule. 2409's Jan19 limited/one-sided observations are a separate availability issue; its Jan20 cell passes.

The full-materialization draft is suspended before preparation. A future universe/regime methodology must be studied, validated and frozen independently, then comparisons must restart with a fresh baseline and identical candidate cohorts.

Authoritative receipt: `../../runs/information_state_20260930/native-h15-mapping-support/coverage-validation.yaml`.
