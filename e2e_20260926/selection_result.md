# Q1: broad quality-only selection did not advance

The registered first-fold comparison fails its continuation gate. Q retains 3,123 quality-qualified native columns; A retains 932 after the existing IC/redundancy selection. With the same frozen learner, cost model and policy rule, Q loses more after costs. Keep the canonical selection default; do not run the later folds or seed search for this registration.

| Eight observed validation days, July 2–15 | A | Q |
|---|---:|---:|
| Round trips | 54 | 81 |
| Gross PnL, TWD | 1,500.000 | -500.000 |
| Fees/tax, TWD | 12,411.736 | 29,666.256 |
| Net PnL, TWD | -10,911.736 | -30,166.256 |
| Net at 1.2 times charged costs, TWD | -13,394.0832 | -36,099.5072 |

Q minus A is **-19,254.520 TWD**, improving only two of eight days. Q satisfies the minimum 20-trip support gate, but fails both positive net and positive paired net. Both arms end every day flat. July 6 remains the preregistered missing recording; all 14 ineligible symbol-days remain explicitly recorded, with no outcome-based exclusions.

Native train/tune/forward inputs are fully paired: 691,953 / 267,796 / 115,825 origins, all keys, 3,407 export columns, labels, sentinels, schemas and omissions agree. Complete PMQ statistics are byte-identical and Q's ordered selection is exactly the registered set. Both Python/native inference gates pass. Both fits use CatBoost RMSE, depth 6, up to 600 trees, seed 1729 and the same early-stopping rule; A retains 196 trees and Q 342. Numeric q95 thresholds are learned separately on the same calibration population, as registered.

Diagnostic interpretation: Q's forward top-decile gross returns barely change (buy 0.8695 versus 0.8750 ticks; sell 0.7566 versus 0.7497), below the 1.7451-tick fee reference before spread/fill selection. Extra activity therefore does not establish stronger usable information. The 3037 stock contributes most incremental fees (4,252.064 to 19,127.534 TWD; two to nine trips), despite higher gross profit. Excluding that symbol still leaves Q minus A at -12,379.050 TWD; this is a falsification diagnostic, not a revised evaluation universe.

An independent Decimal/FIFO reconstruction verifies every native fill, round-trip cash and fee total, per-fill position and flat ending against published execution artifacts. No synthetic fills or canceled-order counterfactuals enter this evidence. Reproduce the audit with `python3.13 build/end_to_end_research_20260926/audit_q1_cash.py`; its receipt SHA-256 is `8989151627f783bdb4f7a86af1eca31e2a54582493304629922d2720582603f5`.

Evidence lives under `build/end_to_end_research_20260926/`: `q1-F1-result/`, `q1-F1-paired-{train,tune,forward,selection}.json`, `q1-F1-paired-predictive-metadata.json`, and `q1-independent-cash/`. Published Q model / execution / report identities are `15829c3f515cc25d9ce81cf4e5584b8b9bd9d6661b3efb13560ba03b8433050a`, `2c66c12e5c7870cd8bd6d4084b004625b820fb57eeca7a651e7bc4257c286e4d`, and `cd0c4a1ebcb78cac923df2863de315e4a64c8205a5e12924025551ffeebd3cb1`.

This is one exposed validation fold and seed, using immutable correctness-v2. It rejects this direct selection broadening under a fixed learning/execution setup. It does not prove general PMQ superiority or the conditional uselessness of excluded features, and does not validate the later correctness-v3 candidate. No final OOS was used and no economic improvement is promoted.
