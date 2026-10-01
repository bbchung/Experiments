# V1 results: failed replacement nomination

Frozen identity: `9d172c97ce3b02b138bec8a9039225586ea5f3654a6bda95362e5d2f0fdd577a`.

All 13 predeclared CatBoost fits completed. No calibration nominee; no accepted representation; no sealed OOS scoring. All rows, labels, splits, training and acceptance were frozen before the first fit.

| 300s arm | calibration joint AP | calibration big AP | calibration direction AUC | development joint AP | development big AP | development direction AUC |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.200015 | 0.404431 | 0.558634 | 0.126787 | 0.260884 | 0.583612 |
| core | 0.183824 | 0.382811 | 0.545909 | 0.122430 | 0.257823 | 0.563971 |
| graph | 0.186774 | 0.385354 | 0.551376 | 0.118960 | 0.252912 | 0.575078 |
| session | 0.189871 | 0.389544 | 0.532443 | 0.122430 | 0.253547 | 0.560227 |
| occupation | 0.183412 | 0.385386 | 0.529385 | 0.122262 | 0.257808 | 0.554789 |
| information_state | 0.186819 | 0.388582 | 0.539115 | 0.120239 | 0.253796 | 0.577296 |
| state_permuted_control | 0.202228 | 0.385800 | 0.626726 | 0.136683 | 0.259563 | 0.666270 |

The whole-day permutation is a noncausal diagnostic and never eligible. Its gains cannot establish usable current-time signal.

| Baseline development horizon | large-move rate | big AP | conditional direction AUC | equal symbol-day direction AUC |
|---|---:|---:|---:|---:|
| 60s | 0.295% | 0.131117 | 0.711151 | 0.654643 |
| 120s | 0.812% | 0.181642 | 0.667263 | 0.660742 |
| 180s | 1.389% | 0.208926 | 0.623132 | 0.576046 |
| 300s | 2.542% | 0.260884 | 0.583612 | 0.545231 |

## What the rejection establishes

The compact core already loses baseline information. Graph adds conditional direction to that core (.563971 to .575078; equal symbol-day .5271 to .5553), but loses magnitude and joint AP. The complete replacement hypothesis failed; absence of complementary information has not been established. V2 adds the missing full-context control and a new marked price-renewal hypothesis under the same judge.

Train-only median absolute 60s returns are zero for 15/16 symbols, and 300s for 5/16. This motivates separate arrival/conditional-mark/directional-episode state rather than Gaussian variation scaling.

A representative 12-day native prefix audit recomputed 265152 rows and compared 253632 prefix origins. All three state families match stored outputs and prefix/full outputs exactly, including NaN; zero fits. This establishes causal implementation integrity, not predictive admission.

The native C++ baseline contains 300s price/pressure path and realized-variation windows; it is not literally limited to short windows. Its dominant local rolling-event summaries do not thereby express completed-prior-day clock-conditioned arrival/mark references. Longer windows alone do not provide those references.

Current dates are historically exposed development dates. The future-only OOS reservation starts 20261001 and needs at least 30 newly unobserved days. No final OOS claim is made.

Artifacts: `runs/information_state_20260930/v1/comparison.csv`, `nomination.yaml`, `development-decisions.yaml`, `native-prefix-validation.yaml`, and `fits/*-metrics.yaml`. The unchanged package and methodology documents are preserved in the run source snapshot.
