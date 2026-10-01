# Prospective H15 compact local context: source and descriptor audit

Status: source/schema audit before any V4 label, feature-value, model or score
read. This is a new prospective representation, not an admission or evidence of
predictive improvement. Selection below does not retain the old 27-column core
by default. No native feature math, material, row cohort or frozen producer was
changed for this audit. Python may select these existing native columns; every
new material feature remains a C++ FeatureModule under `experimental/`.

## Three conditioning roles

The local C representation gives the external H15 mark three complementary
contexts: physical movement/activity opportunity; local directional anchors and
known demand; and conditional response/net observed supply. It intentionally
omits symbol IDs, raw price/volume/depth, a full historical feature catalogue,
and any proposed five-tick traversal cost. L5 geometry does not establish the
resistance to a five-tick MID move. H15 introduces the external information
source; C is a compact source-audited replacement context, not the independent
novel-mechanism claim.

## Exact C18 numeric and C2 categorical order

All numeric indices below are identical in the original train/tune/forward
3296-column `projection.json` recipes. Both categorical descriptors are present
as `large_string` in all three original `nominal.parquet` schemas. The indices
are provenance checks, not a substitute for selection by exact descriptor.

| # | Exact descriptor | Parent index | Role / units / boundary |
|---|---|---:|---|
| 1 | `SpreadState.0.cur_spread_ticks.0` | 2643 | Current full-share-ladder integer spread ticks; physical opportunity context, not fees. |
| 2 | `PriceFormationPath.0.sticky_rv_ticks.1` | 1958 | Existing 60s sticky-path root sum of squared fractional tick moves; retains physical variation magnitude. |
| 3 | `IntradayRegime.0.magnitude_session_range.0` | 1202 | `log1p(abs(integer ticks(low,high)))`; session range is quantized physical ticks, not mid range. |
| 4 | `TimeInfo.0.session_time.0` | 2811 | ALPHA_FACTOR: logical receive-clock elapsed integer minutes from 09:00, common session context. |
| 5 | `IntradayRegime.0.direction_open_displacement.0` | 1207 | `copysign(log1p(abs(fractional ticks(open,sticky))),delta)`; directional physical anchor. |
| 6 | `IntradayRegime.0.direction_vwap_displacement.0` | 1208 | Same signed log-tick displacement from session execution VWAP to sticky price; neither percentage nor range-normalized. |
| 7 | `IntradayRegime.0.direction_range_position.0` | 1209 | Bounded `2*distance_from_low/range-1`; exactly zero when range is zero; no epsilon. |
| 8 | `BookSideCapacity.0.book_imbalance.0` | 393 | `(bid_L5-ask_L5)/(bid_L5+ask_L5)`; dimensionless displayed-capacity share. |
| 9 | `BookClockFlow.0.signed_trade_over_depth.2` | 163 | Existing 60s signed confidence-weighted execution quantity / mean combined TOUCH quantity. |
| 10 | `BookClockFlow.0.trade_over_depth.2` | 169 | Existing 60s total observed execution / same mean combined TOUCH quantity. |
| 11 | `BookClockFlow.0.known_trade_share.2` | 175 | Confidence-weighted known/total observed executions; no executions gives NaN, not known zero. |
| 12 | `BookClockFlow.0.net_supply_imbalance_over_depth.2` | 217 | Existing 60s signed net observed refill/depletion / same mean TOUCH quantity; no gross-liquidity claim. |
| 13 | `TradeVolumeRegime.0.vol_regime_rel_total1.0` | 3144 | Current 120s total / mean of eight previous 120s spans; current excluded from baseline. |
| 14 | `FlowResponseSurprise.0.flow_surprise0.0` | 998 | Existing 300s prior reference, flow innovation in its prior capacity-relative flow standard deviation. |
| 15 | `FlowResponseSurprise.0.response_innovation0.0` | 1001 | Existing 300s tick/second response residual after prior joint flow/price slope, divided by prior residual scale. |
| 16 | `FlowResponseSurprise.0.response_coupling0.0` | 1004 | Prior completed-interval correlation, bounded [-1,1], exactly zero for zero covariance-scale support. |
| 17 | `PressureResponseState.0.absorption_bid_net_supply_ratio.0` | 1871 | Net original bid-price supply / initial displayed reference; includes later quote-only changes while observable. |
| 18 | `PressureResponseState.0.absorption_ask_net_supply_ratio.0` | 1883 | Same ask-side net supply ratio; zero is an observed zero, inactive/unobservable is missing. |
| cat 1 | `CurrentBook.1.sticky_side.0` | nominal | Native A/B/M anchor location; exact parent INSTANCE 1. |
| cat 2 | `PriceFormationPath.0.cause_dominant.1` | nominal | Existing 60s native process-count category: none/tie/1--7; observed classification, not economic causal identification. |

Common physical tick units are meaningful for a target specified in ticks.
They need no mechanical divide-by-five: that constant would change neither
information nor pooled comparability. Signed log-tick compression is already
native. Percentage returns express another role; they are not interchangeable
with target-event-relative five-tick displacement. Ratios cancel quantity scale
where numerator/denominator describe the same capacity or volume unit. Relative
activity and standardized innovations measure surprise, not absolute movement
opportunity: spread, physical variation and session range retain that separate
information. Equal units do not prove equal conditional sensitivities across
symbols; prospective pooled/group diagnostics must still test generalization.

## Exact formulas and missing-state responsibility

Source roots are `src/oms/modules/feature/`:

- `market_state/session/intraday_regime/intraday_regime.cpp`: session VWAP is
  cumulative price-times-volume / cumulative volume. `direction_vwap_displacement`
  calls `get_tickfs(vwap,sticky)` and signed `log1p`. The similarly named
  `intra_regime_vwap_displacement` instead divides by `(range_ticks+1)` and is
  not selected. Neither field is `SessionVWAPDeviation.norm_vwap_dev` (a price
  fraction). VWAP floating summation can give tiny signed numeric displacement;
  we do not turn that into an invented categorical zero/nonzero decision.
- `market_state/session/time_info/time_info.cpp`: `session_time` is registered
  ALPHA_FACTOR, P0/S2/Snapshot, and uses `(clock.now()-open_time)/60000000`.
- `order_flow/imbalance/flow_response_surprise/flow_response_surprise.cpp`:
  interval flow is signed quantity / prior TOUCH mean depth / seconds; movement
  is fractional MID ticks / seconds. The depth uses `max(1,mean_touch_quantity)`.
  This is a physical one-share floor, not a `+1` denominator or numerical epsilon;
  it is inactive for valid positive integer queues on both sides. Prior points
  are evaluated before insertion. Reference variance floors are 0.01
  capacity-relative flow per second and 0.1 tick per second, clip 8. Slope is zero
  unless `xx>flow_floor^2*n`; coupling is zero unless `xx>0 && yy>0`. There is no
  epsilon or `+1` in these innovation/correlation denominators. Zero-quantity
  interval known-share convention is one. Known attribution here can include
  confidence >=0.5; it must not be misdescribed as H15 confidence-one work.
- `order_flow/book_clock_flow/book_clock_flow.cpp`: names
  BID_DEPTH/ASK_DEPTH hold TOUCH quantities; L5 totals are used elsewhere.
  Ratios use denominator `>1e-8`, otherwise NaN. Positive integer observed queues
  keep that guard away from the capacity boundary; no executions gives missing
  known-share. Confidence-weighted supply attribution is not strict known demand.
- `order_flow/trade_flow/trade_volume_regime/trade_volume_regime.cpp`: total activity
  divides by `(history-current)/8` only when positive. Mature current zero with
  positive prior baseline is exact zero; zero baseline is NaN; unfinished history
  is -inf. Other epsilon-shifted side-imbalance variants are not selected.
- `microstructure/liquidity/pressure_response_state/pressure_response_state.cpp`:
  supply adds `current_quantity-previous_episode_quantity+attack` on EVERY valid
  book. Reference is positive initial queue; ratio has no epsilon/+1. Original
  price can survive touch renewal while ObservableBook retains observability.
  This is net observed quantity, not separate gross replenishment/cancellation.
- `microstructure/book/book_side_capacity/book_side_capacity.cpp`,
  `microstructure/book/spread_state/spread_state.cpp`, and
  `microstructure/event_path/price_formation_path/price_formation_path.cpp` own the
  displayed-capacity/spread/path categories. Parent material, not a freshly
  rebuilt implementation, determines all original cached numeric bits/states.

The original projection float32 cast preserves its NaN, -inf, +inf and signed
zero encodings. Selection must not impute, repair, normalize, rescale, relabel
missing states, or remove origins on feature availability. CatBoost conversion
is scoped to the final model input. Native H15 material retains float64 until
that conversion, with exact raw-mid uint64 comparisons and exact five keys.

## Historical producer evidence and limits

Original binary is
`runs/fe_origin_20260930/frozen-native-engine/coco`, SHA256
`2374c82c3fd60f02e636fd2eec54329b60324228b8febefc29e463b3ae57ff60`;
`receipt.json` binds its original runtime dependencies/fingerprint. A metadata-only
invocation of THAT binary's `feature-guide` confirms FlowResponseSurprise is
registered AlphaFactor with History 300/900/1800s, SampleInterval1s, MaxGap5s,
MinSamples60, MinKnownSideShare0.8, MoveStddevFloor0.1, FlowStddevFloor0.01, Clip8.
The older saved `runs/fe_origin_20260930/feature-guide.yaml` omits that module and
must not substitute for this binary's registry evidence.

Exact original native config independently checked by root is the
20260119/2330 config in native artifact `9c10e19edefa61c37bd7ba4c5c3d2d932eef9eb4a20815d4a1595d381b8bff1d`:
BookClockFlow60s is slot2, PriceFormationPath60s is slot1, FlowResponse300s uses
name suffix0 and field slot0, TradeVolumeRegime120s is name suffix1/field slot0.
The corresponding original-export config under
`runs/fe_origin_20260930/sampling-cross-fixed-1/2337/reference_ten_second/config.yaml`
confirms those exact parameter/index mappings without value reads.

`experiments/fe_origin_20260930/SOURCE_AUDIT.md` explicitly records the later
FlowResponseSurprise warmup correction: original unfinished-reference NaN can
be corrected-producer -inf in all NINE fields. Finite values stayed exact in the
documented fixed engineering controls. Original scientific projections stayed
unchanged. Source inspection plus these bounded controls is not a universal
original source/binary equivalence proof. C MUST select the original vintage
states from the full 3296 projection; it may not silently import corrected
warmup states or equate the old 3254 selected baseline to the full source pool.
The three selected Flow fields are absent from 3254 current_nominal selection,
but present in original projection at indices 998/1001/1004.

Schema-only projection receipts (no matrix/label values read):

| Role | `projection.json` SHA256 | x.npy header | categorical schema |
|---|---|---|---|
| train | `5474e72d7cb0011dcaa5a525eb19198f325ffa869246017ea55dcf664d3c6f06` | float32 (2253792,3296) | both large_string |
| tune | `0d5685f99c3123b6b299919c62074e0da6cab5900bd54b902061c38a7ef1c809` | float32 (623872,3296) | both large_string |
| forward | `12eb2c681b76a3089253ad77576ec4b3a503062f545eb427c3323de93a670664` | float32 (662880,3296) | both large_string |

## Deliberate omissions and falsification limits

There is no direct five-tick traversal-resistance field, hidden-order identity,
gross cancel/add decomposition, participant identification, label-driven sector
grouping, or outcome-selected peer in C. Raw PriceMemory volume logs do not
cancel cross-symbol scale and are omitted. DailyAnchor would require a separately
bounded prior-data lineage and is omitted. All six/hundreds of window variants,
the old 27-core derived Python math, and symbol IDs are omitted.

Pressure/flow/PPath are local context, with existing observation limits and short
episode mechanics; longer-horizon sufficiency is not assumed. C can fail because
this compact conditioning loses important local context. H15 can fail because
external marks add no useful information, because coding adds nothing over K,
or because support/drift differs. Those alternatives require the prospective
same-cohort same-judge comparison, not an absence claim from source reasoning.
No feature availability exclusion, OOS symbol cherry-pick, support-gate rescue,
window extension, or hyperparameter expansion is authorized by this audit.
