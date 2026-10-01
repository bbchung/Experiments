# H15: receive-owned external discovery and later target response

Status: prospective source-grounded hypothesis only, outside every existing
frozen comparison closure. No market data, actual labels, scores or models were
read for this plan; no implementation, replay or fit was performed. Root must
review and freeze a support contract before observing counts. Material features
will be C++ FeatureModules under `src/oms/modules/feature/experimental/`, with
native DatasetWriter exports. Python may orchestrate or validate artifacts; it
must not calculate this representation.

## Question and falsifiable interpretation

A peer's new joint quote followed by a strictly known execution at that new
touch may convey external price discovery. Does remembering **which peer mark
became available before which target response** add direction or large-move
information at existing 30-second origins, beyond the same independently
observed peer and target marginals?

This is a different information source from local replenishment/repair. It is
also a narrower claim than informed trading, common fundamental value, economic
causation, or exchange-clock leadership. Receipt order establishes what this
engine could know; vendor buffering can influence that order. H15 must survive
an explicit causal marginal control before that observation is useful evidence.

The fixed cheap pilot is 20260119/20260120, in both directions of the already
declared 2308/2317 pair. Earlier TRAIN-only microstructure eligibility supports
using these cells; it does not prove their relationship or predictive value.
Unavailable inputs are skipped and reported. No replacement dates, symbols,
lead/lag offsets, thresholds or event-origin grid may rescue a failed gate.

## Native interface and observation ownership

Proposed leaf: `experimental/peer_information_transfer/`, Type
`PeerInformationTransfer`, one instance per target with typed `TargetSymbol`
and `PeerSymbol`. A single module owns both legs and follows the Book/Trade
routes of each leg's existing `TwseFilter` configured `RequireTradable:false`.
That route is a source contract, not an eligibility screen based on trading
permission. The module must receive noncontinuous boundaries. Root should
freeze exact routing after checking filter semantics; silent removal of status
callbacks is unacceptable.

The existing interface supports this without Python feature calculation:

- `Module::on_book/on_trade` receives `Quote<T>`, whose contract identifies the
  leg and whose header contains receive time/sequence. `CrossLeadLagResidual`
  already dispatches target/reference books by `quote.contract().symbol`.
- `Module::bind_feature/bind_input` establishes dependency order; if a producer
  supplies a typed mark through `register_value`, lookup occurs in `on_start`
  and `follow_quotes` joins its feeds. A mark accessor alone is not a join:
  explicit producer dependencies remain required. Prefer direct ownership of
  the two event streams for this first bounded module.
- `CausalTradeSideInference` owns a trade-price/tick-rule history, but its caller
  owns book freshness and status. Keep a separate inference object and pending
  exchange cluster per leg. Never infer the peer against the target's book.
- `ObservableBook` and ordinary TWSE-share ladder validation govern observable
  quotes. Exact physical quote identity precedes ratios and normalized moves.
- The leaf OBJECT library links `module` and appends GLOBAL `MODULES`, as other
  experimental leaves do. The recursive module CMake discovers leaf files;
  do not create a parent CMake that also adds those leaves.

For each leg separately, R is monotone effective receive time, E is that leg's
monotone exchange clock. Validate integer boundaries, positive clocks below
2^53, supported price grids, observable positive touch queues and source epoch.
The two E domains are never compared, including for freshness or leadership;
E may exceed R/S. Constant +/-1-hour changes to either leg's E must preserve all
predictors. Per-leg E/R differences can check own-book freshness within five
seconds. An exchange cluster commits only when that leg's next distinct-E
callback arrives, at that callback's R. EOF and logical-clock ticks never close
it. Whole-cluster mixed/ambiguous attribution rejects its pending mark/work.
Fixed capacity is 256 observed callbacks per open cluster; overflow rejects
the entire pending group and is diagnostic, not partial published evidence.

Global R only orders **availability**. Equal-R callbacks from different legs do
not establish peer-first evidence from loader order. A target response is later
only if its causal callback R is strictly greater than the peer mark's committed
availability A. Finalizing at A may inspect target state available strictly
before A; any same-R target callback invalidates that mark's baseline/ordering
qualification before it can receive directional credit. Ties have an explicit
category. Prefix replay with real carriers must establish A<=SampleTime and
that unfinished groups neither flush nor backdate.

## Price/print mark and subsequent response state

1. A valid peer-only joint bid/ask change in one direction opens a candidate.
   Freeze previous/current peer pair, direction, visit ID and positive attacked
   touch capacity. The entire formation cluster must have zero positive prints.
2. A distinct later closed peer cluster must contain confidence-one execution
   at the exact new attacked touch: BUY at new ask for an upward candidate,
   SELL at new bid for a downward candidate. Through-touch, inside-spread,
   conflicting, stale or mixed demand cannot confirm it. Pair departure before
   the known execution ends that candidate. Confirmation freezes a historical
   source mark at A, including its available work and both price anchors.
3. At A, freeze the latest valid closed target mid/pair strictly available before
   A and no older than five receive seconds. Target unobserved or tied means no
   qualified transfer baseline. Do not obtain a new baseline from a future book.
4. Later closed target books and known target execution groups update one
   response state linked to this source mark ID. Retain own target displacement,
   normalized work, first nonzero same-direction joint quote response and any
   opposite response. All updates remain available only on closure. Unknown
   current target attribution cannot become known zero work.
5. Keep at most one latest confirmed historical source mark and one linked
   response state. A new qualified source mark supersedes ownership. A peer
   return through the pre-move pair records refutation; it does not retroactively
   remove the historical confirmation. Target following or opposing quotes are
   observed outcomes, not evidence of an expected beta-implied destination.

The state has no 60/300-second expiry. Age is an explicit predictor, not a hidden
lookback. A historical confirmed mark does not disappear solely because ordinary
unknown demand, a quiet peer, or target pair changes occur. A fresh target book
is required for current target displacement/work interpretation; a fresh peer
book is separately required for current source persistence. Stale views are
missing and may restore the same historical ID when fresh information arrives.
Hard status, invalid book/grid, clock regression or source-epoch boundaries
clear the affected leg's attribution; a broken target baseline cannot bridge a
later epoch. Historical diagnostic counters do not imply surviving predictive
state. These distinctions must be tested rather than replaced by a timer window.

## Maximum output budget: seven numeric and two categorical fields

Let s be the confirmed peer direction, Q its positive pre-execution attacked
touch quantity, X the confirmed same-cluster quantity, and mT the target mid
frozen at A. Let u_s(mT) be the positive absolute log-price displacement of a
physical five-tick move from mT in direction s on the full target share ladder.
It is a target event **unit**, not a fitted beta or a target destination.

| Proposed numeric field | Definition and role |
| --- | --- |
| `external_impulse_target_unit` | peer log-mid displacement from its frozen pre-move pair divided by u_s(mT); signed external percentage impulse, direction/magnitude context |
| `confirmation_capacity_fraction` | X/(Q+X), with wide exact quantity arithmetic; dimensionless confirmation strength, not execution-cost probability |
| `target_response_5ticks` | exact target physical mid displacement from mT divided by five; target direction/magnitude response |
| `target_known_work_imbalance` | (Wbuy-Wsell)/(Wbuy+Wsell) after A, where each closed known-side cluster contributes X/max(Qpre,X); direction information |
| `target_known_work_strength` | log1p(Wbuy+Wsell); accumulated observed demand over local capacity, magnitude/activity context |
| `source_persistence` | current peer net displacement from the frozen pre-move pair divided by the nonzero confirmed peer displacement, in the peer's own exact physical tick coordinate; return/refutation context |
| `mark_receive_age_300s` | (S-A)/300 seconds; continuous historical age, never a 300-second TTL |

The two categorical fields are `transfer_phase` (no qualified mark, awaiting
target response, followed, opposed, refuted) and `observation_phase` (warmup,
valid, peer stale, target stale, ambiguous attribution, tied, hard censored).
Final literals must fit FeatureValue's actual categorical capacity. A field's
role does not require it to predict both direction and magnitude.

The source impulse's percentage-to-target-unit comparison assumes no beta=1
forecast. A later beta-transfer hypothesis would be a new frozen experiment;
fitting a coefficient on later TRAIN days and backdating it into Jan19 is banned.
Five-tick price units must use the complete ladder, not current_tick*5 at a band
boundary. Quote prices are canonical grid observations; analytical log ratios
and estimated references are not rounded into fabricated quote identities.

No raw dollars, quantities, count levels, frequency or book thickness enter as
predictors. Quantity scale changes preserve fractions/work. Known no work gives
exact zero strength and a separately declared zero-imbalance convention; unknown
work gives NaN. A zero source displacement cannot confirm a mark, so persistence
never divides by a tiny rounded zero. Initial warmup is -inf; post-warmup absence
of a required reference is NaN. Numeric error must not cross these boundaries.
Tick/relative-percentage units preserve absolute five-tick base-rate information;
z-scoring away target volatility would remove a different magnitude question.
Normalization supplies comparable meaning, not proof that symbol sensitivities
are equal. Fixed-group results remain necessary under pooled CatBoost training.

## Synchronized marginal contrast and bounded support gate

Native code must expose a control from the SAME causal primitive book/trade
groups, history, scales, availability and original S. It retains the latest
qualified peer mark, but target response state belongs to the target's own last
qualified local quote/execution episode and does not reset/join at peer A. Its
target anchor/work age therefore follows local ownership. This control preserves
peer and target marginals and their current freshness without source-to-response
ID ownership. Export both arms in one pass. Never permute future values, whole
days, dates or symbols; never add artificial delays or generated clocks.

Before counts, freeze: two day jobs, both targets, the same all-16 carrier feeds,
source order/import/filter/warm-in, original periodic10s 09:10--13:00 keys, and
the existing 30s subset. Native all-book writer exports are diagnostic only.
Do not infer availability from all-book timestamps or substitute event samples
for the research cohort. Freeze binary/compiled-source/config/raw/BasicInfo
closure and export predictors separately from mark IDs, clocks and counts.

Proposed nonpredictive gate, to be accepted or changed BEFORE any pilot read:
each available fixed day/target cell requires >=5 distinct confirmed peer marks
with a qualified target baseline, >=5 distinct such mark IDs visible at fresh
existing30s origins before a nonzero target quote response, and >=5 origins with
a genuine ordered-versus-marginal ownership/phase difference. Overall require
>=5 marks in each direction and >=20 ownership contrasts. Preserve missing-cell,
ties, ambiguous/mixed, overflow, stale and source-boundary counts explicitly.
Failure rejects this exact mechanism/sampling combination, without rescue.
An absence of marks in these cells cannot establish absence in every market.

Integrity also requires a fresh original-control replay, exact parent five keys
and raw float64 mids, Alpha/metadata/sentinel validation, old/new existing-output
invariance, per-leg E-offset metamorphic tests, and one real receive-prefix
straddle with all carriers. A cost/support pass is not predictive evidence.

## Coherent twenty-role replacement context and eventual comparison

Context C is a NEW compact representation proposal, not the old 27-column core
with a smaller mask and not an automatic retention list. It has seventeen
numeric roles and three categorical roles. All required math/normalization is
native; a wrapper must own any new scale or state rather than divide arrays in
Python. The same C artifact is used by C, C+M and C+J arms.

| Role count | Proposed context and semantic units |
| --- | --- |
| 4 geometry/path | spread/5 ticks, signed 60s mid displacement/5, 60s physical total variation/5, absolute displacement/total variation efficiency with an explicit observed-flat state |
| 4 demand/observation | touch imbalance, depth-profile HHI difference, signed known execution over displayed capacity, known-side attribution share |
| 4 session/anchor | displacement from first valid continuous session mid/5, displacement from session POC/5, POC volume share, session VWAP price-deviation in the target five-tick percentage unit |
| 3 supply response | signed net displayed supply over prior depth, observed refill over prior depth, observed depletion over prior depth; preserve net snapshot interpretation |
| 2 activity/session | current60s positive volume divided by the mean of five disjoint preceding60s blocks, and fraction of the declared continuous session elapsed |
| 3 categories | ordinary share tick band, source continuous/status phase, local demand/observability phase |

Source neighbors for possible reuse, conditional on exact formula/state audits:
`BookStructure` publishes `imbalance_touch` and `book_profile_hhi_gap`;
`BookClockFlow` publishes `signed_trade_over_depth`, `known_trade_share`,
`net_supply_imbalance_over_depth`, `refill_over_depth`, `depletion_over_depth`;
`PriceMemoryState` publishes `pm_poc_offset_ticks` and `pm_poc_volume_share`;
`TradeVolumeRegime.vol_regime_rel_total*` uses a positive disjoint prior-volume
mean. `SessionVWAPDeviation.norm_vwap_dev` is a price fraction, not a five-tick
unit and not a spread unit: target-unit conversion needs an explicit native
producer and anchor semantics. Existing reusable outputs do not receive blanket
admission from their Alpha attributes. PriceMemory's six log raw-volume outputs
and its `+1`-quantity imbalance shrinkage are not automatically scale invariant.
BookClockFlow's denominator thresholds and net-supply attribution must be
audited at zero boundaries before reuse. C's single 60s context span is fixed
for role coverage; no span/parameter grid is proposed.

If support passes, preregister at most FOUR primary fits from a fresh baseline:
B (qualified historical full control), C, C+M (causal synchronized marginals),
C+J (joined H15). No added full-baseline complement or secondary fit matrix.
Freeze one judge first: either the currently validated unchanged judge, or an
independently completed/adopted methodology study followed by a new baseline.
All arms share sampling, pure-mid labels, cohorts, splits, CatBoost formulation,
training, metrics and acceptance rules. Freeze a planned C+J versus C+M contrast
before any score; J must add evidence beyond M/C and meet the baseline/full
representation direction AND magnitude criteria. A gain over C alone cannot
prove external ordered information. If all replacements fail, do not rehabilitate
raw B or claim success from support, pruning, normalization or repaired pipeline.

The primary 300s nomination and descriptive 60/120/180s effect/decay/base-rate
analysis must follow that contract; no horizon is selected by development gain.
No universe expansion is planned. Later pooled mapping/eligibility would require
a causal TRAIN-only structural rule frozen before labels, not best-peer OOS
selection. Future sealed OOS remains a separate final-proof requirement.

## Existing overlap and exact source boundaries

`CrossLeadLagResidual::align_time_return/publish` remeasures all legs over the
same `[now-span,now]` return interval. Its mapping has Symbol/Weight/Stability
and freshness; it has no cross-leg known execution input or source mark ID.
`CrossReturnContext` maintains held log mids and same-window return/volatility
contrasts. `CrossAssetPressure` combines contemporaneous depth pressure and
three return spans. `CrossMarketShockBreadth` summarizes absolute peer return,
directional coherence and peer-target gaps. H6's TRAIN covariance graph connects
10s sampled histories; it does not recover event-cadence confirmation/ownership.
If H15 merely reconstructs these same-window values, the marginal contrast
should falsify its claimed novelty.

Inspected implementation boundaries:

- `src/oms/modules/module.{h,cpp}` and `feature/feature_module.h`.
- `src/oms/modules/feature/order_flow/trade_flow/trade_side_inference.h`.
- `src/oms/modules/feature/experimental/quote_visit_acceptance/quote_visit_acceptance.{h,cpp}`.
- `src/oms/modules/feature/microstructure/cross_symbol/{cross_lead_lag_residual,cross_return_context,cross_asset_pressure}/`.
- `src/oms/modules/feature/{observable_book.h,order_flow/book_clock_flow/book_clock_flow.cpp}`.
- `src/oms/modules/feature/order_flow/trade_flow/trade_volume_regime/trade_volume_regime.cpp`.
- `src/oms/modules/feature/microstructure/price_memory/price_memory_state.cpp`.
- `src/oms/modules/feature/price_action/reference/session_vwap_deviation/session_vwap_deviation.cpp`.
- `src/oms/modules/writer/dataset_writer/dataset_writer.cpp`.
- `AstraResearch/astra/representation_research/{mechanisms,runner}.py` inspected only to identify existing H6/core overlap; no Python FE extension proposed.

These are proposal-time live sources, not a producer identity. Implementation
must freeze exact compiled copies and receipts before support/material proof.
