# H13 native source-support experiment

This experiment implements quote-led shared-visit bilateral testing directly
as `QuoteVisitAcceptance` under `src/oms/modules/feature/experimental/`.
The actual native writer (`DatasetWriter`, typed Parquet output) materializes
its values. Python only reads these outputs and evaluates support; the earlier
target-only Python proxy is preserved as a diagnostic and cannot qualify this
native experiment or supply predictive features.

The historical landmark hypothesis and its normalization follow the fixed
H13 source plan: both touches move in the same direction without an observed
positive print since the previous valid book or in the whole formation E
cluster; later confidence-one exact-bid SELL and exact-ask BUY must belong to
the same visit in distinct later closed E clusters. A received pair departure
irreversibly suppresses that visit's live view, including a return within the
same still-open cluster. Closing the cluster preserves valid earlier pending
evidence. A new closed admission owns a new visit. First completion freezes
the historical mark and work. Ordinary ambiguity and book gaps clear current
credit; they do not erase completed history. Hard source-epoch breaks do.

Native normalization and `TwseFilter` determine delivered books and trades;
the conservative raw Python proxy did not establish equivalence with these
callbacks. The source contract is therefore the compiled native module and
its actual dependencies, not agreement with Python feature calculations.
Receive and exchange clocks remain separately ordered. No E<=R or E<=origin
assumption, timer closure, EOF flush or floating epsilon is introduced.

The fixed TRAIN cells remain 20260119/2308, 20260119/2317,
20260120/2308, 20260120/2317. Retain all sixteen already declared clock
carriers. Missing source/day/symbol metadata is skipped without replacement.
Use the existing native 10s periodic writer contract, 09:10 through 13:00;
match its original 1381 three-field native keys and float64 origin mids
exactly, then select the existing 461 keys divisible by30s. No new origin
grid, label read, event-based replacement sampling or availability filtering.
The all-book writer is diagnostic and cannot replace prediction origins.

Before actual replay, freeze the three YAML configs (two shared daily jobs and one relocation control), writer/export schemas,
native module and unit-test source closure, compiled binary/runtime identity,
raw BIN/BasicInfo input identities and original native keys/mids. The profile
binds all five Alpha and fifteen Info output names. The buffer holds at most
256 delivered book or positive-trade callbacks per open exchange cluster;
the 257th poisons the whole pending cluster, clears current visit attribution,
and preserves any previously completed historical landmark. There is no
partial completion, EOF flush or timer closure.
Native unit tests must cover leave/return, pending/closed ownership, formation
veto, independent clocks, soft/hard censors, stale view, normalization and EOF.
An actual receive-prefix replay and preservation of the relocated H10 writer
outputs are required before any formal predictive comparison.

The support rule is unchanged: at least one fixed cell is observed; every
present cell has at least5 session completions and5 distinct completed
landmarks seen at an original origin with0<receive age<=300s; in aggregate
at least5 up and5 down completions and20 provisional/unilateral origins.
Missing cells are reported. Repeated appearances of one landmark count once.
The 300s support-age boundary is neither a TTL nor a feature window.
Session completion counts come from native Info counters with receive-time
boundary checks. Neither clocks nor counters enter a future CatBoost model.

Failed native support rejects this exact hypothesis/sampling recipe without
changing gates, lifetime, symbols, dates or origins. Passing support still
requires native causal prefix/parity before a separately frozen FE comparison
from a fresh baseline. No model is fitted in this support experiment and no
previous FE result is rejudged. The previously exposed evaluation dates cannot
establish final OOS.
