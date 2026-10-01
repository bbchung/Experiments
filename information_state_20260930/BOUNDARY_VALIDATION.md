# Native endpoint boundary audit before V2 freeze

The published stock projection has no observed class disagreement between the
original inclusive native `>= +5` / `<= -5` tick rules and an audit-only physical
millitick-grid reference. The V1/V2 labels and scientific judge remain unchanged.

The census read only the four `mid_return_ticks[Hs]` double columns from each
parent `rows.parquet`. It checked all 14,161,120 finite endpoint values, including
114,525 values exactly at a five-tick boundary. There were zero nonexact values
within `1e-8` ticks of either boundary, zero threshold class changes under
`round(move * 1000) / 1000`, and zero nonzero values whose grid reference was zero.
The neighborhood is only a search screen; it is not an evaluation epsilon or an
authorized correction. The largest scaled grid residual was `1.8189894035458565e-10`
milliticks, or approximately `1.82e-13` ticks.

| Role | Horizon | Finite labels | Exactly +5 | Exactly -5 | Class disagreements |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 60 | 2,253,792 | 3,891 | 3,191 | 0 |
| train | 120 | 2,253,792 | 7,896 | 7,452 | 0 |
| train | 180 | 2,253,792 | 11,481 | 10,687 | 0 |
| train | 300 | 2,253,792 | 16,231 | 16,052 | 0 |
| tune | 60 | 623,776 | 1,218 | 926 | 0 |
| tune | 120 | 623,680 | 2,576 | 2,321 | 0 |
| tune | 180 | 623,584 | 3,591 | 3,561 | 0 |
| tune | 300 | 623,392 | 5,093 | 5,401 | 0 |
| forward | 60 | 662,880 | 582 | 282 | 0 |
| forward | 120 | 662,880 | 1,332 | 890 | 0 |
| forward | 180 | 662,880 | 2,107 | 1,467 | 0 |
| forward | 300 | 662,880 | 3,382 | 2,915 | 0 |

The original writer configuration binds both endpoint-return and tail-indicator
labels to `CurrentBook.0.book_mid_ticks.0@symbol`. `CurrentBook::on_book` obtains
that value through `TickCalculator::to_tickf(book.mid())`. `ReturnLabel::value`
uses the direct double subtraction `y1 - y0`; the tail rule compares that same
subtraction to the inclusive integer threshold. The projection casts feature
matrix values to float32 but copies origin metadata and labels as double.

The terminal queue expires only when `deadline < offset`. An arriving feature
update expires earlier deadlines before replacing `last_y`, so an endpoint uses
the latest observed Y at or before the horizon, including an exact-deadline
update. Invalidation records an unavailable endpoint state. If no Y update
occurred after the origin, the source holds the origin Y. This is a source and
published-artifact audit; it does not replay raw quotes or prove that every
hypothetical future symbol/day is free of floating-point boundary problems.

Reproduce with:

```bash
python3.13 runs/information_state_20260930/v2/shared/native-label-boundary-census.py
```

The adjacent `native-label-boundary-census.json` contains the complete per-role,
per-horizon counts, exact-boundary examples, published dataset identities,
verified row-file hashes, script hash and current source/configuration hashes.
Both new shared files are included in V2's frozen input closure. No raw replay,
feature matrix loading, training, V1 write, or label correction was performed.
