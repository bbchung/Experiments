# H12 source feasibility: single-stock-futures demand and cash assimilation

The current H12 study is rejected at source eligibility: **zero qualified stock/futures pairs**. This is a metadata and causal source-qualification result; it provides no predictive rejection of futures-to-cash information transmission. Mapping authority, adjusted-contract semantics, and prior-liquidity gates remain unchanged. No replay, feature implementation, model training, or FE comparison was started.

Inventory timestamp: **2026-09-30 11:33:01 UTC** (2026-09-30 19:33:01 Asia/Taipei). This record describes the local source snapshot observed then; file presence is not an immutable raw-content attestation.

## Motivation and hypothesis boundary

H12 would study whether an observed, confidently sided single-stock-futures demand episode remains unassimilated by its cash underlying, using source-contract queue capacity, signed demand, event availability, and subsequent cash progress. This is different from extending an own-stock imbalance window. Within-contract quantity ratios would cancel contract volume units; cross-market price and progress terms would require an authoritative underlying mapping and verified unadjusted deliverable/price relationship. Native trade-side inference would remain an observable proxy for execution direction, without identifying an informed participant.

The existing H2 basis study is relevant prior evidence, not a substitute for these source requirements. Its frozen `basis_mechanism.yaml` allowed `legacy_name_exact` mappings, used CDF/2330 with DHF/2317 and DVF/2454 comparators, and studied futures-target 1/5/30-second outcomes and raw/60-second valid-dwell basis. Its audit reported the revised economic formulation rejected on its supported development evidence and another arm support-inconclusive. Those results do not establish cash-target 60–300-second, five-tick endpoint predictability. This inventory did not reopen H2 datasets or models.

## Rules fixed before the inventory

The parent research task specified the existing 16-cash-symbol universe and the first 20 TRAIN calendar slots before source eligibility was inspected. These are task-level feasibility rules, not a new frozen FE evaluation contract.

A pair can qualify only if all of the following can be established causally from the available local sources:

1. Dated authoritative underlying identity; a Chinese-name match alone is descriptive and cannot qualify a pair.
2. Verified contract type, unadjusted deliverable and relevant quantity/price conversion. A name inferred standard contract or a caller-supplied point value cannot establish these facts.
3. Listed monthly-contract availability under the existing calendar roll convention, with strictly prior-day liquidity information sufficient to apply a declared front-contract rule. Current-day outcomes, file size, and mere raw-file existence cannot replace prior liquidity.
4. Available native input files on the original fixed dates. Missing symbols or dates are skipped, never replaced, downloaded, synthesized, or compensated by selecting a different date.

No eligibility decision used OOS results, forward labels, model scores, or predictive ranking. Raw market-data files were checked by path existence/stat only; their contents were not read or hashed. Small BasicInfo CSVs, the trading calendar, current implementation sources, and the existing H2 protocol/audit summary were read.

The fixed cash universe is:

`2330, 2317, 2454, 2308, 2382, 3231, 2603, 2609, 2615, 3481, 2409, 2344, 2337, 2481, 3037, 3711`.

The fixed TRAIN slots are:

`20260119, 20260120, 20260121, 20260122, 20260123, 20260126, 20260127, 20260128, 20260129, 20260130, 20260202, 20260203, 20260204, 20260205, 20260206, 20260209, 20260210, 20260211, 20260223, 20260224`.

## Observed source support

All 20 TAIFEX BasicInfo files exist, totaling **1,859,647 bytes**. None provides a nonempty `underlying_symbol` for any row; the column itself is absent in all 20. All files provide `symbol, exchange, name, unit, limit_up, limit_down, day_trade, state`; 16 provide `underlying` and `maturity_date` columns, but those columns do not establish the stock-futures mapping and maturity for the relevant observed contracts. No inspected file supplies prior traded volume, a usable liquidity statistic, or adjusted-deliverable/share-multiplier fields.

The configured local `/mnt/data0/stats/taifex` directory is absent. The inspected metadata and native contract-loading paths do not expose an authoritative prior-liquidity input for this rule. This is a limitation of the inspected source set, not a claim that no such table can exist elsewhere.

`astra.instruments.stock_futures` produces descriptive legacy name mappings for 17 products across the 16 cash symbols. Every relationship in this census is legacy-derived. Its inferred 2,000-share standard-contract mapping is insufficient evidence that each observed contract is unadjusted. The native `TaifexInfo` reader copies a dated `underlying_symbol` only when present, checks BasicInfo `unit == 1`, and obtains its trading quantity multiplier from configured `PointValue`; that configuration cannot supply missing source authority.

The table below reports legacy candidates, not eligible pairs. “Prior raw” is existence of the same candidate contract's file on the preceding calendar trading day; it is not measured liquidity.

| Cash symbol | Legacy product(s) | Product/day cells | Authoritative mappings | Front raw present | Prior raw present | Qualified product/day cells |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2330 | CDF | 20 | 0 | 19 | 19 | 0 |
| 2317 | DHF | 20 | 0 | 19 | 19 | 0 |
| 2454 | DVF | 20 | 0 | 19 | 19 | 0 |
| 2308 | FRF | 20 | 0 | 19 | 19 | 0 |
| 2382 | DKF | 20 | 0 | 19 | 19 | 0 |
| 3231 | DXF | 20 | 0 | 19 | 19 | 0 |
| 2603 | CZF | 20 | 0 | 19 | 19 | 0 |
| 2609 | DAF | 20 | 0 | 19 | 19 | 0 |
| 2615 | QXF | 20 | 0 | 19 | 19 | 0 |
| 3481 | DQF | 20 | 0 | 19 | 19 | 0 |
| 2409 | CHF | 20 | 0 | 19 | 19 | 0 |
| 2344 | FZF | 20 | 0 | 19 | 19 | 0 |
| 2337 | DIF | 20 | 0 | 19 | 19 | 0 |
| 2481 | GYF | 20 | 0 | 19 | 19 | 0 |
| 3037 | IRF, IR1 | 40 | 0 | 38 | 38 | 0 |
| 3711 | OZF | 20 | 0 | 19 | 19 | 0 |
| Total | 17 products | 340 | 0 | 323 | 323 | 0 |

Cash input files exist for **320/320 symbol/day cells**. Descriptive near-month futures files exist for **323/340 product/day cells**, all with native-preferred `.bin.zst` availability. All 17 missing front-contract cells are on **20260202**; that date stays missing. The source roots checked are `/mnt/data0/marketdata/tse/kgi/stock/<symbol>/<day>.bin.zst` and `/mnt/data0/marketdata/taifex/kgi/futures/<contract>/<day>.bin.zst`.

The descriptive near-month preview follows `TaifexSymbolResolve::resolve_monthly_rank`: rank listed months and roll the expiring calendar-month delivery on the third Wednesday or its preceding calendar trading day. Within these slots, the early-roll dates are 20260120 and 20260211. This only explains the previewed A6/B6/C6 monthly files; it does not implement or validate prior-liquidity selection. For 3037, both IRF and IR1 match the same legacy underlying name; their deliverable/adjustment relationship remains unverified. Neither product is selected by guessing its name or raw availability.

## Decision and next study

The authoritative-identity gate fails for every candidate before raw event support or predictive testing. The unadjusted-contract and prior-liquidity requirements are independently unresolved. Therefore, **H12 is stopped for this local source snapshot without relaxing any gate**. No pair, date, source format, or alternate contract is substituted to rescue the study.

H11 remains a source-audited proposal pending the H10 results. A possible distinct source is cross-stock transmission of confidently observed, physically normalized known-demand state, rather than another graph of contemporaneous mid covariance. A sampled 10-second panel alone cannot establish instantaneous leadership. Any later H11 pilot must first establish causal event availability, qualified TRAIN-only directed support, and novelty against native CrossReturn/CrossAsset families; formal baseline/candidate comparisons must retain the same frozen judge. This document authorizes no H11 implementation or experiment.

## Hash-bound metadata and source references

The 60 BasicInfo files below comprise the TAIFEX futures plus TSE/OTC stock metadata on exactly the 20 fixed TRAIN slots. Their combined byte size is **3,496,726**. Hashing was limited to these small metadata files and the listed source/reference files; no raw market-data contents, labels, model artifacts, or OOS datasets were read for the census.

Metadata path convention: `/mnt/data0/contract/<kind>/<day>.csv`. The individual raw market-data existence observations are intentionally not claimed as content-hash provenance.

| Source/reference path | SHA-256 at inventory |
| --- | --- |
| `/home/bb/workspace/coco_dev/AstraResearch/experiments/information_state_20260930/profile-v3.yaml` | `218ca081811ec7dc1eea62e4006f5e6add545a0288989af2ea6db2c8f7fbdea4` |
| `/mnt/data0/info/tw_trading_calendar.yaml` | `8b1b6c57bbc79137516aca57a61ede352523e41ba800010bb6b6b011f2bdcf78` |
| `/home/bb/workspace/coco_dev/AstraResearch/astra/instruments.py` | `519d9372d424b28f41f80912921c69d73a0f84895d0af617b0cf1a77f4fe6551` |
| `/home/bb/workspace/coco_dev/src/oms/api/taifex_symbol_resolve.cpp` | `628721eba6fa15c076c56697dd02030fbe320c64d6b1c54392f1719eaba728e9` |
| `/home/bb/workspace/coco_dev/src/oms/modules/tw/taifex_info/taifex_info.cpp` | `738267b20e2b9f8c0e11daac36ac417f101a11d57921161d9459c0ae21c0cd90` |
| `/home/bb/workspace/coco_dev/AstraResearch/experiments/e2e_20260926/basis_mechanism.yaml` | `6081eb41f3a1efb8ae46fe79666038b3767280cd1b9689a6d1556f7cf184722b` |
| `/home/bb/workspace/coco_dev/build/end_to_end_research_20260926/h2-economic-audit.md` | `dcbebe9206affd72121e35051d0a5a32315c60e580f2b3bb77c3c97418775c84` |

| Day | Kind | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| 20260119 | taifex/futures | 90524 | `d73f999c2d8d63f20efff53411c5426dc8d1bb7faeb0ddff68d49a91539405d7` |
| 20260119 | tse/stock | 45602 | `fae4683729abf9aeaaf17633062b6dba62bc3821d71640a83dce14131f821549` |
| 20260119 | otc/stock | 36031 | `2d57bfc5e6987934e18f24c58945682916935abd9ea9d890d68123abf40a879b` |
| 20260120 | taifex/futures | 90490 | `e8b660f5179af2d39d692eb5b99472de264d24c61653f359f7d2a77d744a8d0e` |
| 20260120 | tse/stock | 45644 | `704c26e2f7785f46fba8ed43a0bdc308f4b26d7a17c1316c6babd13603e147d8` |
| 20260120 | otc/stock | 36027 | `54204a03d5bd69bed2b692c67e9a2e02bc67d75971d664a9b6d9e6f7882b6207` |
| 20260121 | taifex/futures | 87086 | `f8842e43c96a8e90dca9ad87667dcade2bf441c75a10aaf0f785557c6d8f5f8d` |
| 20260121 | tse/stock | 45631 | `b490ea8f5efd6604ae95a2b469d4cf4472a4cfa871ca3a2554775245e9b5da01` |
| 20260121 | otc/stock | 36017 | `2f9dd9af16774b6d0c10cf7f2c7c7e97e28bc70d7a9f901046d2afdf8d8bd3c7` |
| 20260122 | taifex/futures | 90540 | `d65752a2997fa266688164cffd93aef63f52b36f3909db5a9e3447df2edad95e` |
| 20260122 | tse/stock | 45659 | `9bd3b1d243e8bce394c3e408cfbe81a8e4c00990a01a4be217f3bb1ca3f85339` |
| 20260122 | otc/stock | 36196 | `458a5de525416f2a7c9f05582c5aef872b5adc73d703af8d2690dea6fd1220fe` |
| 20260123 | taifex/futures | 90500 | `752068039624f68d5e7fb032310ddab46300501799eb7df1a5191e5511d93b69` |
| 20260123 | tse/stock | 45592 | `44e58cb193e00ac8153ff7e5f641539e2c2dd46fc12b9f06a8cf1556285a74da` |
| 20260123 | otc/stock | 36140 | `94919837de92364f73241203271fd8afe41a73c70ab3432a60fc1e525b0db289` |
| 20260126 | taifex/futures | 90668 | `21560e5882bdd8f53da1f5001a427693400cfa63fe5894eb4a6befcd678e0962` |
| 20260126 | tse/stock | 45605 | `2a67c46fd9e7dd8863ada0594553e509c56fdf346f59e095a2bd283d2741b84b` |
| 20260126 | otc/stock | 36104 | `9a39895742fae9a052421d65a0413a2922a1ce00c409508e756d1afde20bd2dc` |
| 20260127 | taifex/futures | 90714 | `b708b6432ec8cfc21e062cf075b1a713ef2364acfeff0fac4298997a76a877eb` |
| 20260127 | tse/stock | 45579 | `b079ac82a99adf44f7e0e2a839db7f624cc78bef666f1781aba5c3b108626e9c` |
| 20260127 | otc/stock | 36068 | `8d3b71050c178880331ae49f33666ba26a80a568ba5042f598bf222091dfa0a3` |
| 20260128 | taifex/futures | 90744 | `c08c1b1f41b8c1e434e380b52e113ded90075b4908864d3df526213083175073` |
| 20260128 | tse/stock | 45575 | `0c686f7fd08462500dc15543a0da2b7c36d1eb43a46b3fc9a0e2fbf995db4932` |
| 20260128 | otc/stock | 36093 | `43a74dcd4761f4fb1c2098f2e856ed66eeb0710ff9d9f36e2d3e8aa7c0d6ddb7` |
| 20260129 | taifex/futures | 90659 | `103ccebb887895deebe6a999e3e771c86a3367fc6a11b85558771a77ef88a5e6` |
| 20260129 | tse/stock | 45679 | `2115b84a1e52f63ed614d54dd8c18a6fb2aa2e3ab4b9624c8552db9bf8511e80` |
| 20260129 | otc/stock | 36165 | `e811f3b5607ced5294f737374f1f3675472b1161e1176d5d829778382ffc75d3` |
| 20260130 | taifex/futures | 90744 | `298d8289e0816a9ee9e922f85f36ca8e8cb397f015a946e50b513fd9c783a3dc` |
| 20260130 | tse/stock | 45676 | `e10f650e287632cd94b933b70de605aade583b728d1c625a9e6b3b7224bcc747` |
| 20260130 | otc/stock | 36131 | `278bf791b1d6a9b54cb3c5aaa50f76d624c2fc882e0d6ab793f2405e24567841` |
| 20260202 | taifex/futures | 134454 | `10b45af46b4d03649e759a07fd8a4c78b1795237bb2442aced1f6170d6795e88` |
| 20260202 | tse/stock | 45675 | `c9f2be7631440535093231ba6648d7574094187b2fdff43b377e30960f33ae72` |
| 20260202 | otc/stock | 36147 | `1edbb65325246c18e1d479d1b3273ebd5c99a6ac18fdd9b7a4b1003962f95f3d` |
| 20260203 | taifex/futures | 88851 | `e45c348c6a196a11a66e82f333002d5237a84a4f939969788ea082028ea56019` |
| 20260203 | tse/stock | 45711 | `212a0a1db1d15d6f391f242f2311214e9034d445783c44c6862b96317470c47b` |
| 20260203 | otc/stock | 36279 | `fd1e7f1f78f1fb27ac9a150b595ad68f059968639a3fab64ceb73ca825f38475` |
| 20260204 | taifex/futures | 92365 | `4f665bd840644a4d287fbfa0ca3693ae97236d2690c48744673f742a18629185` |
| 20260204 | tse/stock | 45711 | `532a74d03c250f280e206b6efac2a114b350bf4e75381648fe917c46885005ad` |
| 20260204 | otc/stock | 36239 | `0a6e6262677cd5d93324bcb1714012060a93c07f86b1789fd4eecb6440f0760e` |
| 20260205 | taifex/futures | 89054 | `c9fa11decdb3934737c92790a581160d92a8dd0e6e912772c881c6e5a2e3cbbf` |
| 20260205 | tse/stock | 45778 | `cf335c3a1e1f3c3ce56ea2487d2400ef29a9d45f6b714595ab0876d2688ad13b` |
| 20260205 | otc/stock | 36263 | `800c9a92262c4c04dd125b6ffb5374e524dc09c70fba2bde31580c6e8d7fd9b4` |
| 20260206 | taifex/futures | 89014 | `137e0e481663d9a208ee000615be09eea82791de65bc7ae52e6854ed49215a10` |
| 20260206 | tse/stock | 45803 | `f99176591640557a7e50c6bc8b4acd202dd818c7776a3f1fbe05d633eb6be366` |
| 20260206 | otc/stock | 36273 | `c08adfdde6716a61a55fd4ef741b94f54104188a9320e812d1732250a4937fe2` |
| 20260209 | taifex/futures | 92650 | `767e2bae4a3e8db3e24b93f59e7d6b21ba53cada74f84bb0f6ea7cae6439ad61` |
| 20260209 | tse/stock | 45741 | `ac8f47090fc52be5206ad4fbd9406a952ca875b5a81ff124c08d2a5717b76f2b` |
| 20260209 | otc/stock | 36248 | `e384b2d1e2c97a0d3d701988cb79d435bd5642734fe31fa4dffd38d8a1a56fd4` |
| 20260210 | taifex/futures | 92647 | `9f6767b98b0174cada160ac05661a554f73847ef3df87d7a670a65ba5a019d60` |
| 20260210 | tse/stock | 45749 | `6bcc36aae60c79c668795aec9e456cbd72797d984ee33710a656877ddcf4fd11` |
| 20260210 | otc/stock | 36250 | `e8a25fd786716eaaee23f89570a2c8ae42bcf9d89ad4118fe0f4815ad9030f99` |
| 20260211 | taifex/futures | 92702 | `e4deafe8d197601c88db17dfa20a24f4a7bf06bfeda226e51eaaa21b56538347` |
| 20260211 | tse/stock | 45748 | `c16b67bc71cb4ce278f5ec18f5cfe20d7ffaa6dcbe67603ce50efc2493b67e92` |
| 20260211 | otc/stock | 36273 | `37d5907338e33ddc0a6d6566fb0cb951e8ecd0e1ae3fa11465ec99b8e8fe68bf` |
| 20260223 | taifex/futures | 92669 | `5127772840057148eb35a12f05971cec983e5be95fa6768f4406ef7b31e13ec0` |
| 20260223 | tse/stock | 45734 | `685c7f15d6466649460cb34b9e8f9bae9355054293ac52b15ba640f8e42be512` |
| 20260223 | otc/stock | 36273 | `5bc4ac21106d81ccb49fd4b2a1c452afc456bec9f82e2fcb31cd11f90d082fb6` |
| 20260224 | taifex/futures | 92572 | `c65ab59b6535d5a50f36ab9a276f7ef667dbaa572a56a211edfbf78fa612fb65` |
| 20260224 | tse/stock | 45722 | `80a07224c995989e0f3a44d6fa508716ba88975cf2fca5cf5285cd4e77564d3e` |
| 20260224 | otc/stock | 36248 | `a6a0fde624b514c35b2fed9d038b07ddb2eda7baea749cbde5ddd2747dca8616` |
