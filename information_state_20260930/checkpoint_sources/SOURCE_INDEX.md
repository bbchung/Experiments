# Native research helper source archive

This checkpoint preserves exact bytes of authored Python orchestration, native output validators, tests and root pipelines that currently live under ignored `AstraResearch/runs/`. It excludes generated outputs, copied engine sources, binaries, datasets, models and runtime libraries. No frozen original was moved or edited.

The archive stores each helper directory directly beneath `checkpoint_sources/`; its canonical path adds the prefix `AstraResearch/runs/information_state_20260930/`. These copies are review and recovery sources, not a relocated execution entrypoint. Canonical executable paths remain `AstraResearch/<path below>` because frozen closures bind those paths and scripts derive their working paths from `__file__`. If recovering into a new checkout, restore only absent files to the canonical paths; compare existing files with the SHA-256 below and never overwrite a different version. Historical frozen run manifests and data artifacts remain local and are required to reproduce evidence.

`native-h16-auction-cycle-v3` is an unexecuted, unfrozen prospective draft. `native-h16-receive-prefix` still targets the old H16 output schema and must be adapted before any raw V3 proof. Earlier prototype and rejected source-gate tools are retained for traceability; their presence does not authorize predictive admission or rejudging completed comparisons.

All 49 source copies were byte-verified at checkpoint preparation.

| Canonical path relative to AstraResearch | Bytes | SHA-256 |
|---|---:|---|
| `runs/information_state_20260930/h10-pilot/census.py` | 16630 | `9c30f653612ea770c88eb2b85127334f93637457a431d1f3560b34b1eb1c6ae2` |
| `runs/information_state_20260930/h10-pilot/eligibility.py` | 10585 | `6cbf5bd1b8f2918ca2c09453520ec0ee14d093e063143a5e393dd11c858c60cd` |
| `runs/information_state_20260930/h10-pilot/support.py` | 17039 | `65099db4b302b237a41e571a131af5fa9443f452a71031bcccfdc5868f5c43cf` |
| `runs/information_state_20260930/h10-pilot/support_distinct_price.py` | 17853 | `594b16117610a42bb712118ed7ac36c8ed63ee10df890a0d54761859a539c930` |
| `runs/information_state_20260930/native-h10/prepare.py` | 10868 | `4846ba6cdfaf9fdcfb44f21db8cc4053644c47aeecef4a05e9f868e11b84b338` |
| `runs/information_state_20260930/native-h10/prepare_ablation.py` | 11676 | `33acb54a134ce8130d811a5389c47c7c52470261816babee01eda7ecc2033199` |
| `runs/information_state_20260930/native-h10-ordered-reset/execute_full.py` | 9460 | `591f1277f82fa9975f46463a74096083c4dadbcceffdd950ac52bb9d7ca36b7e` |
| `runs/information_state_20260930/native-h10-ordered-reset/execute_original.py` | 2002 | `22dc1862a7d92316b21d6243b55a27fa6992137a19863fcf34ea6b1b4234c528` |
| `runs/information_state_20260930/native-h10-ordered-reset/execute_pilot.py` | 2749 | `815d15071e2f10e40f211df06e83fe311d5d9cb7dd2eddd5d9e6448a9ec04a25` |
| `runs/information_state_20260930/native-h10-ordered-reset/freeze_component_method.py` | 8700 | `76402329578084073747c1eb809ff9ab3ff656d497cf156f99705363af6b81ba` |
| `runs/information_state_20260930/native-h10-ordered-reset/independent_validate.py` | 13606 | `e25bfa690723efd9c0c2439a8a530ffc64316606168b72ade014c26b3479f3bb` |
| `runs/information_state_20260930/native-h10-ordered-reset/prepare_full.py` | 9858 | `196d455a0458dde4db18ec820c7732992b54dd997b0b38e8ae35d5f666178941` |
| `runs/information_state_20260930/native-h10-ordered-reset/prepare_receive_prefix.py` | 11452 | `de2b5a74598b34c8c90aa71194856194c8b3d0ee87e2f748e799ac10517c719f` |
| `runs/information_state_20260930/native-h10-ordered-reset/run_v3_pipeline.py` | 1045 | `0530730693b67f6456a1c985aacc843baad69daf04236ec523f860e0a43ad750` |
| `runs/information_state_20260930/native-h10-ordered-reset/snapshot_compiled_source.py` | 1415 | `c0d101ab66422cdc6707a283049d3b9fa66cd5319720be0e40a37dcc4b2f2fac` |
| `runs/information_state_20260930/native-h10-ordered-reset/validate_qualified_gate.py` | 3705 | `7ab5a7ccf3258053b23bbe79d1536489d536ec3826b567d6dc2d7d4310a3f571` |
| `runs/information_state_20260930/native-h10-ordered-reset/validate_receive_prefix.py` | 8201 | `fcb9c92e2366033b0987bf11738d798e913d0e38063dc11f7a8329bd43ba66bf` |
| `runs/information_state_20260930/native-h10-ordered-reset/validate_receive_prefix_snapshot.py` | 9729 | `0cf66db49eee4d630ec546e52895decf2faa44d17b9265cf402eec2a265e5652` |
| `runs/information_state_20260930/native-h13-quote-visit/evaluate_native_support.py` | 28014 | `fba3bb1b2e5e40a40f660cc5fde695285c262e96ab4f87a1783d7b85e849032e` |
| `runs/information_state_20260930/native-h13-quote-visit/prepare_native.py` | 27797 | `e1b0296cccc2864546ff9578e61ec4e4ca63d3f7667aa81dc844c685b87c44f6` |
| `runs/information_state_20260930/native-h13-quote-visit/test_evaluate_native_support.py` | 13609 | `c0ee1139bc0b9658eacb51efcc6c69667ae587160902dfc72bce195064fc7b48` |
| `runs/information_state_20260930/native-h15-full/prepare_full.py` | 36455 | `26631b1515ae0f837aaf3c12e7a08b313f3a18ab65a5c2c38862628462a24e35` |
| `runs/information_state_20260930/native-h15-mapping-support/coverage.py` | 34765 | `cf3ecf4cbf1dba9a6f47d0bf1c3f72281e48b151e74ad604a979e40213065277` |
| `runs/information_state_20260930/native-h15-mapping-support/run_mapping_pipeline.py` | 3696 | `2c5b34966f5e0f4c753188d5b54f60cdec979d71651bf7bf0b316c5f6d1759ef` |
| `runs/information_state_20260930/native-h15-mapping-support/test_coverage.py` | 16337 | `562f3b781e599ee086cf035c26cb20c6d2576415e15aba2118de6db3c9d46ff0` |
| `runs/information_state_20260930/native-h15-peer-trade/prepare_native.py` | 32312 | `e42f270559e7e212f8b88185db54358b4280bdb85b811eff59cc5a3c87804987` |
| `runs/information_state_20260930/native-h15-peer-trade-v2/evaluate_native_support.py` | 31568 | `11193e5b70d2dcb15b711e1f742210dda7fc3c8787bcd5964719e9ecef50b1cb` |
| `runs/information_state_20260930/native-h15-peer-trade-v2/prepare_native.py` | 42652 | `68478054e55e48646eb241e57ea18100ef10fd1b73f82ff13a1358a0a0610f77` |
| `runs/information_state_20260930/native-h15-peer-trade-v2/run_pilot_pipeline.py` | 5177 | `0864dd2ad295315624cb00b905b267fa4f1e1e3471672a2d683baf221ed9ad41` |
| `runs/information_state_20260930/native-h15-peer-trade-v2/test_evaluate_native_support.py` | 15673 | `7ed08c37fe103d00082233358f0589ab2037d5dda046a3e8d54675241800c4ad` |
| `runs/information_state_20260930/native-h15-peer-trade-v2/test_prepare_native.py` | 11687 | `adca499e0bc8a26b17eefd3782649cde4d6969f52cb76b6ff15d596362326605` |
| `runs/information_state_20260930/native-h15-receive-prefix/prepare_receive_prefix.py` | 30828 | `2426d66b474886d9e594e110a82b161b3065b9567f947df84f97fcb5126b0b06` |
| `runs/information_state_20260930/native-h15-receive-prefix/run_prefix_pipeline.py` | 2934 | `230563297a9a95f806e85d983bfca3dda694a57be7e92203b2078fa675c43a44` |
| `runs/information_state_20260930/native-h15-receive-prefix/test_receive_prefix.py` | 13650 | `d1360799f0347dc6d95da136a2120e8c5400c0fda460c93f42a1a1db1bdaf060` |
| `runs/information_state_20260930/native-h15-receive-prefix-v2/run_comparison_pipeline.py` | 847 | `b461de67cc6802c9d6a8805ba5f1b218fd06d21d2816501e12547e6b222b2134` |
| `runs/information_state_20260930/native-h15-receive-prefix-v2/test_prefix.py` | 15178 | `26a02dd392f6ec2b1ece430af31914f6e98063f6279379c20a9602e1995a16c3` |
| `runs/information_state_20260930/native-h15-receive-prefix-v2/validate_receive_prefix.py` | 15967 | `e0bbbb5e52429cab640bf71e16c1674da7f344234a3d5387869fc6782bbb819f` |
| `runs/information_state_20260930/native-h16-auction-cycle/native_pilot.py` | 55761 | `35730530f98b3536dbe4340935608a418f67f90b11e699e58fa18738f6236a83` |
| `runs/information_state_20260930/native-h16-auction-cycle/run_native_pipeline.py` | 2717 | `3af53570a8898cb3a12bffc0f9c7713dcaa33e8d7beb498246f05610dd4c38dc` |
| `runs/information_state_20260930/native-h16-auction-cycle/test_native_pilot.py` | 32033 | `eb084200d31bbd356e8ad122447d1818ccb8c656038aae27da8453b52a34181b` |
| `runs/information_state_20260930/native-h16-auction-cycle-v2/repair_pipeline.py` | 5142 | `0e645e9d6ddaba9d696e17e58e8ce30816a4b1b6a3fcbf87e55be208b35fac10` |
| `runs/information_state_20260930/native-h16-auction-cycle-v2/test_repair.py` | 2184 | `09d72059f3a91a2abbd1030496a1c5650e895f2a6d22e34f241b3447cface557` |
| `runs/information_state_20260930/native-h16-auction-cycle-v3/native_pilot.py` | 62050 | `e9af5ad0f400a25154fad3cb7a4ad1a63897f51809a8257e644f745e8ef1f6ae` |
| `runs/information_state_20260930/native-h16-auction-cycle-v3/run_native_pipeline.py` | 2857 | `78ac4ee1dfbf470a56c4ed72d21c02e299743d8ca66b5a2ed86dd2a3f9ef725d` |
| `runs/information_state_20260930/native-h16-auction-cycle-v3/test_native_pilot.py` | 38148 | `2ba4b18050d63c492abb2721056ae8285b7f5fe391fca402f25d3bde9c6672cc` |
| `runs/information_state_20260930/native-h16-receive-prefix/receive_prefix.py` | 36972 | `e545fe19e0966c4a77bf1b753f001f12679606e1dbb3339cdb9634e47c5423d8` |
| `runs/information_state_20260930/native-h16-receive-prefix/test_receive_prefix.py` | 16659 | `ea141508599ebac649979259b8f7c723532d5ab4ce744eb3dd867081e88cc850` |
| `runs/information_state_20260930/native-h16-support-census/census.py` | 5033 | `66223cf7645619753456a8a72d0995a847db092a265e83ffcbd46067c4385075` |
| `runs/information_state_20260930/native-h16-support-census/run_census_pipeline.py` | 973 | `cd472a7a7b69296640781441847d12d1bc9333d799890e2f192c1c4cfa44b794` |
