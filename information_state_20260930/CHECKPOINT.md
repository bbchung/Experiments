# FE research checkpoint — 2026-10-01

目前尚未找到通過固定判準的 predictive representation，最終 Goal 未完成。依使用者要求，整理本 checkpoint 後暫停研究，等待檢視與明確 resume；本輪不執行 commit。

## 目前可以下的結論

- V1 的跨股票關係、過往 session prior、price occupation representation，以及 V2 的 marked renewal／完整 context complement，均未通過各自事前 frozen 的比較。
- V3 原生 C++ demand → repair → touch renewal hypothesis 有真實事件支援，但加入完整 baseline 後未通過 predictive admission；ordered-versus-reset ablation 也未獲支持。
- V5 使用既有原生 factor 的 C15／C18 精簡、具語義 normalization 的 representation，明確失去 magnitude context：development large-move AP 比同輪 baseline 下降 7.87%／8.38%，joint AP 下降 2.25%／1.89%。略高的 pooled direction AUC 無法補足，而且 equal-symbol/day direction 表現下降。這個結果不證明每個既有 feature 都應保留，也不證明增加 features 就是解法。
- C++ FE 已有 300s path 與較長歷史，所以不能宣稱它完全只有短 window。V1 的固定 development cohort 中，60s → 300s 的 large-move base rate 由 0.295% 升到 2.542%，conditional direction AUC 由 0.711151 降到 0.583612。這支持繼續研究不同 horizon 的資訊需求，尚未證明更長 horizon 或任何新 representation 有優勢。
- Methodology 與 conditional-direction training studies 均未證明符合採用標準；未更改正式 predictive judge，也未重新裁判已失敗的 FE 比較。

完整 hypotheses、數字、frozen identities、native source gates 與限制見 [CHECKPOINT_REVIEW.md](CHECKPOINT_REVIEW.md)，各輪完整分析見 `RESULTS_V1.md`、`RESULTS_V2.md`、`RESULTS_V3.md`、`RESULTS_V5.md`。

## H16 停在什麼位置

H16 研究反覆撮合的實際 clearing anchor、當前 indicative proposal、配對 residual book，以及上次 proposal acceptance error。它是新的 observation-mechanism hypothesis，尚未有 predictive evidence。

舊 H16 native pilot 的 registration control 通過，但 source evaluation 拒絕了 origin MID parity：拿掉 TwseFilter 後，Jan19/2409 的 1277 筆與 Jan19/3037 的 773 筆 MID 改變；全部 32 cells 的 keys 一致。TwseFilter 原本會把合法 zero-price market touch 正規化到價格上下限，也會丟棄 quantity-zero trial Trade。這是原生資料路由問題，不能當成 H16 訊號失敗，也不能拿變動後的 MID 比較 FE。

新 `AuctionCycleRawInformation` 保留 TwseFilter／CurrentBook，另由私有 C++ queue 保存 filter 前的完整 raw rows。Source receive time 與 publication availability 分開；pre-clock staging 不得改變任何公開 Alpha／Info。它的 19 個 native 單元測試已通過，包含真實 filter 路由、原 reference bits、全部 34 個公開 fields 的 pre-clock privacy、已知零與缺值、overflow 及 exchange/receive 獨立時間軸。

此新版仍 **UNFROZEN／UNEXECUTED**：沒有新 producer capsule、native pilot、raw 41-column prefix proof、full material 或 CatBoost 比較。舊 prefix draft 仍對應舊 36-column schema，不能直接拿來當新版證據。Profile／helper 的 30 個 synthetic tests 通過，只驗證 evaluator 與 lifecycle 的程式契約。

## Checkpoint 的程式與證據

- Research profile、versioned contract／evaluation／runner packages、tests 與 signal CLI 的 profile pointer 保留完整比較歷程。`signal.yaml` 的 pointer 仍是歷史 V1 recipe，不代表最新 candidate 已採用。
- 五個 native experimental modules 集中在 `src/oms/modules/feature/experimental/`，沒有搬進正式 feature 資料夾；後續新增的 material hypotheses 走 C++ FeatureModule／ValueWriter。早期 Python representation prototypes 僅保留作歷史研究證據，不作新版 C++ hypothesis 的 material。
- `expansion.py` 修復了可直接確認的 empty StatusFilter 問題：需要所有 noncontinuous statuses 時，產生 native parser 可接受的 tautology，並有 focused regression tests。
- 完整 Python suite 首次檢查發現兩個 H16 module 缺少 registry review／sample coverage，已補齊 JSON metadata 與 original/full sample 宣告。兩個 module 保持 required Symbol；seed expansion 必須回報缺少 Symbol，不能自行猜測 instrument。相關 21/21 focused tests 通過。這些是工程登錄檢查，沒有更動 frozen FE evaluation contract。
- Ignored runs 中的 49 份自寫 orchestration／validation／test／pipeline sources，以相同 bytes 保存於 [checkpoint_sources/SOURCE_INDEX.md](checkpoint_sources/SOURCE_INDEX.md) 列出的 archive；原 frozen 路徑沒有搬動或修改。Archive 供 review／recovery，原 canonical paths 才是執行位置。
- Datasets、models、native outputs、binary/runtime capsules、logs 及大型 receipts 留在本機 ignored runs/store。Checkpoint 不把這些生成物納入 commit；報告保留其精確本機路徑與 identities。

本次已重新完成 Release build、全部 105/105 CTests（8.81s）、全部 1039/1039 Python tests（190.343s），以及 raw V3 helper 的 30/30 tests。Ruff format/check 的 63 個 active Python paths 通過。完整 Python log 為 `AstraResearch/runs/information_state_20260930/checkpoint-python-tests-corrected-registry.log`；第一次 registry 失敗的 log 另行保留。

另檢查 23 份 `frozen*.yaml` 直接列出的 source／profile records，共 6770 個唯一 path/hash 組合，沒有 source drift；這項檢查沒有重跑或重讀所有 frozen inputs。`PLAN.md` 的 bytes 已受 frozen hash 綁定，`.gitattributes` 僅針對此檔案保留原本的 EOF 空行，其他 whitespace checks 維持不變。

## 證據邊界與恢復順序

目前 baseline 是歷史 native 3254 numeric + 132 categorical controls，含不可直接跨股票比較的 raw scales，不能當成已符合 normalization 要求的完整 representation。C15／C18 的失敗也不能反過來讓 raw control 通過 admission。

2026-01-01 之後、目前已使用的日期均明確列為 exposed development。尚未評分 pristine／sealed OOS；預留的 future-only OOS 不早於 2026-10-01，仍需要至少 30 個先前未觀察、實際存在的日期，不補 missing inputs。沒有 OOS 改善或最終成功宣稱。

只有在使用者明確 resume 後才繼續：先 frozen 新 raw producer／evaluator、驗證全部 32 pilot cells 的原 keys／MID bits 與固定 support gates，再完成 genuine raw receive-prefix proof。通過後才能做 bounded full material 與新的同裁判 FE 比較。`H16_COMPARISON_DESIGN_DRAFT.md` 仍是未採用 proposal；不得逕行執行或用來修改已完成的 comparison。
