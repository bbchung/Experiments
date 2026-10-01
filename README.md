# AstraResearch Experiments

本 repository 是 AstraResearch 的實驗區，收錄實驗性腳本（scripts）、設定檔（configs）及實驗結果（例如研究紀錄與報告）。這些內容用於探索和驗證，可能隨實驗調整；個別結果請連同其設定與實驗條件閱讀。

本 repository 作為 `AstraResearch/Experiments` 子模組使用；Python 模組以 `AstraResearch.Experiments.*` 匯入。執行時產生的資料與大型輸出應存放於主專案的 `AstraResearch/runs/`。

`instruments.py` 供股票、期貨與權證研究腳本共用，不屬於 factory workflow。
