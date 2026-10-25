# LSTM for fraud detection — Abdullah Balilah

Notebook: [lstm_fraud.ipynb](lstm_fraud.ipynb) (run on Google Colab, Tesla T4). Data and results are in Google Drive under
`MyDrive/fraud_detection_project/` (`data/ieee_sorted_split.parquet`, `results/lstm/`).

## What was done

1. Loaded the team's split file `ieee_sorted_split.parquet` (590,540 payments, sorted by time; train 413,378 / validation 88,581 / test 88,581).
2. Kept column set A from the paper (390 columns). Numbers: training-median fill, standardised with training mean and std. Text columns: categories learned from training, unseen values = "unknown", small embedding.
3. Built a simple client id = `card1` + `addr1` and, for every payment, looked up the client's earlier payments (never later ones).
4. Model: inputs → Linear(64) → LSTM(64) → Linear → fraud score. `W = 8` = current payment + up to 7 earlier ones; `W = 1` = the same network without history.
5. Training as in the team plan: Adam (lr 0.001), weighted BCE with fraud weight √(N0/N1) = 5.24, early stopping on validation ROC-AUC (patience 10, up to 80 epochs), seeds 42, 1, 2. The test set is scored once at the end.
6. Saved for every run: `<run>_val.parquet`, `<run>_test.parquet` (TransactionID, isFraud, logit) and `<run>_history.json`, plus `results_table.csv` and `lstm_curves.png`.

## Results (test set)

| Run | Best epoch | ROC-AUC | PR-AUC |
| --- | --- | --- | --- |
| lstm_A_seed42 (W=8) | 2 | 0.8642 | 0.4189 |
| lstm_A_seed1 (W=8) | 1 | 0.8554 | 0.4200 |
| lstm_A_seed2 (W=8) | 2 | 0.8531 | 0.4168 |
| lstm_nohistory_A_seed42 (W=1) | 2 | 0.8634 | 0.4364 |
| lstm_nohistory_A_seed1 (W=1) | 3 | 0.8559 | 0.4415 |
| lstm_nohistory_A_seed2 (W=1) | 1 | 0.8550 | 0.4232 |

Mean (lowest–highest seed): with history ROC-AUC 0.858 (0.853–0.864), PR-AUC 0.419 (0.417–0.420); without history ROC-AUC 0.858 (0.855–0.863), PR-AUC 0.434 (0.423–0.441).

Reference (team plan, page 1): plain network on column set A 0.841 / 0.410; paper's TabNet–GraphSAGE 0.791 / 0.206; paper's LightGBM 0.834 / 0.394.

## Observations

1. The simple client id (card1 + addr1) gives 39,974 clients, and most test payments (69,154 of 88,581) belong to a "client" with 20 or more earlier payments, so the key probably still mixes several people.
2. The model learns fast and then overfits: the best validation epoch was 1–3 in every run; after that the training loss keeps falling but the validation ROC-AUC falls too.
3. The payment history did not help: for every seed the model without history had the higher PR-AUC (0.434 vs 0.419 on average); ROC-AUC was the same (0.858).
4. A likely reason: the V columns already summarise earlier activity of the card, so the past is already in each row; the rough client id can also mix in other people's payments.
5. The LSTM is at the level of the team's plain network on the same columns and above the numbers reported in the paper (pipelines differ, so this is only a rough comparison).
6. One run takes about 0.8 min with history and 0.5 min without on a T4.

**Limits:** 3 seeds, one window size (8), approximate client id, no tuning.
