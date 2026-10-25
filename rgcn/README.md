# R-GCN on the team's entity graph — Abdullah Balilah

Notebook: [rgcn_fraud.ipynb](rgcn_fraud.ipynb) (run on Google Colab, Tesla T4). Data and results are in Google Drive under
`MyDrive/fraud_detection_project/` (`data/ieee_sorted_split.parquet`, `results/rgcn/`).

## What was done

1. Loaded the team's split file `ieee_sorted_split.parquet` (590,540 payments, sorted by time; train 413,378 / validation 88,581 / test 88,581).
2. Built the entity graph with the plan's rules: 6 entity columns (card1, addr1, P_emaildomain, R_emaildomain, DeviceInfo, DeviceType), each value is a node, plus one "missing" and one "unknown" node per column. Two entities are linked if they appear in the same training payment. Validation and test payments are never used to build the graph.
3. Checked the graph against the plan: 14,239 nodes and 12,401,340 links before repeats are removed (168,118 after). Both match.
4. Built the shuffled control graph: inside the training payments, each entity column is shuffled on its own (seeds 42, 1, 2), so the nodes stay the same but the links become random.
5. Model: node embedding + entity-type embedding (64) → 2 R-GCN layers with one weight matrix per link type (30 types) → the payment's 6 node vectors are joined (6×64) → Linear(384,128) → ReLU → dropout → Linear(128,1). 10% of training entities are replaced by their "unknown" node so that node learns something.
6. Training as in the team plan: Adam (lr 0.001), weighted BCE with fraud weight √(N0/N1) = 5.24, 12 epochs, best epoch chosen by validation ROC-AUC, seeds 42, 1, 2 on the real graph and on the shuffled graph. The test set is scored once at the end.
7. Saved for every run: `<run>_val.parquet`, `<run>_test.parquet` (TransactionID, isFraud, logit) and `<run>_history.json`, plus `results_table.csv`, `rgcn_curves.png` and `link_type_drop.csv`.

## Results (test set)

| Run | Best epoch | ROC-AUC | PR-AUC |
| --- | --- | --- | --- |
| rgcn_seed42 | 8 | 0.7609 | 0.1562 |
| rgcn_seed1 | 10 | 0.7488 | 0.1467 |
| rgcn_seed2 | 6 | 0.7601 | 0.1332 |
| rgcn_shuf_seed42 | 8 | 0.7629 | 0.1530 |
| rgcn_shuf_seed1 | 8 | 0.7517 | 0.1459 |
| rgcn_shuf_seed2 | 8 | 0.7578 | 0.1407 |

Mean (lowest–highest seed): real graph ROC-AUC 0.757 (0.749–0.761), PR-AUC 0.145 (0.133–0.156); shuffled graph ROC-AUC 0.758 (0.752–0.763), PR-AUC 0.147 (0.141–0.153).

Reference (team plan): GraphSAGE alone 0.741 / 0.138 (1 seed); plain network on column set A 0.841 / 0.410.

## Observations

1. The graph matches the plan's checks, so it follows the same rules as the SOA member's graph.
2. Much of the entity information is missing: in the test set DeviceInfo is missing for 83.0% of payments, DeviceType for 79.5% and R_emaildomain for 79.1%. For most payments the model only sees card1, addr1 and P_emaildomain.
3. R-GCN alone is weak but a little above GraphSAGE alone. With no payment columns it cannot reach the tabular models.
4. The real links do not beat random links: real minus shuffled per seed is −0.002, −0.003, +0.002 in ROC-AUC and +0.003, +0.001, −0.008 in PR-AUC, smaller than the spread between seeds.
5. A likely reason: each payment still keeps its own card, address, e-mail and device nodes with their own learned embeddings, so most of the score comes from "which card / which e-mail domain", not from the neighbours.
6. Switching off one link type at a time lowers the validation ROC-AUC (0.7945) by at most 0.006 (R_emaildomain → card1), then DeviceInfo → card1 (0.005) and P_emaildomain → card1 (0.004). No single link type matters much, which agrees with point 4.
7. The best validation epoch was 6–10 out of 12; one run takes about 1 minute on a T4.

**Limits:** graph rebuilt with the plan's rules, not yet the SOA member's file; 3 seeds; learning rate and dropout fixed, not tuned; no bootstrap intervals here (the evaluation member computes them from the saved scores).
