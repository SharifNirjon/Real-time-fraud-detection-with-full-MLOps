# Results (TEST split, thresholds chosen on VALID)

Test: 16,224 transactions, 529 frauds (3.26%). Cost of approving everything (no model): **$65,197**.
Cost model: missed fraud = amount; review = $5 per transaction; blocking a legitimate customer = $25; review catch rate = 100%.

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision @80% recall | Brier | Expected cost | Saved vs no model | Saved vs rules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules engine | 0.0376 | 0.5427 | 0.011 | 0.033 | 0.0414 | $46,041 | $19,156 (29.4%) | $0 |
| Logistic regression | 0.3663 | 0.8591 | 0.293 | 0.096 | 0.0259 | $42,048 | $23,149 (35.5%) | $3,993 |
| LightGBM (no reweighting) | 0.5157 | 0.8911 | 0.461 | 0.094 | 0.0209 | $32,164 | $33,033 (50.7%) | $13,878 |
| LightGBM (scale_pos_weight) | 0.5189 | 0.8857 | 0.463 | 0.112 | 0.0219 | $30,735 | $34,463 (52.9%) | $15,307 |
| LightGBM (sqrt scale_pos_weight) | 0.5268 | 0.8852 | 0.465 | 0.108 | 0.0215 | $31,673 | $33,524 (51.4%) | $14,368 |
| LightGBM tuned (Optuna, raw) | 0.5189 | 0.8857 | 0.463 | 0.112 | 0.0219 | $30,735 | $34,463 (52.9%) | $15,307 |
| LightGBM tuned + isotonic (served) | 0.5189 | 0.8857 | 0.463 | 0.112 | 0.0212 | $30,735 | $34,463 (52.9%) | $15,307 |

## Served model at the chosen thresholds

Thresholds (calibrated probability): review >= 0.0347, block >= 0.8148.
Decisions on test: approve 14,095, review 2,005, block 124.

| Confusion (flagged = review or block) | Predicted fraud | Predicted legit |
|---|---:|---:|
| Actual fraud | 369 | 160 |
| Actual legit | 1,760 | 13,935 |

| Confusion (block only) | Blocked | Not blocked |
|---|---:|---:|
| Actual fraud | 114 | 415 |
| Actual legit | 10 | 15,685 |

## Data split (by TransactionDT)

| Split | Rows | Dates | Fraud rate |
|---|---:|---|---:|
| train | 75,709 | 2017-12-01 .. 2018-03-30 | 3.37% |
| valid | 16,223 | 2018-03-30 .. 2018-04-30 | 3.10% |
| test | 16,224 | 2018-04-30 .. 2018-05-31 | 3.26% |
| live (tail of test) | 8,112 | 2018-05-15 .. 2018-05-31 | |
