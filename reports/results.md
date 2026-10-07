# Results (TEST split, thresholds chosen on VALID)

Test: 88,581 transactions, 3,083 frauds (3.48%). Cost of approving everything (no model): **$469,609**.
Cost model: missed fraud = amount; review = $5 per transaction; blocking a legitimate customer = $25; review catch rate = 100%.

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision @80% recall | Brier | Expected cost | Saved vs no model | Saved vs rules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rules engine | 0.0413 | 0.5467 | 0.009 | 0.035 | 0.0434 | $271,289 | $198,320 (42.2%) | $0 |
| Logistic regression | 0.2755 | 0.8463 | 0.256 | 0.095 | 0.0304 | $211,483 | $258,125 (55.0%) | $59,806 |
| LightGBM (no reweighting) | 0.5427 | 0.8985 | 0.474 | 0.140 | 0.0225 | $176,377 | $293,232 (62.4%) | $94,912 |
| LightGBM (scale_pos_weight) | 0.5406 | 0.8956 | 0.469 | 0.140 | 0.0233 | $174,600 | $295,008 (62.8%) | $96,689 |
| LightGBM (sqrt scale_pos_weight) | 0.5445 | 0.8993 | 0.470 | 0.139 | 0.0223 | $169,429 | $300,179 (63.9%) | $101,860 |
| LightGBM tuned (Optuna, raw) | 0.5427 | 0.8985 | 0.474 | 0.140 | 0.0225 | $176,377 | $293,232 (62.4%) | $94,912 |
| LightGBM tuned + isotonic (served) | 0.5427 | 0.8985 | 0.474 | 0.140 | 0.0222 | $176,377 | $293,232 (62.4%) | $94,912 |

## Served model at the chosen thresholds

Thresholds (calibrated probability): review >= 0.0250, block >= 0.7547.
Decisions on test: approve 73,600, review 13,822, block 1,159.

| Confusion (flagged = review or block) | Predicted fraud | Predicted legit |
|---|---:|---:|
| Actual fraud | 2,405 | 678 |
| Actual legit | 12,576 | 72,922 |

| Confusion (block only) | Blocked | Not blocked |
|---|---:|---:|
| Actual fraud | 957 | 2,126 |
| Actual legit | 202 | 85,296 |

## Data split (by TransactionDT)

| Split | Rows | Dates | Fraud rate |
|---|---:|---|---:|
| train | 413,378 | 2017-12-01 .. 2018-03-30 | 3.52% |
| valid | 88,581 | 2018-03-30 .. 2018-05-01 | 3.43% |
| test | 88,581 | 2018-05-01 .. 2018-05-31 | 3.48% |
| live (tail of test) | 44,292 | 2018-05-15 .. 2018-05-31 | |
