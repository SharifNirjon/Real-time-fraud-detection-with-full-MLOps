# Retraining log

Every retraining run and its decision (newest last).

| Run (UTC) | Trigger | Champion | Challenger | Holdout rows (frauds) | Champion PR-AUC | Challenger PR-AUC | Challenger p95 latency | Decision | Reason |
|---|---|---|---|---|---:|---:|---:|---|---|
| 2026-10-07 05:03:28 | drift | v1 | v2 | 11,639 (449) | 0.4880 | 0.6699 | 3.7 ms | **PROMOTED** | PR-AUC gain +0.1819 >= +0.0050 (challenger 0.6699 vs champion 0.4880), latency p95 3.7ms within budget |
