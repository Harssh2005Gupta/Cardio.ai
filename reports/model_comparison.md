| Model | CV ROC-AUC (mean +/- std) | Selected |
|---|---|---|
| logistic_regression | 0.790 +/- 0.063 |  |
| decision_tree | 0.741 +/- 0.054 |  |
| random_forest | 0.771 +/- 0.085 |  |
| svm | 0.805 +/- 0.062 | **YES** |
| neural_network_dl | 0.731 +/- 0.075 |  |

**Final test-set evaluation (computed once, winner only: `svm`):**

| Metric | Value |
|---|---|
| accuracy | 0.6964 |
| precision | 0.7037 |
| recall | 0.6786 |
| f1_score | 0.6909 |
| roc_auc | 0.7895 |
| pr_auc | 0.7792 |
| brier_score | 0.1896 |