# Performance metrics

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from statsmodels.stats.contingency_tables import mcnemar

from main import models, predictions, test_data


# Test labels
y_test = test_data["y_test"]

# Metrics
results = []

for name, y_pred in predictions.items():

    model = models[name]

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(test_data["X_test"])[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": auc,
    })

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# Classification reports
print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS")
print("=" * 60)

for name, y_pred in predictions.items():
    print(f"\n--- {name} ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Disease", "Has Disease"],
            digits=4,
            zero_division=0,
        )
    )


# Confusion matrices
print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)

for name, y_pred in predictions.items():
    print(f"\n--- {name} ---")
    print(confusion_matrix(y_test, y_pred))


# Pairwise McNemar test
print("\n" + "=" * 60)
print("PAIRWISE McNEMAR'S TEST")
print("=" * 60)

model_names = list(predictions.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):

        model_a = model_names[i]
        model_b = model_names[j]

        pred_a = np.asarray(predictions[model_a])
        pred_b = np.asarray(predictions[model_b])
        y_true = np.asarray(y_test)

        correct_a = pred_a == y_true
        correct_b = pred_b == y_true

        table = [
            [
                np.sum(correct_a & correct_b),
                np.sum(correct_a & ~correct_b),
            ],
            [
                np.sum(~correct_a & correct_b),
                np.sum(~correct_a & ~correct_b),
            ],
        ]

        test = mcnemar(table, exact=False, correction=True)

        print(
            f"{model_a} vs {model_b}: "
            f"p-value = {test.pvalue:.4f}"
        )


# Save results
Path("results").mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    "results/model_performance.csv",
    index=False,
)

print("\nSaved: results/model_performance.csv")
