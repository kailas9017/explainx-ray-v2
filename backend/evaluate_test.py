"""Evaluate the saved model on NIH test data using thresholds tuned on validation.

Run from the backend directory after preparing data and training a checkpoint:
    python evaluate_test.py

Per-class Youden's J thresholds are derived exclusively from validation labels.
The test split is used only for AUROC and thresholded classification metrics.
Results are written to test_evaluation.csv and test_evaluation.json.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import DEVICE, LABELS_PATH, MODEL_PATH, TEST_CSV, TRAIN_CSV, VAL_CSV, DEFAULT_PATHOLOGIES
from dataset import get_dataloaders
from model import build_model


OUTPUT_CSV = "test_evaluation.csv"
OUTPUT_JSON = "test_evaluation.json"


def collect_predictions(model, loader, device):
    labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for images, batch_labels, _observed in loader:
            logits = model(images.to(device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(labels), np.concatenate(predictions)


def youden_threshold(y_true, y_score):
    """Return the ROC threshold maximizing sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    j_values = tpr[finite] - fpr[finite]
    best = np.flatnonzero(finite)[int(np.argmax(j_values))]
    return float(thresholds[best])


def evaluate_arrays(labels, val_y, val_p, test_y, test_p, pathologies):
    rows = []
    for index, name in enumerate(pathologies):
        # Guard against one-class validation/test labels; ROC/Youden are undefined.
        val_has_two = np.unique(val_y[:, index]).size == 2
        test_has_two = np.unique(test_y[:, index]).size == 2
        threshold = (
            youden_threshold(val_y[:, index], val_p[:, index]) if val_has_two else float("nan")
        )
        test_auroc = (
            float(roc_auc_score(test_y[:, index], test_p[:, index])) if test_has_two else float("nan")
        )
        if np.isfinite(threshold):
            predicted = test_p[:, index] >= threshold
            precision = float(precision_score(test_y[:, index], predicted, zero_division=0))
            recall = float(recall_score(test_y[:, index], predicted, zero_division=0))
            f1 = float(f1_score(test_y[:, index], predicted, zero_division=0))
        else:
            precision = recall = f1 = float("nan")
        rows.append({
            "pathology": name,
            "validation_youden_threshold": threshold,
            "test_auroc": test_auroc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "validation_positives": int(val_y[:, index].sum()),
            "test_positives": int(test_y[:, index].sum()),
        })
    mean_auroc = float(np.nanmean([row["test_auroc"] for row in rows]))
    return rows, mean_auroc


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not os.path.isfile(LABELS_PATH):
        raise FileNotFoundError(f"Model labels not found: {LABELS_PATH}")
    for path in (TRAIN_CSV, VAL_CSV, TEST_CSV):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prepared split not found: {path}. Run prepare_data.py first.")

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        pathologies = json.load(file)["pathologies"]
    if pathologies != DEFAULT_PATHOLOGIES:
        raise ValueError(
            "Checkpoint labels do not match the configured data columns. "
            f"Checkpoint={pathologies}; config={DEFAULT_PATHOLOGIES}"
        )

    _train_loader, val_loader, test_loader = get_dataloaders(pathologies)
    model = build_model(num_classes=len(pathologies), pretrained=False)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)

    val_y, val_p = collect_predictions(model, val_loader, DEVICE)
    test_y, test_p = collect_predictions(model, test_loader, DEVICE)
    rows, mean_auroc = evaluate_arrays(pathologies, val_y, val_p, test_y, test_p, pathologies)

    for row in rows:
        row["mean_test_auroc"] = mean_auroc
    results = {
        "model_path": MODEL_PATH,
        "validation_split": VAL_CSV,
        "test_split": TEST_CSV,
        "threshold_method": "Youden's J (max TPR - FPR), fit on validation split only",
        "mean_test_auroc": mean_auroc,
        "per_class": rows,
    }
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=True)

    print(f"Mean test AUROC: {mean_auroc:.4f}")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved {OUTPUT_CSV} and {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
