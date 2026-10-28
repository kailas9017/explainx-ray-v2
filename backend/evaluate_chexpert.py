"""Evaluate the experimental checkpoint on untouched official CheXpert valid.

Run from backend after training:
    python evaluate_chexpert.py
"""

import json

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from config import (
    DEFAULT_PATHOLOGIES, DEVICE, EXPERIMENT_LABELS_PATH, EXPERIMENT_MODEL_PATH,
    MODEL_PATH,
)
from dataset import get_chexpert_official_valid_loader
from model import build_model
from train import evaluate


def evaluate_checkpoint(checkpoint_path, labels, loader):
    model = build_model(num_classes=len(labels), pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    loss, mean_auroc, aurocs = evaluate(
        model, loader, nn.BCEWithLogitsLoss(reduction="none"), DEVICE, labels
    )
    # Cache predictions for threshold-free AP and a fixed 0.5 reference point.
    predictions, targets, observed_rows = [], [], []
    with torch.no_grad():
        for images, batch_labels, observed in loader:
            logits = model(images.to(DEVICE))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch_labels.numpy())
            observed_rows.append(observed.numpy().astype(bool))
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    observed_rows = np.concatenate(observed_rows)
    metrics = {}
    for index, name in enumerate(labels):
        known = observed_rows[:, index]
        y = targets[known, index].astype(int)
        p = predictions[known, index]
        metrics[name] = {
            "auroc": aurocs[name],
            "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "precision@0.5": float(precision_score(y, p >= 0.5, zero_division=0)),
            "recall@0.5": float(recall_score(y, p >= 0.5, zero_division=0)),
            "f1@0.5": float(f1_score(y, p >= 0.5, zero_division=0)),
        }
    return loss, mean_auroc, metrics


def main():
    with open(EXPERIMENT_LABELS_PATH, "r", encoding="utf-8") as labels_file:
        labels = json.load(labels_file)["pathologies"]
    loader = get_chexpert_official_valid_loader(labels)

    baseline_loss, baseline_mean, baseline = evaluate_checkpoint(MODEL_PATH, labels, loader)
    tuned_loss, tuned_mean, tuned = evaluate_checkpoint(EXPERIMENT_MODEL_PATH, labels, loader)
    print(f"Official CheXpert valid baseline loss: {baseline_loss:.4f}")
    print(f"Fine-tuned loss: {tuned_loss:.4f}")
    print(f"Baseline mean AUROC: {baseline_mean:.4f}")
    print(f"Fine-tuned mean AUROC: {tuned_mean:.4f} (delta {tuned_mean - baseline_mean:+.4f})")
    print("Per-class metrics (baseline -> fine-tuned):")
    for name in labels:
        before, after = baseline[name], tuned[name]
        if before["auroc"] != before["auroc"] or after["auroc"] != after["auroc"]:
            print(f"  {name}: not evaluable")
        else:
            print(
                f"  {name}: AUROC {before['auroc']:.3f}->{after['auroc']:.3f}, "
                f"AUPRC {before['auprc']:.3f}->{after['auprc']:.3f}, "
                f"F1@0.5 {before['f1@0.5']:.3f}->{after['f1@0.5']:.3f}, "
                f"P/R@0.5 {after['precision@0.5']:.3f}/{after['recall@0.5']:.3f}"
            )


if __name__ == "__main__":
    main()
