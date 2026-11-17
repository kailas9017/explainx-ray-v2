import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from config import (
    DEFAULT_PATHOLOGIES, DEVICE, NUM_EPOCHS, LEARNING_RATE,
    MODEL_PATH, EXPERIMENT_MODEL_DIR, EXPERIMENT_MODEL_PATH, EXPERIMENT_LABELS_PATH,
    CHEXPERT_MANIFEST_DIR,
)
from dataset import get_dataloaders, get_chexpert_validation_loader
from model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_observed = 0.0
    for images, labels, observed in loader:
        images, labels, observed = images.to(device), labels.to(device), observed.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        losses = criterion(outputs, labels)
        loss = (losses * observed).sum() / observed.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        total_loss += losses.detach().mul(observed).sum().item()
        total_observed += observed.sum().item()
    return total_loss / max(total_observed, 1.0)


def evaluate(model, loader, criterion, device, pathologies):
    model.eval()
    all_labels, all_preds, all_observed = [], [], []
    total_loss = 0.0
    total_observed = 0.0
    with torch.no_grad():
        for images, labels, observed in loader:
            images, labels, observed = images.to(device), labels.to(device), observed.to(device)
            outputs = model(images)
            losses = criterion(outputs, labels)
            total_loss += losses.mul(observed).sum().item()
            total_observed += observed.sum().item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())
            all_observed.append(observed.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_observed = np.concatenate(all_observed).astype(bool)

    aurocs = []
    for i in range(all_labels.shape[1]):
        known = all_observed[:, i]
        try:
            auroc = roc_auc_score(all_labels[known, i], all_preds[known, i])
        except ValueError:
            auroc = float("nan")
        aurocs.append(auroc)

    avg_loss = total_loss / max(total_observed, 1.0)
    mean_auroc = float(np.nanmean(aurocs))
    return avg_loss, mean_auroc, dict(zip(pathologies, aurocs))


def find_youden_thresholds(model, loader, device, pathologies):
    """Choose per-label operating points that maximize Youden's J on validation."""
    model.eval()
    labels_all, probs_all, observed_all = [], [], []
    with torch.no_grad():
        for images, labels, observed in loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            labels_all.append(labels.numpy())
            probs_all.append(probs)
            observed_all.append(observed.numpy().astype(bool))

    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)
    observed_all = np.concatenate(observed_all)
    thresholds = {}
    for index, name in enumerate(pathologies):
        known = observed_all[:, index]
        y_true = labels_all[known, index]
        y_score = probs_all[known, index]
        if len(np.unique(y_true)) < 2:
            thresholds[name] = 0.5
            continue
        fpr, tpr, candidates = roc_curve(y_true, y_score)
        thresholds[name] = round(float(candidates[np.argmax(tpr - fpr)]), 6)
    return thresholds


def save_checkpoint(model):
    """Save experimental weights separately from the model currently being served."""
    os.makedirs(EXPERIMENT_MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), EXPERIMENT_MODEL_PATH)
    with open(EXPERIMENT_LABELS_PATH, "w") as f:
        json.dump({"pathologies": DEFAULT_PATHOLOGIES}, f, indent=2)


def main():
    os.makedirs(EXPERIMENT_MODEL_DIR, exist_ok=True)
    print("Using device:", DEVICE)
    print("Training on pathologies:", DEFAULT_PATHOLOGIES)
    train_loader, val_loader, test_loader = get_dataloaders(DEFAULT_PATHOLOGIES)
    print(f"Training samples per epoch: {len(train_loader.dataset):,}")

    model = build_model(num_classes=len(DEFAULT_PATHOLOGIES)).to(DEVICE)
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Fine-tuning existing checkpoint: {MODEL_PATH}")
    else:
        print("No existing checkpoint found; starting from ImageNet initialization.")

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = float("-inf")

    # When CheXpert manifests exist, select checkpoints on the patient-held-out
    # CheXpert validation subset, not a mixed NIH/CheXpert score. This matches
    # the target domain and avoids validation leakage from the NIH replay set.
    has_chexpert = os.path.isfile(os.path.join(CHEXPERT_MANIFEST_DIR, "train.csv"))
    target_loader = (
        get_chexpert_validation_loader(DEFAULT_PATHOLOGIES) if has_chexpert else None
    )

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auroc, per_class = evaluate(model, val_loader, criterion, DEVICE, DEFAULT_PATHOLOGIES)
        selection_auroc = val_auroc
        selection_loss = val_loss
        if has_chexpert:
            # The loader returns the mixed validation set; use a dedicated
            # CheXpert validation loader for model selection.
            selection_loss, selection_auroc, target_per_class = evaluate(
                model, target_loader, criterion, DEVICE, DEFAULT_PATHOLOGIES
            )
            print("   CheXpert validation AUROC:", ", ".join(
                f"{name}={score:.3f}" for name, score in target_per_class.items()
            ))
        scheduler.step(selection_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f}")
        print("   Per-class validation AUROC:", ", ".join(
            f"{name}={score:.3f}" for name, score in per_class.items()
        ))

        if selection_auroc > best_auroc:
            best_auroc = selection_auroc
            save_checkpoint(model)
            threshold_loader = target_loader if has_chexpert else val_loader
            thresholds = find_youden_thresholds(model, threshold_loader, DEVICE, DEFAULT_PATHOLOGIES)
            with open(os.path.join(EXPERIMENT_MODEL_DIR, "thresholds.json"), "w") as f:
                json.dump({"method": "youden_j", "thresholds": thresholds}, f, indent=2)
            print(f"   -> Experimental checkpoint saved to {EXPERIMENT_MODEL_PATH} (target AUROC {best_auroc:.4f}).")

    with open(os.path.join(EXPERIMENT_MODEL_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(EXPERIMENT_MODEL_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(history["val_auroc"], label="Validation Mean AUROC", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Validation AUROC over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(EXPERIMENT_MODEL_DIR, "auroc_curve.png"), dpi=150, bbox_inches="tight")

    print("Training complete. Best validation AUROC:", best_auroc)
    print(f"Experimental checkpoint: {EXPERIMENT_MODEL_PATH}")
    print(f"Labels:                  {EXPERIMENT_LABELS_PATH}")


if __name__ == "__main__":
    main()
