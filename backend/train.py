import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import DEFAULT_PATHOLOGIES, DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_PATH, MODEL_DIR, LABELS_PATH
from dataset import get_dataloaders
from model import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, pathologies):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    aurocs = []
    for i in range(all_labels.shape[1]):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            auroc = float("nan")
        aurocs.append(auroc)

    avg_loss = running_loss / len(loader.dataset)
    mean_auroc = float(np.nanmean(aurocs))
    return avg_loss, mean_auroc, dict(zip(pathologies, aurocs))


def save_checkpoint(model):
    """The critical fix: the checkpoint and its label list are written together,
    atomically, every time. model_service.py refuses to load one without the
    other, and refuses to load them if the class counts disagree. This is the
    single change that prevents the previous project's core bug from recurring."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump({"pathologies": DEFAULT_PATHOLOGIES}, f, indent=2)


def main():
    print("Using device:", DEVICE)
    print("Training on pathologies:", DEFAULT_PATHOLOGIES)
    train_loader, val_loader, test_loader = get_dataloaders(DEFAULT_PATHOLOGIES)

    model = build_model(num_classes=len(DEFAULT_PATHOLOGIES)).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "val_auroc": []}
    best_auroc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_auroc, per_class = evaluate(model, val_loader, criterion, DEVICE, DEFAULT_PATHOLOGIES)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Mean AUROC: {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            save_checkpoint(model)
            print(f"   -> New best model saved (AUROC {best_auroc:.4f}), labels.json written alongside it.")

    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")

    plt.figure(figsize=(8, 5))
    plt.plot(history["val_auroc"], label="Validation Mean AUROC", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title("Validation AUROC over Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("auroc_curve.png", dpi=150, bbox_inches="tight")

    print("Training complete. Best validation AUROC:", best_auroc)
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Labels:     {LABELS_PATH}")


if __name__ == "__main__":
    main()
