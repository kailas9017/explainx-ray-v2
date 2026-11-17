"""Retrain the NIH-only DenseNet-121 for the paper.

The archived checkpoint (trained_model_old/best_model.pth, Aug 2026) never
learned Infiltrate: its outputs are ~0 for every image although 17.7% of
images are positive. It was most likely trained on CSVs whose Infiltrate
column was all zeros, and train.evaluate()'s np.nanmean silently dropped that
class's undefined AUROC, so the mean looked healthy. This script:

* uses NIH data only (dataset.get_dataloaders() mixes in CheXpert when its
  manifests exist),
* starts from ImageNet weights, same recipe as train.py (Adam 1e-4, batch 16,
  BCE, ReduceLROnPlateau, flips + 10 degree rotation),
* refuses to start if any class has no positives in train or validation,
* reports per-class validation AUROC every epoch and fails on any NaN,
* stops early after PATIENCE epochs without a mean-AUROC improvement,
* writes everything to backend/trained_model_nih/ (nothing else is touched).

Mixed precision and DataLoader workers are used only for speed.

Run from backend/:  python research/train_nih.py
"""
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(RESEARCH_DIR))

from config import (BASE_DIR, BATCH_SIZE, DEFAULT_PATHOLOGIES, DEVICE, IMAGE_DIR,  # noqa: E402
                    LEARNING_RATE, NUM_EPOCHS, TRAIN_CSV, VAL_CSV)
from dataset import ChestXrayDataset, eval_transform, train_transform  # noqa: E402
from model import build_model  # noqa: E402
from train import evaluate, find_youden_thresholds  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "trained_model_nih")
PATIENCE = 3
SEED = 2026
WORKERS = 6


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_labels(frame, name):
    pos = frame[DEFAULT_PATHOLOGIES].sum()
    print(f"{name}: {len(frame):,} images; positives per class: {pos.astype(int).to_dict()}")
    empty = pos[pos == 0].index.tolist()
    if empty:
        raise ValueError(f"{name} split has no positives for {empty}; the labels are broken. "
                         "Re-run prepare_data.py before training.")


def train_one_epoch_amp(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    total, n, t0 = 0.0, 0, time.time()
    for i, (images, labels, _observed) in enumerate(loader):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            loss = criterion(model(images), labels).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(images)
        n += len(images)
        if (i + 1) % 500 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  epoch {epoch} batch {i + 1}/{len(loader)} loss {total / n:.4f} "
                  f"({rate:.3f}s/batch, ~{rate * (len(loader) - i - 1) / 60:.0f} min left in epoch)", flush=True)
    return total / max(n, 1)


def main():
    seed_everything(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)
    check_labels(train_df, "train")
    check_labels(val_df, "val")

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(ChestXrayDataset(train_df, IMAGE_DIR, DEFAULT_PATHOLOGIES, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS,
                              pin_memory=True, persistent_workers=True, generator=gen)
    val_loader = DataLoader(ChestXrayDataset(val_df, IMAGE_DIR, DEFAULT_PATHOLOGIES, eval_transform),
                            batch_size=64, shuffle=False, num_workers=WORKERS,
                            pin_memory=True, persistent_workers=True)

    model = build_model(num_classes=len(DEFAULT_PATHOLOGIES), pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auroc_per_class": [], "epoch_minutes": []}
    best, best_epoch, stale = -1.0, 0, 0
    print(f"device {DEVICE}; up to {NUM_EPOCHS} epochs, early-stopping patience {PATIENCE}", flush=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch_amp(model, train_loader, optimizer, criterion, scaler, epoch)
        val_loss, val_auroc, per_class = evaluate(model, val_loader, criterion, DEVICE, DEFAULT_PATHOLOGIES)
        if any(np.isnan(v) for v in per_class.values()):
            raise RuntimeError(f"NaN validation AUROC: {per_class}")
        scheduler.step(val_loss)
        minutes = (time.time() - t0) / 60
        for k, v in (("train_loss", train_loss), ("val_loss", val_loss), ("val_auroc", val_auroc),
                     ("val_auroc_per_class", per_class), ("epoch_minutes", minutes)):
            history[k].append(v)
        print(f"Epoch {epoch}: train {train_loss:.4f} | val loss {val_loss:.4f} | val mean AUROC {val_auroc:.4f} "
              f"| {minutes:.1f} min\n   " + "  ".join(f"{k} {v:.3f}" for k, v in per_class.items()), flush=True)

        if val_auroc > best:
            best, best_epoch, stale = val_auroc, epoch, 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            with open(os.path.join(OUT_DIR, "labels.json"), "w") as f:
                json.dump({"pathologies": DEFAULT_PATHOLOGIES}, f, indent=2)
            print(f"   -> saved best checkpoint (epoch {epoch})", flush=True)
        else:
            stale += 1
        history["best_epoch"], history["best_val_auroc"] = best_epoch, best
        with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if stale >= PATIENCE:
            print(f"early stopping: no improvement for {PATIENCE} epochs", flush=True)
            break

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best_model.pth"), map_location=DEVICE))
    thresholds = find_youden_thresholds(model, val_loader, DEVICE, DEFAULT_PATHOLOGIES)
    with open(os.path.join(OUT_DIR, "thresholds.json"), "w") as f:
        json.dump({"method": "youden_j", "split": "NIH validation", "thresholds": thresholds}, f, indent=2)
    print(f"done. best epoch {best_epoch}, val mean AUROC {best:.4f}; outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
