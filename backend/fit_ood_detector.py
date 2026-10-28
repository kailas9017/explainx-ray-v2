"""Fit Mahalanobis OOD statistics from the trained model's training features.

Run from the backend directory after training:
    python fit_ood_detector.py
"""

import json
import os

import numpy as np
import torch

from config import MODEL_PATH, LABELS_PATH, DEVICE, MODEL_DIR
from dataset import get_dataloaders
from model import build_model


OOD_STATS_PATH = os.path.join(MODEL_DIR, "ood_stats.npz")


def extract_penultimate_features(model, loader, device):
    """Collect the 1024-dimensional vectors fed into DenseNet's classifier."""
    features = []

    def hook(_module, inputs, _output):
        features.append(inputs[0].detach().cpu().numpy())

    handle = model.classifier.register_forward_hook(hook)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch[0]
                model(images.to(device))
    finally:
        handle.remove()
    return np.concatenate(features, axis=0)


def main():
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)["pathologies"]

    model = build_model(num_classes=len(labels), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    train_loader, _, _ = get_dataloaders(labels)
    feats = extract_penultimate_features(model, train_loader, DEVICE)
    mean = np.mean(feats, axis=0)
    cov = np.cov(feats, rowvar=False)
    # The covariance can be singular; a pseudo-inverse is intentional here.
    cov_inv = np.linalg.pinv(cov)

    centered = feats - mean
    distances = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, cov_inv, centered), 0.0))
    threshold = float(np.percentile(distances, 99))

    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(OOD_STATS_PATH, mean=mean, cov_inv=cov_inv, threshold=threshold)
    print(f"Saved OOD stats to {OOD_STATS_PATH}")
    print(
        f"Train distance stats: mean={distances.mean():.2f}, "
        f"p99={threshold:.2f}, max={distances.max():.2f}"
    )


if __name__ == "__main__":
    main()
