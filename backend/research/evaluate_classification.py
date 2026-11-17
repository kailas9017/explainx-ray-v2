"""Classification performance of the NIH model on the held-out NIH test split.

Reuses evaluate_test.py's prediction and Youden-threshold logic, and adds what a
paper table needs: AUROC and AUPRC with 95% patient-level bootstrap confidence
intervals, plus sensitivity/specificity/F1 at validation-fitted Youden cut-offs.

NIH-only loaders are built directly: dataset.get_dataloaders() would merge the
CheXpert validation manifest into the validation set when it exists.

Usage (from backend/):  python research/evaluate_classification.py
Outputs: research/outputs/classification_{per_class.csv,summary.json,predictions.npz}
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RESEARCH_DIR)
from evaluate_zones import DEFAULT_LABELS, DEFAULT_MODEL  # noqa: E402

from config import DEVICE, IMAGE_DIR, TEST_CSV, VAL_CSV  # noqa: E402
from dataset import ChestXrayDataset, eval_transform  # noqa: E402
from evaluate_test import collect_predictions, youden_threshold  # noqa: E402
from model import build_model  # noqa: E402

OUT_DIR = os.path.join(RESEARCH_DIR, "outputs")
N_BOOT = 1000
SEED = 2026


def predict(model, csv_path, labels):
    frame = pd.read_csv(csv_path)
    loader = DataLoader(ChestXrayDataset(frame, IMAGE_DIR, labels, eval_transform),
                        batch_size=64, shuffle=False, num_workers=0)
    y, p = collect_predictions(model, loader, DEVICE)
    return frame["Patient ID"].to_numpy(), y, p


def patient_bootstrap(patients, y, p, metric, n_boot=N_BOOT, seed=SEED):
    """95% CI of a per-class metric, resampling patients (not images) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        per_class = [metric(y[idx, c], p[idx, c]) for c in range(y.shape[1])
                     if np.unique(y[idx, c]).size == 2]
        stats.append(per_class + [np.mean(per_class)])
    stats = np.array(stats)
    return np.percentile(stats, 2.5, axis=0), np.percentile(stats, 97.5, axis=0)


def main():
    with open(DEFAULT_LABELS) as f:
        labels = json.load(f)["pathologies"]
    model = build_model(num_classes=len(labels), pretrained=False)
    model.load_state_dict(torch.load(DEFAULT_MODEL, map_location=DEVICE))
    model.to(DEVICE).eval()

    print("predicting validation split ...", flush=True)
    _, val_y, val_p = predict(model, VAL_CSV, labels)
    print("predicting test split ...", flush=True)
    test_pid, test_y, test_p = predict(model, TEST_CSV, labels)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "classification_predictions.npz"),
                        val_y=val_y, val_p=val_p, test_y=test_y, test_p=test_p, test_pid=test_pid)

    print(f"bootstrapping ({N_BOOT} patient-level resamples) ...", flush=True)
    auc_lo, auc_hi = patient_bootstrap(test_pid, test_y, test_p, roc_auc_score)
    ap_lo, ap_hi = patient_bootstrap(test_pid, test_y, test_p, average_precision_score)

    rows = []
    for c, name in enumerate(labels):
        thr = youden_threshold(val_y[:, c], val_p[:, c])
        pred = test_p[:, c] >= thr
        pos, neg = test_y[:, c] == 1, test_y[:, c] == 0
        tp, fp = int((pred & pos).sum()), int((pred & neg).sum())
        sens = tp / pos.sum()
        spec = int((~pred & neg).sum()) / neg.sum()
        prec = tp / max(tp + fp, 1)
        rows.append({
            "pathology": name, "test_positives": int(pos.sum()), "prevalence": pos.mean(),
            "auroc": roc_auc_score(test_y[:, c], test_p[:, c]), "auroc_lo": auc_lo[c], "auroc_hi": auc_hi[c],
            "auprc": average_precision_score(test_y[:, c], test_p[:, c]), "auprc_lo": ap_lo[c], "auprc_hi": ap_hi[c],
            "youden_threshold_val": thr, "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": 2 * prec * sens / max(prec + sens, 1e-12),
        })
    table = pd.DataFrame(rows)
    summary = {
        "model": DEFAULT_MODEL, "test_images": int(len(test_y)),
        "test_patients": int(len(np.unique(test_pid))),
        "mean_auroc": float(table["auroc"].mean()), "mean_auroc_ci": [float(auc_lo[-1]), float(auc_hi[-1])],
        "mean_auprc": float(table["auprc"].mean()), "mean_auprc_ci": [float(ap_lo[-1]), float(ap_hi[-1])],
        "bootstrap": f"{N_BOOT} patient-level resamples, percentile 95% CI, seed {SEED}",
        "thresholds": "Youden's J fit on NIH validation split only",
    }
    table.to_csv(os.path.join(OUT_DIR, "classification_per_class.csv"), index=False)
    with open(os.path.join(OUT_DIR, "classification_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
