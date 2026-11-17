"""Qualitative figure: CAM heatmaps with anatomy-aware zones and expert boxes.

Picks, for each of a few pathologies, the held-out (val/test) box image where the
model is most confident, and shows the radiograph with its lung zones and box,
followed by each CAM method's heatmap.

Run from backend/:  python research/make_example_figure.py
Output: research/outputs/fig_examples.{png,pdf}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RESEARCH_DIR)
from evaluate_zones import (DEFAULT_LABELS, DEFAULT_MODEL, EVAL_TRANSFORM, METHODS,  # noqa: E402
                            load_model, make_cam_extractors, run_cam, split_lookup)
from lung_zones import anatomical_zones, bbox_mask, segment_lungs  # noqa: E402
from analyze_results import METHOD_NAME, save  # noqa: E402

from config import DATA_DIR, DEVICE, IMAGE_DIR  # noqa: E402

PATHOLOGIES = ["Nodule", "Mass", "Effusion", "Pneumothorax"]


def zone_outline(ax, zones):
    for mask in zones.values():
        ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=0.6, alpha=0.9)


def draw_box(ax, box):
    ys, xs = np.nonzero(box)
    ax.add_patch(plt.Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                               fill=False, edgecolor="#1baf7a", lw=1.6))


def main():
    model, labels = load_model(DEFAULT_MODEL, DEFAULT_LABELS)
    extractors = make_cam_extractors(model, METHODS)
    splits = split_lookup()
    boxes = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv")).iloc[:, :6]
    boxes.columns = ["image", "label", "x", "y", "w", "h"]
    boxes = boxes[boxes.image.map(splits).isin(["val", "test"])]

    picks = []
    for lab in PATHOLOGIES:
        best, best_p = None, -1
        for b in boxes[boxes.label == lab].itertuples():
            with Image.open(os.path.join(IMAGE_DIR, b.image)) as img:
                t = EVAL_TRANSFORM(img.convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p = float(torch.sigmoid(model(t))[0, labels.index(lab)])
            if p > best_p:
                best, best_p = b, p
        picks.append((best, best_p))

    ncols = 1 + len(METHODS)
    fig, axes = plt.subplots(len(picks), ncols, figsize=(2.0 * ncols, 2.15 * len(picks)))
    for r, (b, p) in enumerate(picks):
        with Image.open(os.path.join(IMAGE_DIR, b.image)) as img:
            gray = np.asarray(img.convert("L"))
            rgb = img.convert("RGB")
        W, H = rgb.size
        img224 = np.asarray(rgb.convert("L").resize((224, 224)))
        t = EVAL_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
        zones, _ = anatomical_zones(*segment_lungs(gray, cache_key=b.image))
        box = bbox_mask(b.x, b.y, b.w, b.h, W, H)

        ax = axes[r, 0]
        ax.imshow(img224, cmap="gray")
        zone_outline(ax, zones)
        draw_box(ax, box)
        ax.set_ylabel(f"{b.label}\np = {p:.2f}", fontsize=8)
        if r == 0:
            ax.set_title("Zones + expert box", fontsize=8)
        for c, m in enumerate(METHODS, start=1):
            cam = run_cam(extractors[m], t, labels.index(b.label))
            ax = axes[r, c]
            ax.imshow(img224, cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            zone_outline(ax, zones)
            draw_box(ax, box)
            if r == 0:
                ax.set_title(METHOD_NAME[m], fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, "fig_examples")
    print("saved fig_examples; picks:", [(b.image, b.label, round(p, 3)) for b, p in picks])


if __name__ == "__main__":
    main()
