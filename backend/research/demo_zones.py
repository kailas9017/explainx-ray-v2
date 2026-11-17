"""Visual sanity check: fixed grid zones vs anatomy-aware lung zones.

Draws, for a few NIH bounding-box images, the image with its expert box and
each zone partition, and prints which zone each partition assigns the box to.
Output: research/outputs/zones_demo.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lung_zones import (anatomical_zones, bbox_mask, bbox_zone_overlap,  # noqa: E402
                        grid_zones, segment_lungs)
from config import DATA_DIR, IMAGE_DIR  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
ZONE_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4"]


def zone_overlay(zones):
    rgba = np.zeros((224, 224, 4))
    for (name, mask), color in zip(zones.items(), ZONE_COLORS):
        rgba[mask] = matplotlib.colors.to_rgba(color, alpha=0.38)
    return rgba


def main(labels=("Nodule", "Effusion", "Pneumothorax")):
    boxes = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
    boxes.columns = ["image", "label", "x", "y", "w", "h"] + list(boxes.columns[6:])
    rows = [boxes[boxes.label == lab].iloc[0] for lab in labels]

    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)))
    grid = grid_zones()
    for i, row in enumerate(rows):
        pil = Image.open(os.path.join(IMAGE_DIR, row.image)).convert("L")
        W, H = pil.size
        img224 = np.asarray(pil.resize((224, 224)))
        right, left = segment_lungs(np.asarray(pil), cache_key=row.image)
        raw, ok = anatomical_zones(right, left, use_hull=False)
        anat, _ = anatomical_zones(right, left)
        box = bbox_mask(row.x, row.y, row.w, row.h, W, H)

        for j, (title, zones) in enumerate((("Fixed 3x2 grid", grid), ("Raw lung masks", raw),
                                            ("Anatomy-aware (hull)", anat))):
            ax = axes[i, j]
            ax.imshow(img224, cmap="gray")
            if zones is not None:
                ax.imshow(zone_overlay(zones))
                overlap = bbox_zone_overlap(box, zones)
                best = max(overlap, key=overlap.get)
                in_zones = sum(overlap.values())
                print(f"{row.label:<13} {title:<20} box zone = {best:<13} "
                      f"({overlap[best]:.0%} of box; {in_zones:.0%} of box inside any zone)")
                title = f"{title}\nbox -> {best}"
            ys, xs = np.nonzero(box)
            ax.add_patch(plt.Rectangle((xs.min(), ys.min()), np.ptp(xs), np.ptp(ys),
                                       fill=False, edgecolor="white", lw=2))
            ax.set_title(f"{row.label}: {title}", fontsize=9)
            ax.axis("off")
        if not ok:
            print(f"{row.label}: lung segmentation failed for {row.image}")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in ZONE_COLORS]
    fig.legend(handles, list(grid.keys()), loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "zones_demo.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
