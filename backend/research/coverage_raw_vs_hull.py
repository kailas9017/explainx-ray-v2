"""Share of each expert box inside raw lung masks vs. hull-based anatomical zones,
plus the share of CAM mass inside the anatomical zones, per finding.

Uses the lung masks cached by evaluate_zones.py (no model needed).
Run from backend/:  python research/coverage_raw_vs_hull.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(RESEARCH_DIR))
sys.path.insert(0, RESEARCH_DIR)

from config import DATA_DIR, IMAGE_DIR  # noqa: E402
from lung_zones import anatomical_zones, bbox_mask, segment_lungs  # noqa: E402

OUT_DIR = os.path.join(RESEARCH_DIR, "outputs")


def main():
    det = pd.read_csv(os.path.join(OUT_DIR, "zone_eval_details.csv"))
    lung = det[(det.label != "Cardiomegaly") & (det.seg_ok == 1)]
    boxes = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv")).iloc[:, :6]
    boxes.columns = ["image", "label", "x", "y", "w", "h"]

    rows = []
    for box_id in sorted(lung.box_id.unique()):
        b = boxes.iloc[box_id]
        with Image.open(os.path.join(IMAGE_DIR, b.image)) as img:
            W, H = img.size
            gray = np.asarray(img.convert("L"))
        right, left = segment_lungs(gray, cache_key=b.image)
        zones, ok = anatomical_zones(right, left)
        box = bbox_mask(b.x, b.y, b.w, b.h, W, H)
        area = box.sum()
        hull = np.zeros_like(box)
        for m in zones.values():
            hull |= m
        rows.append({"box_id": box_id, "label": lung[lung.box_id == box_id].label.iloc[0],
                     "raw": float((box & (right | left)).sum() / area),
                     "hull": float((box & hull).sum() / area)})
    cov = pd.DataFrame(rows)

    lc = lung[lung.method == "layercam"]
    per_class = cov.groupby("label")[["raw", "hull"]].mean()
    per_class["cam_in_lung_zones_layercam"] = lc.groupby("label").anat_cam_in_zones.mean()
    per_class["n"] = cov.groupby("label").size()
    cam_in = lung.groupby("method").anat_cam_in_zones.agg(["mean", "median"])
    out = {"n_boxes": len(cov),
           "box_in_raw_mask_mean": cov.raw.mean(), "box_in_hull_mean": cov.hull.mean(),
           "box_outside_raw_mask_mean": 1 - cov.raw.mean(), "box_outside_hull_mean": 1 - cov.hull.mean(),
           "per_class": per_class.round(3).to_dict(orient="index"),
           "cam_mass_in_anatomical_zones_by_method": cam_in.round(3).to_dict(orient="index")}
    print(json.dumps(out, indent=2))
    with open(os.path.join(OUT_DIR, "coverage_summary.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
