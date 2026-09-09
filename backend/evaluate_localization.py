"""Offline Grad-CAM localization evaluation against NIH bounding boxes.

Run from the backend directory after placing NIH images in ``data/images`` and
``BBox_List_2017.csv`` in ``data``::

    python evaluate_localization.py
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from config import BBOX_CSV_PATH, IMAGE_DIR, DEVICE, DEFAULT_PATHOLOGIES
from gradcam_utils import compute_gradcam
from model_service import EVAL_TRANSFORM, _LABELS, _MODEL, ModelNotReadyError
from severity import compute_asfi

OUTPUT_CSV = "localization_results.csv"
OUTPUT_CHART = "localization_iou.png"


def _column(frame, *names):
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]
    raise ValueError(f"Missing required CSV column. Expected one of: {names}")


def _bbox_from_row(row, frame):
    """Read either NIH's four bbox columns or its single bbox string column."""
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    candidates = [
        ("bbox [x", "bbox [y", "w", "h"),
        # NIH's unquoted canonical header is split by CSV parsers this way.
        ("bbox [x", "y", "w", "h]"),
        ("x", "y", "w", "h"),
    ]
    for names in candidates:
        if all(name in normalized for name in names):
            return tuple(float(row[normalized[name]]) for name in names)

    value = row[_column(frame, "Bbox [x, y, w, h]", "Bbox")]
    values = [float(number) for number in re.findall(r"[-+]?\d*\.?\d+", str(value))]
    if len(values) != 4:
        raise ValueError(f"Could not parse bounding box: {value!r}")
    return tuple(values)


def _image_path(image_name):
    direct = os.path.join(IMAGE_DIR, image_name)
    if os.path.exists(direct):
        return direct
    for root, _, files in os.walk(IMAGE_DIR):
        if image_name in files:
            return os.path.join(root, image_name)
    return None


def _mask_iou(mask, box):
    x, y, width, height = box
    x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x1, y1 = min(mask.shape[1], int(np.ceil(x + width))), min(mask.shape[0], int(np.ceil(y + height)))
    ground_truth = np.zeros(mask.shape, dtype=bool)
    if x1 > x0 and y1 > y0:
        ground_truth[y0:y1, x0:x1] = True
    union = np.logical_or(mask, ground_truth).sum()
    return float(np.logical_and(mask, ground_truth).sum() / union) if union else 0.0


def main():
    if _MODEL is None:
        raise ModelNotReadyError("The trained model must be available to evaluate localization.")
    if not os.path.exists(BBOX_CSV_PATH):
        raise FileNotFoundError(f"Bounding-box CSV not found: {BBOX_CSV_PATH}")
    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"NIH image directory not found: {IMAGE_DIR}")

    frame = pd.read_csv(BBOX_CSV_PATH).dropna(axis=1, how="all")
    image_col = _column(frame, "Image Index")
    label_col = _column(frame, "Finding Label", "Finding Labels")
    frame[label_col] = frame[label_col].astype(str).str.strip()
    frame = frame[frame[label_col].isin(DEFAULT_PATHOLOGIES)].copy()

    rows = []
    missing_images = 0
    for _, row in frame.iterrows():
        label = row[label_col]
        if label not in _LABELS:
            continue
        path = _image_path(str(row[image_col]).strip())
        if path is None:
            missing_images += 1
            continue

        with Image.open(path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        cam = compute_gradcam(_MODEL, tensor, _LABELS.index(label))
        _, _, _, threshold = compute_asfi(cam, 0.0)
        activation_mask = cam > threshold
        x, y, width, height = _bbox_from_row(row, frame)
        scaled_box = (x * 224 / original_width, y * 224 / original_height,
                      width * 224 / original_width, height * 224 / original_height)
        rows.append({"image": row[image_col], "pathology": label, "iou": _mask_iou(activation_mask, scaled_box)})

    if not rows:
        raise RuntimeError("No annotated images were evaluated. Check data/images and the bbox CSV.")

    details = pd.DataFrame(rows)
    summary = details.groupby("pathology", as_index=False).agg(mean_iou=("iou", "mean"), samples=("iou", "size"))
    summary = summary.sort_values("mean_iou", ascending=False)
    overall = details["iou"].mean()
    summary.to_csv(OUTPUT_CSV, index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(summary["pathology"], summary["mean_iou"], color="#2dd4e8")
    plt.ylabel("Mean IoU")
    plt.title(f"Grad-CAM localization IoU (overall: {overall:.3f})")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=200)

    print(summary.to_string(index=False, formatters={"mean_iou": "{:.3f}".format}))
    print(f"\nOverall mean IoU: {overall:.3f}")
    print(f"Saved {OUTPUT_CSV} and {OUTPUT_CHART}.")
    if missing_images:
        print(f"Skipped {missing_images} annotation(s) because the image file was unavailable.")


if __name__ == "__main__":
    main()
