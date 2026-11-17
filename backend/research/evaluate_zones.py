"""Main experiment: zone-level faithfulness of CAM explanations vs NIH expert boxes.

For every NIH bounding box (984 boxes, 8 pathologies) and every CAM method,
this computes the CAM for the box's labelled class and scores it against the
box under two zone partitions:

* ``grid``        — the production fixed 3x2 image grid
* ``anatomical``  — lung-segmentation zones (dilated convex hull, upper/mid/lower thirds)

Per (box, method, partition):
  zone_hit      peak CAM zone (highest mean activation) == zone holding most of the box
  zone_hit_any  peak CAM zone is any zone holding >= 20% of the box
  zone_top2     box zone is among the two highest-activation zones
  cam_share_box_zone   fraction of total CAM mass in the box's zone
Partition-independent:
  pointing      CAM's maximum pixel lies inside the box
  energy_in_box fraction of total CAM mass inside the box
  iou_otsu      IoU of the Otsu-thresholded CAM with the box (as in evaluate_localization.py)
Also stored for the ASFI/ZCI analysis: probability, ASFI, activated fraction,
ZCI under each partition, and the box's area fraction.

Usage (from backend/):
  python research/evaluate_zones.py --limit 20                 # quick check
  python research/evaluate_zones.py                            # full run
  python research/evaluate_zones.py --summarize-only           # rebuild tables from CSV
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

RESEARCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RESEARCH_DIR)
from lung_zones import (anatomical_zones, bbox_mask, bbox_zone_overlap,  # noqa: E402
                        grid_zones, segment_lungs, zone_activation, zone_cam_share)

from config import BASE_DIR, DATA_DIR, DEVICE, IMAGE_DIR  # noqa: E402
from model import build_model  # noqa: E402
from severity import compute_asfi, compute_zci  # noqa: E402

OUT_DIR = os.path.join(RESEARCH_DIR, "outputs")
DETAILS_CSV = os.path.join(OUT_DIR, "zone_eval_details.csv")
# Retrained NIH-only model (research/train_nih.py). The archived trained_model_old
# checkpoint never learned Infiltrate and must not be used for the paper.
DEFAULT_MODEL = os.path.join(BASE_DIR, "trained_model_nih", "best_model.pth")
DEFAULT_LABELS = os.path.join(BASE_DIR, "trained_model_nih", "labels.json")
METHODS = ["gradcam", "gradcam++", "layercam", "scorecam"]
ANY_ZONE_MIN_OVERLAP = 0.20

EVAL_TRANSFORM = T.Compose([  # identical to model_service.EVAL_TRANSFORM
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path, labels_path):
    with open(labels_path) as f:
        labels = json.load(f)["pathologies"]
    model = build_model(num_classes=len(labels), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # torchvision's DenseNet applies an in-place ReLU *after* model.features, so a
    # hook on model.features sees pre-ReLU maps (the CAM library copies them before
    # the ReLU; the app's compute_gradcam keeps the tensor and sees post-ReLU).
    # Appending an explicit ReLU makes every CAM method use the post-ReLU maps that
    # actually feed the classifier. forward() is replaced with the same computation
    # minus the in-place ReLU (which would overwrite the hooked tensor and break
    # backward). Logits are unchanged.
    model.features.add_module("relu_final", torch.nn.ReLU())

    def forward(self, x):
        pooled = torch.nn.functional.adaptive_avg_pool2d(self.features(x), (1, 1))
        return self.classifier(torch.flatten(pooled, 1))

    model.forward = types.MethodType(forward, model)
    return model.to(DEVICE).eval(), labels


def make_cam_extractors(model, methods):
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM

    classes = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus,
               "layercam": LayerCAM, "scorecam": ScoreCAM}
    # Same layer the production Grad-CAM hooks: DenseNet121's final feature block.
    extractors = {m: classes[m](model=model, target_layers=[model.features]) for m in methods}
    if "scorecam" in extractors:
        extractors["scorecam"].batch_size = 64
    return extractors


def run_cam(extractor, tensor, class_idx):
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = extractor(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    cam = np.maximum(cam.astype(np.float32), 0)
    return cam / cam.max() if cam.max() > 0 else cam


def split_lookup():
    lookup = {}
    for split in ("train", "val", "test"):
        path = os.path.join(DATA_DIR, f"{split}.csv")
        if os.path.exists(path):
            frame = pd.read_csv(path)
            col = next(c for c in frame.columns if "image" in c.lower() or "index" in c.lower())
            lookup.update({name: split for name in frame[col]})
    return lookup


def zone_metrics(cam, box, zones):
    overlap = bbox_zone_overlap(box, zones)
    act = zone_activation(cam, zones)
    share = zone_cam_share(cam, zones)
    box_zone = max(overlap, key=overlap.get)
    ranked = sorted(act, key=act.get, reverse=True)
    peak = ranked[0]
    valid = cam.sum() > 0
    return {
        "box_zone": box_zone,
        "box_in_zones": sum(overlap.values()),
        "peak_zone": peak if valid else "none",
        "zone_hit": int(valid and peak == box_zone),
        "zone_hit_any": int(valid and overlap[peak] >= ANY_ZONE_MIN_OVERLAP),
        "zone_top2": int(valid and box_zone in ranked[:2]),
        "cam_share_box_zone": share[box_zone],
        "cam_in_zones": sum(share.values()),
        "zci": compute_zci([{"activation": v} for v in act.values()]),
    }


def box_metrics(cam, box):
    """An all-zero CAM (no positive class evidence) scores as a miss everywhere."""
    total = float(cam.sum())
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    _, af, _, thr = compute_asfi(cam, 0.0)
    pred = cam > thr
    union = (pred | box).sum()
    return {
        "cam_empty": int(total <= 0),
        "pointing": int(total > 0 and box[peak_y, peak_x]),
        "energy_in_box": float(cam[box].sum()) / total if total > 0 else 0.0,
        "iou_otsu": float((pred & box).sum()) / union if union else 0.0,
    }


def evaluate(args):
    model, labels = load_model(args.model, args.labels)
    methods = args.methods.split(",")
    extractors = make_cam_extractors(model, methods)
    splits = split_lookup()
    grid = grid_zones()

    boxes = pd.read_csv(os.path.join(DATA_DIR, "BBox_List_2017.csv"))
    boxes = boxes.iloc[:, :6]
    boxes.columns = ["image", "label", "x", "y", "w", "h"]
    boxes["label"] = boxes["label"].str.strip().replace({"Infiltration": "Infiltrate"})
    boxes = boxes[boxes["label"].isin(labels)].reset_index(drop=True)
    if args.limit:
        boxes = boxes.groupby("label", group_keys=False).head(max(1, args.limit // len(labels)))

    rows, seg_failed, t0 = [], 0, time.time()
    for i, b in enumerate(boxes.itertuples()):
        with Image.open(os.path.join(IMAGE_DIR, b.image)) as img:
            gray = np.asarray(img.convert("L"))
            rgb = img.convert("RGB")
        W, H = rgb.size
        tensor = EVAL_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)
        cls = labels.index(b.label)
        with torch.no_grad():
            prob = float(torch.sigmoid(model(tensor))[0, cls])

        right, left = segment_lungs(gray, cache_key=b.image)
        anat, ok = anatomical_zones(right, left)
        seg_failed += int(not ok)
        box = bbox_mask(b.x, b.y, b.w, b.h, W, H)

        for method in methods:
            cam = run_cam(extractors[method], tensor, cls)
            asfi, af, _, _ = compute_asfi(cam, prob)
            row = {"box_id": b.Index, "image": b.image, "label": b.label,
                   "split": splits.get(b.image, "unknown"), "method": method,
                   "probability": prob, "asfi": asfi, "activated_fraction": af,
                   "box_area_frac": float(box.mean()), "seg_ok": int(ok)}
            row.update(box_metrics(cam, box))
            for pname, zones in (("grid", grid), ("anat", anat)):
                if zones is None:
                    continue
                row.update({f"{pname}_{k}": v for k, v in zone_metrics(cam, box, zones).items()})
            rows.append(row)

        if (i + 1) % 25 == 0 or i + 1 == len(boxes):
            rate = (time.time() - t0) / (i + 1)
            print(f"[{i + 1}/{len(boxes)}] {rate:.2f}s/box, ~{rate * (len(boxes) - i - 1) / 60:.1f} min left",
                  flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    details = pd.DataFrame(rows)
    details.to_csv(args.out, index=False)
    print(f"saved {len(details)} rows to {args.out}; lung segmentation failed on {seg_failed} boxes")
    return details


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(details):
    """Print the headline tables and save them as CSVs next to the details file."""
    # Cardiomegaly is a cardiac finding: its box sits on the heart, between the
    # lung zones, so lung-zone hit rates are not meaningful for it.
    lung = details[details["label"] != "Cardiomegaly"]
    subsets = {"all": lung, "held-out (val+test)": lung[lung["split"].isin(["val", "test"])]}

    out = []
    for sname, frame in subsets.items():
        for method, g in frame.groupby("method"):
            n = g["box_id"].nunique()
            rec = {"subset": sname, "method": method, "n_boxes": n,
                   "cam_empty": g["cam_empty"].mean(), "pointing": g["pointing"].mean(), "energy_in_box": g["energy_in_box"].mean(),
                   "iou_otsu": g["iou_otsu"].mean()}
            for p in ("grid", "anat"):
                k = int(g[f"{p}_zone_hit"].sum())
                lo, hi = wilson(k, len(g))
                rec.update({f"{p}_zone_hit": k / len(g), f"{p}_hit_ci": f"[{lo:.3f}, {hi:.3f}]",
                            f"{p}_hit_any": g[f"{p}_zone_hit_any"].mean(),
                            f"{p}_top2": g[f"{p}_zone_top2"].mean(),
                            f"{p}_box_in_zones": g[f"{p}_box_in_zones"].mean()})
            out.append(rec)
    headline = pd.DataFrame(out)

    per_class = (lung.groupby(["method", "label"])
                 .agg(n=("box_id", "nunique"), grid_zone_hit=("grid_zone_hit", "mean"),
                      anat_zone_hit=("anat_zone_hit", "mean"), pointing=("pointing", "mean"),
                      energy_in_box=("energy_in_box", "mean"), anat_box_in_zones=("anat_box_in_zones", "mean"))
                 .reset_index())

    # Prior baseline: always predict the class's most frequent box zone. A CAM
    # that does not beat this is not localising beyond dataset priors.
    prior = []
    first = lung[lung["method"] == lung["method"].iloc[0]]
    for p in ("grid", "anat"):
        hits = sum(g[f"{p}_box_zone"].value_counts().iloc[0] for _, g in first.groupby("label"))
        prior.append({"partition": p, "prior_zone_hit": hits / len(first), "chance_uniform": 1 / 6})
    prior = pd.DataFrame(prior)

    pd.set_option("display.width", 200)
    print("\n=== Headline (lung findings, Cardiomegaly excluded) ===")
    print(headline.round(3).to_string(index=False))
    print("\n=== Per-class zone hit (anatomical vs grid) ===")
    print(per_class.round(3).to_string(index=False))
    print("\n=== Baselines ===")
    print(prior.round(3).to_string(index=False))

    headline.to_csv(os.path.join(OUT_DIR, "zone_eval_headline.csv"), index=False)
    per_class.to_csv(os.path.join(OUT_DIR, "zone_eval_per_class.csv"), index=False)
    prior.to_csv(os.path.join(OUT_DIR, "zone_eval_baselines.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("XRAY_MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--labels", default=os.environ.get("XRAY_LABELS_PATH", DEFAULT_LABELS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--limit", type=int, default=0, help="evaluate ~N boxes, spread over classes")
    parser.add_argument("--out", default=DETAILS_CSV)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    details = pd.read_csv(args.out) if args.summarize_only else evaluate(args)
    summarize(details)


if __name__ == "__main__":
    main()
