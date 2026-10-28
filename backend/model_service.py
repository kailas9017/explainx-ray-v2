import os
import io
import json

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from config import MODEL_PATH, LABELS_PATH, DEVICE, DEFAULT_PATHOLOGIES, MODEL_DIR
from model import build_model
from gradcam_utils import compute_gradcam, zone_breakdown, make_overlay_base64
from ood_detection import check_out_of_distribution
from severity import compute_asfi, asfi_severity, compute_zci, distribution_pattern
from report_generator import generate_clinical_report

EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ANATOMICAL_REGION = {
    "Atelectasis": "Basal lung fields",
    "Cardiomegaly": "Cardiac silhouette",
    "Effusion": "Costophrenic angle / pleural base",
    "Infiltrate": "Bilateral parenchyma",
    "Mass": "Focal parenchymal region",
    "Nodule": "Focal pulmonary parenchyma",
    "Pneumonia": "Lobar / bronchovascular parenchyma",
    "Pneumothorax": "Pleural space / lung margin",
}

OOD_STATS_PATH = os.path.join(MODEL_DIR, "ood_stats.npz")

class ModelNotReadyError(Exception):
    """Raised when best_model.pth / labels.json are missing or inconsistent.
    The API layer turns this into a clear 503 response — never into fake data."""


def _load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"No trained model found at {MODEL_PATH}. Run backend/train.py first."
    if not os.path.exists(LABELS_PATH):
        return None, None, (
            f"Found {MODEL_PATH} but no matching {LABELS_PATH}. "
            "A checkpoint without its label manifest can silently mismatch classes — "
            "refusing to load. Retrain with train.py, which saves both together."
        )

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)["pathologies"]

    model = build_model(num_classes=len(labels), pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Fail loudly and specifically if shapes don't match, instead of the raw
    # PyTorch RuntimeError — this is exactly the bug that broke the last build.
    checkpoint_out_features = state_dict["classifier.weight"].shape[0]
    if checkpoint_out_features != len(labels):
        return None, None, (
            f"Checkpoint classifier has {checkpoint_out_features} outputs but "
            f"labels.json lists {len(labels)} pathologies. These must always be "
            "saved together — retrain with train.py."
        )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, labels, None


_MODEL, _LABELS, _LOAD_ERROR = _load_model_and_labels()

MODEL_STATUS = {
    "ready": _MODEL is not None,
    "numClasses": len(_LABELS) if _LABELS else None,
    "pathologies": _LABELS or DEFAULT_PATHOLOGIES,
    "device": str(DEVICE),
    "error": _LOAD_ERROR,
}

if _MODEL is not None:
    print(f"[model_service] Loaded model with {len(_LABELS)} classes on {DEVICE}: {_LABELS}")
else:
    print(f"[model_service] Model not ready: {_LOAD_ERROR}")


def analyze_image_bytes(image_bytes: bytes):
    if _MODEL is None:
        raise ModelNotReadyError(_LOAD_ERROR)

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pil_img_224 = pil_img.resize((224, 224))
    img_tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)

    penultimate_features = []

    def capture_features(_module, inputs, _output):
        penultimate_features.append(inputs[0].detach().cpu().numpy())

    # Capture this batch's classifier input during the existing forward pass;
    # OOD checking must not add a second model inference.
    handle = _MODEL.classifier.register_forward_hook(capture_features)
    try:
        with torch.no_grad():
            logits = _MODEL(img_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
    finally:
        handle.remove()

    feature_vec = penultimate_features[0][0]
    is_ood, ood_reason, ood_details = check_out_of_distribution(pil_img, feature_vec, OOD_STATS_PATH)
    if is_ood:
        return {
            "modelReady": True,
            "outOfDistribution": True,
            "outOfDistributionReason": ood_reason,
            "outOfDistributionDetails": ood_details,
            "findings": [],
            "heatmapOverlay": None,
            "primaryZones": [],
            "clinicalReport": None,
        }

    disease_prob_map = {name: round(float(p), 4) for name, p in zip(_LABELS, probs)}
    sorted_findings = sorted(disease_prob_map.items(), key=lambda x: x[1], reverse=True)

    top_prob = sorted_findings[0][1]
    threshold = max(0.25, top_prob * 0.45)
    # Do not force the top class into the report when every class is below the
    # minimum confidence threshold. That creates misleading findings for
    # unrelated images (for example, a leg X-ray classified as 6% Effusion).
    top_findings = [f for f in sorted_findings if f[1] >= threshold][:3]

    findings_list = []
    primary_cam = None

    for idx, (name, prob) in enumerate(top_findings):
        class_idx = _LABELS.index(name)
        cam_224 = compute_gradcam(_MODEL, img_tensor, class_idx)
        if idx == 0:
            primary_cam = cam_224

        zones = zone_breakdown(cam_224)
        top_zone = max(zones, key=lambda z: z["activation"])
        asfi_score, af, mi, otsu_threshold = compute_asfi(cam_224, prob)
        zci = compute_zci(zones)
        pattern = distribution_pattern(zci)

        findings_list.append({
            "id": f"f-{idx + 1}",
            "name": name,
            "probability": prob,
            "severity": asfi_severity(asfi_score),
            "anatomicalRegion": ANATOMICAL_REGION.get(name, "Pulmonary field"),
            "primaryZone": top_zone["zone"],
            "zones": zones,
            "asfi": {
                "score": asfi_score,
                "activatedFraction": af,
                "meanIntensity": mi,
                "otsuThreshold": otsu_threshold,
            },
            "distribution": {"zci": zci, "pattern": pattern},
        })

    overlay_b64 = make_overlay_base64(pil_img_224, primary_cam) if primary_cam is not None else None

    clinical_report = generate_clinical_report(findings_list, {"numClasses": len(_LABELS)})
    return {
        "modelReady": True,
        "numClasses": len(_LABELS),
        "pathologiesScreened": _LABELS,
        "allProbabilities": disease_prob_map,
        "findings": findings_list,
        "heatmapOverlay": overlay_b64,
        "primaryZones": zone_breakdown(primary_cam) if primary_cam is not None else [],
        "clinicalReport": clinical_report,
    }
