"""Anatomy-aware six-zone partition of a chest radiograph.

The production app (``gradcam_utils.zone_breakdown``) splits the 224x224 image
into a fixed 3x2 grid. That grid ignores where the lungs actually are: a
"Right Upper" cell can be mostly shoulder, and a "Lower" cell mostly abdomen.

This module instead segments each lung with the pretrained torchxrayvision
PSPNet (ChestX-Det) model and splits every lung into Upper / Mid / Lower thirds
of its own vertical extent. Zones are boolean masks on the same 224x224 grid as
the Grad-CAM, so both partitions can be scored with identical code.

Side convention matches ``config.ZONE_COLS``: patient's Right is image-left.
torchxrayvision's "Right Lung" target is the patient's right lung.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from config import DATA_DIR, DEVICE, ZONE_COLS, ZONE_ROWS  # noqa: E402

CAM_SIZE = 224
SEG_SIZE = 512
MIN_LUNG_FRACTION = 0.02  # a lung covering <2% of the image is treated as a failed segmentation
HULL_DILATION_PX = 4      # on the 224 grid (~18 px on a 1024 NIH image)
MASK_CACHE_DIR = os.path.join(DATA_DIR, "lung_masks")

_seg_model = None


def _import_xrv():
    """Import torchxrayvision without clashing with backend/model.py.

    torchxrayvision's jfhealthcare sub-package puts its own folder on sys.path
    and does ``from model.utils import ...``; if our ``model`` module is already
    imported it fails, and if xrv's ``model`` package wins, ours is shadowed.
    Hide ours during the import, then restore it and drop xrv's.
    """
    if "torchxrayvision" in sys.modules:
        return sys.modules["torchxrayvision"]
    is_model = lambda k: k == "model" or k.startswith("model.")  # noqa: E731
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if is_model(k)}
    path_before = list(sys.path)
    try:
        import torchxrayvision as xrv
    finally:
        for k in [k for k in sys.modules if is_model(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
        sys.path[:] = path_before
    return xrv


xrv = _import_xrv()


def _get_seg_model():
    global _seg_model
    if _seg_model is None:
        # torchxrayvision prints a unicode progress bar while downloading weights,
        # which crashes on Windows cp1252 consoles.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _seg_model = xrv.baseline_models.chestx_det.PSPNet().to(DEVICE).eval()
    return _seg_model


def _to_gray_uint8(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


@torch.no_grad()
def segment_lungs(img, cache_key=None):
    """Return ``(right_mask, left_mask)`` as 224x224 boolean arrays.

    ``img`` is a grayscale or RGB array at any resolution. When ``cache_key``
    (e.g. the NIH image file name) is given, masks are cached under
    ``data/lung_masks`` so repeated experiments do not re-run segmentation.
    """
    cache_path = None
    if cache_key:
        cache_path = os.path.join(MASK_CACHE_DIR, os.path.splitext(cache_key)[0] + ".npz")
        if os.path.exists(cache_path):
            cached = np.load(cache_path)
            return cached["right"], cached["left"]

    model = _get_seg_model()
    gray = _to_gray_uint8(img).astype(np.float32)
    x = xrv.datasets.normalize(gray, 255)[None]               # [1, H, W] in [-1024, 1024]
    x = torch.from_numpy(x)[None].to(DEVICE)                   # [1, 1, H, W]
    x = F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)
    out = model(x)                                             # [1, 14, 512, 512], sigmoid
    out = F.interpolate(out, size=(CAM_SIZE, CAM_SIZE), mode="bilinear", align_corners=False)[0]

    right = (out[model.targets.index("Right Lung")] > 0.5).cpu().numpy()
    left = (out[model.targets.index("Left Lung")] > 0.5).cpu().numpy()

    if cache_path:
        os.makedirs(MASK_CACHE_DIR, exist_ok=True)
        np.savez_compressed(cache_path, right=right, left=left)
    return right, left


def grid_zones(size=CAM_SIZE):
    """The production 3x2 grid, expressed as masks (same layout as zone_breakdown)."""
    row_edges = np.linspace(0, size, len(ZONE_ROWS) + 1).astype(int)
    col_edges = np.linspace(0, size, len(ZONE_COLS) + 1).astype(int)
    zones = {}
    for ri, row in enumerate(ZONE_ROWS):
        for ci, col in enumerate(ZONE_COLS):
            mask = np.zeros((size, size), dtype=bool)
            mask[row_edges[ri]:row_edges[ri + 1], col_edges[ci]:col_edges[ci + 1]] = True
            zones[f"{col} {row}"] = mask
    return zones


def _split_lung(lung_mask):
    """Split one lung mask into Upper/Mid/Lower thirds of its own height."""
    ys = np.nonzero(lung_mask.any(axis=1))[0]
    top, bottom = ys.min(), ys.max() + 1
    edges = np.linspace(top, bottom, len(ZONE_ROWS) + 1).round().astype(int)
    parts = {}
    for ri, row in enumerate(ZONE_ROWS):
        band = np.zeros_like(lung_mask)
        band[edges[ri]:edges[ri + 1], :] = True
        parts[row] = lung_mask & band
    return parts


def hemithorax_region(lung_mask, dilation_px=HULL_DILATION_PX):
    """Convex hull of a lung mask, slightly dilated.

    Segmentation models trace *aerated* lung, so fluid (effusion), collapsed
    lung (atelectasis) or pleural air (pneumothorax) at the lung edge is often
    left outside the mask — exactly where those findings are annotated. The
    convex hull fills the costophrenic angle and lung margin back in.
    """
    from scipy.ndimage import binary_dilation
    from skimage.morphology import convex_hull_image, disk

    region = convex_hull_image(lung_mask)
    if dilation_px > 0:
        region = binary_dilation(region, structure=disk(dilation_px))
    return region


def anatomical_zones(right_mask, left_mask, use_hull=True):
    """Return ``(zones, ok)``.

    ``zones`` maps "Right Upper" ... "Left Lower" to 224x224 boolean masks.
    With ``use_hull`` (default) each lung is replaced by its dilated convex
    hull (see ``hemithorax_region``); ``use_hull=False`` uses the raw masks.
    ``ok`` is False when either lung segmentation failed; the caller should
    then exclude the image (and report how many were excluded).
    """
    total = right_mask.size
    ok = right_mask.sum() / total >= MIN_LUNG_FRACTION and left_mask.sum() / total >= MIN_LUNG_FRACTION
    if not ok:
        return None, False

    zones = {}
    regions = {"Right": right_mask, "Left": left_mask}
    if use_hull:
        regions = {side: hemithorax_region(m) for side, m in regions.items()}
        # Dilated hulls can touch at the mediastinum; give shared pixels to neither.
        overlap = regions["Right"] & regions["Left"]
        regions = {side: m & ~overlap for side, m in regions.items()}
    for side, mask in regions.items():
        for row, part in _split_lung(mask).items():
            zones[f"{side} {row}"] = part
    # Keep the same ordering as the grid so tables line up.
    ordered = {f"{c} {r}": zones[f"{c} {r}"] for r in ZONE_ROWS for c in ZONE_COLS}
    return ordered, True


def zone_activation(cam, zones):
    """Mean CAM activation (x100) inside each zone mask — same scale as zone_breakdown."""
    return {name: float(cam[m].mean() * 100.0) if m.any() else 0.0 for name, m in zones.items()}


def zone_cam_share(cam, zones):
    """Fraction of total CAM mass falling in each zone (sums to <=1; rest is outside lungs)."""
    total = float(cam.sum())
    if total <= 0:
        return {name: 0.0 for name in zones}
    return {name: float(cam[m].sum()) / total for name, m in zones.items()}


def bbox_mask(x, y, w, h, orig_width, orig_height, size=CAM_SIZE):
    """Rasterise an NIH bounding box (original-pixel coords) onto the CAM grid."""
    sx, sy = size / orig_width, size / orig_height
    x0, y0 = int(np.floor(x * sx)), int(np.floor(y * sy))
    x1, y1 = int(np.ceil((x + w) * sx)), int(np.ceil((y + h) * sy))
    mask = np.zeros((size, size), dtype=bool)
    mask[max(y0, 0):min(y1, size), max(x0, 0):min(x1, size)] = True
    return mask


def bbox_zone_overlap(box_mask, zones):
    """Fraction of the box's area that lies in each zone."""
    area = box_mask.sum()
    if area == 0:
        return {name: 0.0 for name in zones}
    return {name: float((box_mask & m).sum()) / area for name, m in zones.items()}
