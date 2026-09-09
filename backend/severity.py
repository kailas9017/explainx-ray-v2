import numpy as np
from skimage.filters import threshold_otsu


def compute_asfi(cam_224: np.ndarray, probability: float):
    """Return ``(ASFI, activated_fraction, mean_intensity, threshold)``.

    The CAM is expected to be normalized to [0, 1], as returned by
    ``compute_gradcam``.  Otsu is deliberately evaluated independently for
    every scan so activation extent adapts to that scan's CAM distribution.
    """
    flat = np.asarray(cam_224, dtype=np.float32).flatten()
    if flat.size == 0 or flat.max() <= 0:
        return 0.0, 0.0, 0.0, 0.0

    try:
        threshold = float(threshold_otsu(flat))
    except ValueError:
        # Constant CAMs do not have a meaningful foreground/background split.
        return 0.0, 0.0, 0.0, 0.0

    binary_mask = cam_224 > threshold
    af = float(np.mean(binary_mask))
    mi = float(np.mean(cam_224[binary_mask])) if af > 0 else 0.0
    probability = float(np.clip(probability, 0.0, 1.0))
    asfi = 100.0 * (0.40 * af + 0.30 * mi + 0.30 * probability)
    return round(asfi, 1), round(af, 3), round(mi, 3), round(threshold, 3)


def asfi_severity(asfi_score: float) -> str:
    if asfi_score >= 75:
        return "Critical"
    if asfi_score >= 55:
        return "Severe"
    if asfi_score >= 35:
        return "Moderate"
    if asfi_score >= 15:
        return "Mild"
    return "Normal"


def compute_zci(zones: list) -> float:
    """Calculate the six-zone Herfindahl-Hirschman concentration index."""
    if not zones:
        return 0.0
    total = sum(float(zone.get("activation", 0.0)) for zone in zones)
    if total <= 0:
        return round(1 / len(zones), 3)
    shares = [float(zone.get("activation", 0.0)) / total for zone in zones]
    return round(sum(share ** 2 for share in shares), 3)


def distribution_pattern(zci: float) -> str:
    if zci >= 0.55:
        return "Focal"
    if zci >= 0.30:
        return "Multifocal"
    return "Diffuse"
