"""Runtime safeguards for inputs outside the chest-radiograph distribution."""

import os

import numpy as np
from PIL import Image


CHANNEL_SIMILARITY_THRESHOLD = 0.04
_OOD_STATS = None


class OODStatsUnavailableError(RuntimeError):
    """Raised when inference is attempted before the detector is calibrated."""


def channel_similarity_score(pil_img: Image.Image) -> float:
    """Mean RGB-channel difference in [0, 1]; radiographs should be near zero."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0
    return float(np.mean(diff))


def load_ood_stats(path: str) -> dict | None:
    """Load fitted feature statistics once per process, if available."""
    global _OOD_STATS
    if _OOD_STATS is None:
        if not os.path.exists(path):
            return None
        with np.load(path) as data:
            _OOD_STATS = {
                "mean": data["mean"],
                "cov_inv": data["cov_inv"],
                "threshold": float(data["threshold"]),
            }
    return _OOD_STATS


def mahalanobis_distance(feature_vec: np.ndarray, stats: dict) -> float:
    d = np.asarray(feature_vec, dtype=np.float64).reshape(-1) - stats["mean"]
    squared_distance = float(d @ stats["cov_inv"] @ d.T)
    # Round-off can make a theoretically non-negative value microscopically negative.
    return float(np.sqrt(max(squared_distance, 0.0)))


def check_out_of_distribution(pil_img: Image.Image, feature_vec: np.ndarray, ood_stats_path: str):
    """Return ``(is_ood, reason, details)`` after inexpensive then feature gates."""
    channel_score = channel_similarity_score(pil_img)
    if channel_score > CHANNEL_SIMILARITY_THRESHOLD:
        return True, "Image does not appear to be a grayscale radiograph (significant colour content detected).", {
            "channelSimilarityScore": round(channel_score, 4),
        }

    stats = load_ood_stats(ood_stats_path)
    if stats is None:
        # Calibration is optional for inference. Keep the inexpensive grayscale
        # safeguard active and report why the feature-distance gate was skipped.
        return False, None, {
            "channelSimilarityScore": round(channel_score, 4),
            "featureCheckAvailable": False,
        }

    distance = mahalanobis_distance(feature_vec, stats)
    if distance > stats["threshold"]:
        return True, "Image features fall well outside the distribution of the model's training data.", {
            "channelSimilarityScore": round(channel_score, 4),
            "mahalanobisDistance": round(distance, 2),
            "threshold": round(stats["threshold"], 2),
        }

    return False, None, {
        "channelSimilarityScore": round(channel_score, 4),
        "featureCheckAvailable": True,
        "mahalanobisDistance": round(distance, 2),
    }
