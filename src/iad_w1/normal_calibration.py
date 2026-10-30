"""Pre-registered normal-only image thresholds and spatial post-processing.

No test labels or test scores are accepted by the calibration functions.
The caller is responsible for proving that calibration records come from the
locked normal-only split before invoking these functions.
"""

from __future__ import annotations

import math

import numpy as np


CALIBRATION_QUANTILE = 0.95
SMOOTHING_KERNEL = (1.0, 2.0, 1.0)


def upper_empirical_quantile(values: np.ndarray, quantile: float = CALIBRATION_QUANTILE) -> float:
    """Return the nearest-rank upper quantile, without interpolation.

    For n observations, the selected one-based rank is ceil(q*n). Values
    equal to this threshold remain normal; only a strictly greater score
    triggers an alert. This is an empirical rule, not a future-FPR guarantee.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("Calibration scores must be a nonempty one-dimensional array.")
    if not np.isfinite(sample).all():
        raise ValueError("Calibration scores must be finite.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("Quantile must lie strictly between zero and one.")
    rank = math.ceil(quantile * sample.size)
    return float(np.sort(sample)[rank - 1])


def smooth_map_3x3(anomaly_map: np.ndarray) -> np.ndarray:
    """Apply a fixed separable 3x3 binomial/Gaussian-like kernel.

    Edge replication avoids changing map dimensions and works for maps with
    at least two rows and columns. This is a predeclared method, not tuned on
    test images. Pixel maps are never rescaled or normalized using test data.
    """
    source = np.asarray(anomaly_map, dtype=np.float32)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("Anomaly map must be 2D and at least 2x2.")
    if not np.isfinite(source).all():
        raise ValueError("Anomaly map must contain only finite values.")
    padded_h = np.pad(source, ((0, 0), (1, 1)), mode="edge")
    horizontal = (padded_h[:, :-2] + 2.0 * padded_h[:, 1:-1] + padded_h[:, 2:]) / 4.0
    padded_v = np.pad(horizontal, ((1, 1), (0, 0)), mode="edge")
    return ((padded_v[:-2] + 2.0 * padded_v[1:-1] + padded_v[2:]) / 4.0).astype(
        np.float32
    )


def threshold_from_normal_scores(scores: np.ndarray) -> float:
    """Calibrate variant C from normal-only raw image scores."""
    return upper_empirical_quantile(scores)


def threshold_from_normal_maps(maps: np.ndarray) -> float:
    """Calibrate variant D from maxima of smoothed normal-only maps."""
    maps_array = np.asarray(maps)
    if maps_array.ndim != 3 or maps_array.shape[0] == 0:
        raise ValueError("Calibration maps must have shape [N, H, W].")
    maxima = np.asarray([float(smooth_map_3x3(item).max()) for item in maps_array])
    return upper_empirical_quantile(maxima)


def decisions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return 1 for alert and 0 for OK; scores equal to threshold stay OK."""
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite.")
    return (values > threshold).astype(np.uint8)


def confusion_counts(labels: np.ndarray, predicted: np.ndarray) -> dict[str, int]:
    """Compute image-level counts without changing the locked threshold."""
    true = np.asarray(labels, dtype=np.uint8)
    pred = np.asarray(predicted, dtype=np.uint8)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError("Labels and decisions must have matching one-dimensional shapes.")
    if not np.isin(true, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Labels and decisions must be binary.")
    return {
        "tp": int(np.sum((true == 1) & (pred == 1))),
        "fp": int(np.sum((true == 0) & (pred == 1))),
        "fn": int(np.sum((true == 1) & (pred == 0))),
        "tn": int(np.sum((true == 0) & (pred == 0))),
    }
