from __future__ import annotations

import unittest

import numpy as np

from iad_w1.normal_calibration import (
    confusion_counts,
    decisions,
    smooth_map_3x3,
    threshold_from_normal_maps,
    threshold_from_normal_scores,
    upper_empirical_quantile,
)


class NormalCalibrationTests(unittest.TestCase):
    def test_nearest_rank_and_strict_alert_rule(self) -> None:
        sample = np.arange(1, 47, dtype=np.float64)
        threshold = threshold_from_normal_scores(sample)
        self.assertEqual(threshold, 44.0)  # ceil(0.95 * 46) = 44
        self.assertEqual(decisions(np.array([43, 44, 45]), threshold).tolist(), [0, 0, 1])

    def test_quantile_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            upper_empirical_quantile(np.array([]))
        with self.assertRaises(ValueError):
            upper_empirical_quantile(np.array([1.0, np.nan]))
        with self.assertRaises(ValueError):
            upper_empirical_quantile(np.array([1.0]), quantile=1.0)

    def test_filter_preserves_constant_map_and_reduces_impulse(self) -> None:
        constant = np.full((5, 5), 7.0, dtype=np.float32)
        np.testing.assert_allclose(smooth_map_3x3(constant), constant)
        impulse = np.zeros((5, 5), dtype=np.float32)
        impulse[2, 2] = 16.0
        filtered = smooth_map_3x3(impulse)
        self.assertEqual(float(filtered.max()), 4.0)
        self.assertEqual(float(filtered.sum()), 16.0)

    def test_d_threshold_uses_only_supplied_normal_maps(self) -> None:
        maps = np.stack([np.full((2, 2), float(i)) for i in range(1, 47)])
        self.assertEqual(threshold_from_normal_maps(maps), 44.0)

    def test_confusion_counts(self) -> None:
        observed = confusion_counts(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
        self.assertEqual(observed, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})


if __name__ == "__main__":
    unittest.main()
