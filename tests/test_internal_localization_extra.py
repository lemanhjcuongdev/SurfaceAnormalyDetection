"""IoU, Dice and AU-PRO checks for the post-hoc localization analysis (no model, no images)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_internal_localization_extra import aupro, iou_dice, pixel_counts, pro_curves  # noqa: E402


def toy_masks() -> np.ndarray:
    masks = np.zeros((4, 32, 32), np.uint8)
    masks[0, 2:6, 2:6] = 1  # two regions in one image
    masks[0, 20:24, 20:28] = 1
    masks[1, 10:18, 10:18] = 1
    return masks


def area(masks: np.ndarray, maps: np.ndarray, limit: float) -> float:
    neg, survival, regions = pro_curves(masks, maps, bins=512)
    return aupro(neg.sum(0), survival.sum(0), int(regions.sum()), limit)


def test_pixel_counts_are_strict_at_the_threshold():
    masks = np.array([[[1, 1], [0, 0]]], np.uint8)
    maps = np.array([[[2.0, 1.0], [1.0, 0.0]]])
    counts = pixel_counts(masks, maps, 1.0)
    assert {k: int(v[0]) for k, v in counts.items()} == {"tp": 1, "fp": 0, "fn": 1, "tn": 2}


def test_iou_dice_values_and_empty_case():
    assert iou_dice(2, 1, 1) == pytest.approx((0.5, 2 / 3))
    assert iou_dice(0, 3, 4) == (0.0, 0.0)
    with pytest.raises(ValueError):
        iou_dice(0, 0, 0)


def test_regions_are_eight_connected():
    masks = np.zeros((1, 8, 8), np.uint8)
    masks[0, 1, 1] = masks[0, 2, 2] = 1  # diagonal neighbours form one region
    masks[0, 6, 6] = 1
    _, _, regions = pro_curves(masks, np.random.default_rng(0).random((1, 8, 8)), bins=16)
    assert regions.tolist() == [2]


@pytest.mark.parametrize("limit", [0.30, 0.05])
def test_aupro_is_one_for_a_perfect_map_and_zero_for_an_inverted_one(limit):
    masks = toy_masks()
    noise = np.random.default_rng(0).random(masks.shape)
    assert area(masks, noise + 2 * masks, limit) == pytest.approx(1.0, abs=1e-6)
    assert area(masks, noise - 2 * masks, limit) == pytest.approx(0.0, abs=1e-6)


def test_aupro_of_noise_is_near_half_the_limit():
    masks = toy_masks()
    noise = np.random.default_rng(1).random(masks.shape)
    assert area(masks, noise, 0.30) == pytest.approx(0.15, abs=0.04)


def test_aupro_weights_regions_equally_not_by_area():
    masks = np.zeros((2, 32, 32), np.uint8)
    masks[0, 0:20, 0:20] = 1  # large region missed
    masks[1, 5:7, 5:7] = 1  # small region found
    maps = np.random.default_rng(2).random(masks.shape)
    maps[1] += 2 * masks[1]
    maps[0] -= 2 * masks[0]
    assert area(masks, maps, 0.30) == pytest.approx(0.5, abs=1e-6)


def test_aupro_rejects_a_sample_without_regions():
    with pytest.raises(ValueError):
        aupro(np.ones(8), np.zeros(8), 0, 0.3)
