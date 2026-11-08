import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_internal_robustness import geometry, perturb_image, perturb_mask, pixel_counts  # noqa: E402


def frame() -> np.ndarray:
    rgb = np.full((40, 60, 3), 100, dtype=np.uint8)
    rgb[10:20, 20:30] = 200
    return rgb


def test_nominal_condition_returns_the_same_pixels():
    rgb = frame()
    assert geometry({}, 60, 40) is None
    assert np.array_equal(perturb_image(rgb, {}), rgb)
    mask = (rgb[:, :, 0] == 200).astype(np.uint8) * 255
    assert perturb_mask(mask, {}) is mask


def test_gain_scales_and_clips():
    out = perturb_image(frame(), {"gain": 1.5})
    assert out[0, 0, 0] == 150 and out[15, 25, 0] == 255


def test_gain_ramp_follows_the_width():
    out = perturb_image(np.full((4, 11, 3), 100, dtype=np.uint8), {"gain_x": [0.5, 1.5]})
    assert out[0, 0, 0] == 50 and out[0, 5, 0] == 100 and out[0, 10, 0] == 150


def test_shift_moves_image_and_mask_together():
    rgb = frame()
    mask = (rgb[:, :, 0] == 200).astype(np.uint8) * 255
    spec = {"dx": 5, "dy": -3}
    moved, moved_mask = perturb_image(rgb, spec), perturb_mask(mask, spec)
    assert np.array_equal(moved_mask > 0, moved[:, :, 0] == 200)
    assert moved_mask[7, 25] == 255 and moved_mask[10, 20] == 0


def test_lighting_does_not_move_the_mask():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 20:30] = 255
    assert perturb_mask(mask, {"gain": 0.8}) is mask


def test_rotation_keeps_mask_binary():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 20:30] = 255
    assert set(np.unique(perturb_mask(mask, {"rotate_deg": 2}))) <= {0, 255}


def test_pixel_counts_split_defect_and_normal_pixels():
    masks = np.array([[[1, 0], [0, 0]]], dtype=np.uint8)
    maps = np.array([[[0.9, 0.1], [0.2, 0.3]]], dtype=np.float32)
    pos, neg = pixel_counts(masks, maps, np.array([0.5]))
    assert pos.tolist() == [[0, 1]] and neg.tolist() == [[3, 0]]
