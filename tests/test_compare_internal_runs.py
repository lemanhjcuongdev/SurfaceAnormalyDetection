"""Paired run comparison: signs, own thresholds and zero difference on identical runs (no model, no images)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_internal_runs import image_differences  # noqa: E402

BOOT = (200, 0.95, 42)


def toy():
    labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
    weak = np.asarray([0.1, 0.2, 0.3, 0.9, 0.5, 0.6, 0.4, 0.7, 0.8, 0.95])
    strong = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    return labels, groups, weak, strong


def run(scores: np.ndarray, threshold: float) -> dict:
    return {"scores": {"D": scores}, "thresholds": {"D": threshold}}


def row(rows: list[dict], metric: str) -> dict:
    return next(r for r in rows if r["metric"] == metric)


def test_identical_runs_give_zero_differences():
    labels, groups, _, strong = toy()
    rows = image_differences(labels, groups, run(strong, 0.65), run(strong, 0.65), BOOT)
    assert {r["metric"] for r in rows} == {"image_auroc", "image_auprc", "recall", "fpr", "f1"}
    for r in rows:
        assert r["difference"] == pytest.approx(0.0)
        assert r["ci_low"] == pytest.approx(0.0) and r["ci_high"] == pytest.approx(0.0)


def test_difference_is_later_minus_baseline():
    labels, groups, weak, strong = toy()
    rows = image_differences(labels, groups, run(strong, 0.65), run(weak, 0.65), BOOT)
    assert row(rows, "image_auroc")["difference"] > 0
    assert row(rows, "fpr")["difference"] == pytest.approx(0 - 1 / 6)


def test_each_run_uses_its_own_threshold():
    labels, groups, _, strong = toy()
    rows = image_differences(labels, groups, run(strong, 0.65), run(strong, 0.85), BOOT)
    assert row(rows, "recall")["difference"] == pytest.approx(1.0 - 0.5)
    assert row(rows, "image_auroc")["difference"] == pytest.approx(0.0)


def test_variant_without_a_threshold_reports_ranking_only():
    labels, groups, _, strong = toy()
    later = {"scores": {"B": strong}, "thresholds": {}}
    rows = image_differences(labels, groups, later, later, BOOT)
    assert {r["metric"] for r in rows} == {"image_auroc", "image_auprc"}
