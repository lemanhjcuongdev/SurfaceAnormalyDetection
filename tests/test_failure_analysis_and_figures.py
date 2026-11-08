"""Case selection and geometry checks for the failure analysis (D-134) and the schematic figures (D-133)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_schematic_figures import field_of_view_cm  # noqa: E402
from run_internal_failure_analysis import outcome, peak, select_figure4, select_figure5  # noqa: E402


def row(image_id, label, d, e_outcome="TN", defect="", threshold=10.0):
    return {"image_id": image_id, "label": label, "defect_type": defect, "D_score": d, "D_margin": d - threshold,
            "D_outcome": outcome(1 if label == "NG" else 0, d, threshold), "E_outcome": e_outcome}


def test_outcome_is_strict_at_the_threshold():
    assert outcome(1, 10.0, 10.0) == "FN"
    assert outcome(0, 10.0, 10.0) == "TN"
    assert outcome(1, 10.1, 10.0) == "TP"
    assert outcome(0, 10.1, 10.0) == "FP"


def test_peak_reports_distance_to_the_nearest_border():
    score_map = np.zeros((16, 16))
    score_map[3, 12] = 1.0
    assert peak(score_map) == (3, 12, 3)


def test_figure4_takes_the_lower_median_per_defect_type_and_the_median_ok():
    rows = [row("a1", "NG", 5, defect="a"), row("a2", "NG", 7, defect="a"),
            row("b1", "NG", 1, defect="b"), row("b2", "NG", 2, defect="b"), row("b3", "NG", 9, defect="b"),
            row("o1", "OK", 3), row("o2", "OK", 4), row("o3", "OK", 8)]
    assert select_figure4(rows) == ["a1", "b2", "o2"]


def test_figure5_order_without_duplicates():
    rows = [row("n1", "NG", 1), row("n2", "NG", 2), row("n3", "NG", 8), row("n4", "NG", 9), row("n5", "NG", 12),
            row("o1", "OK", 9.5, e_outcome="FP"), row("o2", "OK", 7), row("o3", "OK", 2)]
    picks = select_figure5(rows)
    assert [p[0] for p in picks] == ["n1", "n2", "n4", "n3", "o1", "o2"]
    assert len({p[0] for p in picks}) == len(picks)


def test_field_of_view_at_twenty_centimetres():
    short, long = field_of_view_cm(20.0)
    assert short == pytest.approx(7.52, abs=0.01)
    assert long == pytest.approx(10.03, abs=0.01)
    assert long / short == pytest.approx(4 / 3)
