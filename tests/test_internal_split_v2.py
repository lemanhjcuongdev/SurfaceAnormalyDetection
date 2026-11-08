"""Pinned v2 split: only relabelled images move, and everything else must match v1 (no images)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_internal_split_v2 import check_against_v1, pin_assignments, relabel_rows  # noqa: E402


def registry_rows() -> list[dict[str, object]]:
    def row(image_id: str, sample: str, label: str = "OK", sheet: str = "Anh_OK") -> dict[str, object]:
        return {"image_id": image_id, "sample_id": sample, "label": label, "registry_sheet": sheet, "capture_qc_status": "ok"}

    return [row("m1", "M01"), row("m2", "M01"), row("m3", "M02"), row("c1", "M03"), row("c2", "M03"),
            row("c3", "M04"), row("t1", "M05"), row("t2", "M05", "NG", "Anh_NG")]


V1 = {"m1": "model", "m2": "model", "m3": "model", "c1": "calibration", "c2": "calibration",
      "c3": "calibration", "t1": "test", "t2": "test"}


def build(relabelled: set[str], rows: list[dict[str, object]] | None = None):
    return pin_assignments(relabel_rows(rows or registry_rows(), relabelled), V1, relabelled)


def test_only_relabelled_images_move_and_a_sample_can_empty():
    assignments, report = build({"m3", "c1"})
    assert assignments == {**V1, "m3": "excluded", "c1": "excluded"}
    assert report["model_sample_count"] == 1  # M02 lost its only image; no reshuffle follows
    assert report["calibration_samples"] == ["M03", "M04"]


def test_result_is_the_same_after_rows_move_to_the_ng_sheet():
    moved = registry_rows()
    moved[2].update({"label": "NG", "registry_sheet": "Anh_NG"})
    moved[2].pop("capture_qc_status")
    before = relabel_rows(registry_rows(), {"m3"})
    after = relabel_rows(moved, {"m3"})
    assert {k: before[2][k] for k in ("label", "registry_sheet", "capture_qc_status")} == \
           {k: after[2][k] for k in ("label", "registry_sheet", "capture_qc_status")}
    assert pin_assignments(before, V1, {"m3"})[0] == pin_assignments(after, V1, {"m3"})[0]


def test_unlisted_ng_in_development_stops_the_build():
    rows = registry_rows()
    rows[0]["label"] = "NG"
    with pytest.raises(SystemExit, match="not on the relabelled list"):
        build(set(), rows)


def test_a_listed_test_image_stops_the_build():
    with pytest.raises(SystemExit, match="must come from model/calibration"):
        build({"t1"})


def test_an_unknown_listed_image_stops_the_build():
    with pytest.raises(SystemExit, match="not in the registry"):
        build({"missing"})


def test_a_registry_that_differs_from_v1_stops_the_build():
    with pytest.raises(SystemExit, match="Registry and v1 differ"):
        pin_assignments(registry_rows()[:-1], V1, set())


def test_check_against_v1_ignores_only_the_label_of_listed_images():
    def row(image_id: str, label: str = "OK", digest: str = "a") -> dict[str, str]:
        return {"image_id": image_id, "sample_id": "M01", "capture_session": "S04", "label": label, "image_sha256": digest}

    v1 = [row("x"), row("y")]
    assert check_against_v1([row("x", "NG"), row("y")], v1, {"x"}) == []
    assert check_against_v1([row("x"), row("y", "NG")], v1, {"x"}) == ["y: label differs from v1"]
    assert check_against_v1([row("x", "NG", "b"), row("y")], v1, {"x"}) == ["x: image_sha256 differs from v1"]
