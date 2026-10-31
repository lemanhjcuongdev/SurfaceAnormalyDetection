from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "validate_internal_split.py"
FIELDS = [
    "image_id",
    "parent_image_id",
    "sample_id",
    "coupon_id",
    "surface_region_id",
    "capture_session",
    "capture_role",
    "label",
    "data_source",
    "consent_or_data_status",
    "image_sha256",
    "annotation_path",
]


def example_rows() -> list[dict[str, str]]:
    rows = []
    for number, (split, label, session) in enumerate(
        [
            ("model", "OK", "S01"),
            ("calibration", "OK", "S01"),
            ("test", "OK", "S02"),
            ("test", "NG", "S02"),
        ],
        start=1,
    ):
        rows.append({
            "image_id": f"IMG-{number}",
            "parent_image_id": "none",
            "sample_id": f"SAMPLE-{number}",
            "coupon_id": f"COUPON-{number}",
            "surface_region_id": f"REGION-{number}",
            "capture_session": session,
            "capture_role": "dataset_candidate",
            "label": label,
            "data_source": "nonconfidential_coupon",
            "consent_or_data_status": "non-confidential-approved",
            "image_sha256": str(number) * 64,
            "annotation_path": "mask.png" if label == "NG" else "",
            "split": split,
        })
    return rows


def run_validator(rows: list[dict[str, str]]) -> tuple[int, dict[str, object]]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        metadata = root / "metadata.csv"
        assignments = root / "assignments.csv"
        report = root / "result.json"
        with metadata.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with assignments.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_id", "split"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--metadata",
                str(metadata),
                "--assignments",
                str(assignments),
                "--report",
                str(report),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode, json.loads(report.read_text(encoding="utf-8"))


class ValidateInternalSplitTests(unittest.TestCase):
    def test_valid_split_is_ready_without_using_real_images(self) -> None:
        code, result = run_validator(example_rows())
        self.assertEqual(code, 0)
        self.assertEqual(result["integrity_status"], "PASS")
        self.assertTrue(result["experiment_ready"])
        self.assertEqual(result["split_counts"]["test"], 2)
        self.assertEqual(result["unique_active_sample_ids"], 4)

    def test_repeated_physical_sample_cannot_cross_splits(self) -> None:
        rows = example_rows()
        rows[2]["sample_id"] = rows[0]["sample_id"]
        code, result = run_validator(rows)
        self.assertEqual(code, 1)
        self.assertTrue(any("sample_id" in error for error in result["errors"]))

    def test_test_session_must_be_held_out(self) -> None:
        rows = example_rows()
        rows[2]["capture_session"] = "S01"
        code, result = run_validator(rows)
        self.assertEqual(code, 1)
        self.assertTrue(any("Phiên chụp test" in error for error in result["errors"]))

    def test_reference_and_unapproved_images_cannot_enter_model(self) -> None:
        rows = example_rows()
        rows[0]["capture_role"] = "stability_reference"
        rows[0]["consent_or_data_status"] = "pending"
        code, result = run_validator(rows)
        self.assertEqual(code, 1)
        self.assertTrue(any("tham chiếu" in error for error in result["errors"]))
        self.assertTrue(any("được phép" in error for error in result["errors"]))

    def test_registry_confirmed_allowed_status_is_accepted(self) -> None:
        rows = example_rows()
        for row in rows:
            row["consent_or_data_status"] = "confirmed_allowed"
        code, result = run_validator(rows)
        self.assertEqual(code, 0)
        self.assertEqual(result["integrity_status"], "PASS")

    def test_surface_region_can_be_optional_but_still_cannot_cross_splits(self) -> None:
        sys.path.insert(0, str(SCRIPT.parent))
        from validate_internal_split import validate

        rows = example_rows()
        for row in rows:
            row["surface_region_id"] = "TBD"
        assignments = [{"image_id": row["image_id"], "split": row["split"]} for row in rows]
        required = ("sample_id", "capture_session", "data_source")
        self.assertEqual(validate(rows, assignments, required)["integrity_status"], "PASS")
        self.assertEqual(validate(rows, assignments)["integrity_status"], "FAIL")  # default stays strict
        rows[0]["surface_region_id"] = rows[2]["surface_region_id"] = "REGION-X"
        self.assertEqual(validate(rows, assignments, required)["integrity_status"], "FAIL")

    def test_child_and_parent_must_share_split(self) -> None:
        rows = example_rows()
        rows[2]["parent_image_id"] = rows[0]["image_id"]
        code, result = run_validator(rows)
        self.assertEqual(code, 1)
        self.assertTrue(any("ảnh dẫn xuất" in error for error in result["errors"]))

    def test_ng_cannot_enter_normal_only_calibration(self) -> None:
        rows = example_rows()
        rows[1]["label"] = "NG"
        code, result = run_validator(rows)
        self.assertEqual(code, 1)
        self.assertTrue(any("chỉ được chứa ảnh OK" in error for error in result["errors"]))

    def test_duplicate_content_cannot_cross_splits(self) -> None:
        rows = example_rows()
        rows[2]["image_sha256"] = rows[0]["image_sha256"]
        code, result = run_validator(rows)
        self.assertEqual(code, 1)
        self.assertTrue(any("Nội dung ảnh trùng" in error for error in result["errors"]))

    def test_truncated_metadata_row_returns_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata = root / "metadata.csv"
            assignments = root / "assignments.csv"
            report = root / "result.json"
            metadata.write_text(
                ",".join(FIELDS) + "\nIMG-1,none\n",
                encoding="utf-8",
            )
            assignments.write_text("image_id,split\nIMG-1,model\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--metadata",
                    str(metadata),
                    "--assignments",
                    str(assignments),
                    "--report",
                    str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            validation = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(validation["integrity_status"], "FAIL")
            self.assertTrue(any("thiếu sample_id" in error for error in validation["errors"]))


if __name__ == "__main__":
    unittest.main()
