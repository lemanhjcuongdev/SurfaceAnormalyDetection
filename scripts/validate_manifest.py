"""Kiểm tra tính toàn vẹn split MVTec W1 và ghi record JSON tái lập."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SPLIT_COUNTS = {"model": 184, "calibration": 46, "test": 117}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root() / "data" / "manifests" / "mvtec_tile_w1_seed42.csv",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    report = (
        args.report.resolve()
        if args.report
        else manifest.with_suffix(".validation.json")
    )
    with manifest.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    errors: list[str] = []
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        errors.append(
            f"Số lượng split là {dict(split_counts)}, kỳ vọng {EXPECTED_SPLIT_COUNTS}."
        )

    paths_by_split: dict[str, set[str]] = {}
    hashes_by_split: dict[str, set[str]] = {}
    for split in EXPECTED_SPLIT_COUNTS:
        split_rows = [row for row in rows if row["split"] == split]
        paths = [row["image_relative_path"] for row in split_rows]
        hashes = [row["image_sha256"] for row in split_rows]
        if len(paths) != len(set(paths)):
            errors.append(f"Đường dẫn ảnh trùng lặp trong split {split}.")
        if len(hashes) != len(set(hashes)):
            errors.append(f"Hash nội dung ảnh trùng lặp trong split {split}.")
        paths_by_split[split] = set(paths)
        hashes_by_split[split] = set(hashes)

    split_names = list(EXPECTED_SPLIT_COUNTS)
    for index, first in enumerate(split_names):
        for second in split_names[index + 1 :]:
            overlapping_paths = paths_by_split[first] & paths_by_split[second]
            overlapping_hashes = hashes_by_split[first] & hashes_by_split[second]
            if overlapping_paths:
                errors.append(
                    f"Đường dẫn chồng chéo giữa {first} và {second}: "
                    f"{sorted(overlapping_paths)[:3]}"
                )
            if overlapping_hashes:
                errors.append(
                    f"Hash nội dung chồng chéo giữa {first} và {second}: "
                    f"{sorted(overlapping_hashes)[:3]}"
                )

    model_or_calibration = [
        row for row in rows if row["split"] in {"model", "calibration"}
    ]
    if any(row["label"] != "OK" or row["label_binary"] != "0" for row in model_or_calibration):
        errors.append("Split model hoặc calibration có dòng không phải OK.")

    test_rows = [row for row in rows if row["split"] == "test"]
    bad_mask_rows = [
        row
        for row in test_rows
        if row["label"] == "NG"
        and (not row["mask_relative_path"] or not row["mask_sha256"])
    ]
    if bad_mask_rows:
        errors.append("Ít nhất một dòng test bất thường thiếu ground-truth mask hash.")

    status = "PASS" if not errors else "FAIL"
    report.parent.mkdir(parents=True, exist_ok=True)
    report_record = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "name": manifest.name,
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        },
        "status": status,
        "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        "observed_split_counts": {
            split: int(split_counts[split]) for split in EXPECTED_SPLIT_COUNTS
        },
        "checks_performed": [
            "unique_image_relative_path_within_each_split",
            "unique_image_sha256_within_each_split",
            "no_image_relative_path_overlap_between_splits",
            "no_image_sha256_overlap_between_splits",
            "model_and_calibration_contain_only_ok",
            "anomalous_test_rows_have_mask_path_and_hash",
        ],
        "test_set_locked_for_configuration_or_threshold_selection": True,
        "errors": errors,
    }
    report.write_text(
        json.dumps(report_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(status)
    print(report)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
