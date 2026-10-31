"""Check a proposed internal train/calibration/test assignment without reading images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ACTIVE_SPLITS = {"model", "calibration", "test"}
ALL_SPLITS = ACTIVE_SPLITS | {"excluded"}
METADATA_FIELDS = {
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
}
ASSIGNMENT_FIELDS = {"image_id", "split"}
# `confirmed_allowed` is the registry value for images the researcher confirmed
# as inside the company-permitted scope (D-095).
APPROVED_STATUSES = {"non-confidential-approved", "factory-approved", "confirmed_allowed"}
UNSET = {"", "none", "not_applicable", "unknown", "tbd", "pending"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            return [], [f"{path.name}: thiếu cột {', '.join(missing)}"]
        rows: list[dict[str, str]] = []
        errors: list[str] = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                errors.append(f"{path.name}: dòng {line_number} có nhiều ô hơn tiêu đề")
            rows.append({
                key: value or ""
                for key, value in row.items()
                if key is not None
            })
    return rows, errors


def group_overlap(
    rows: list[dict[str, str]], assignments: dict[str, str], field: str
) -> dict[str, list[str]]:
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = assignments.get(row["image_id"])
        value = row[field].strip()
        if split in ACTIVE_SPLITS and value.lower() not in UNSET:
            by_group[value].add(split)
    return {
        key: sorted(splits)
        for key, splits in by_group.items()
        if len(splits) > 1
    }


DEFAULT_REQUIRED_FIELDS = ("sample_id", "surface_region_id", "capture_session", "data_source")


def validate(
    metadata: list[dict[str, str]],
    assignment_rows: list[dict[str, str]],
    required_fields: tuple[str, ...] = DEFAULT_REQUIRED_FIELDS,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    metadata_by_id: dict[str, dict[str, str]] = {}
    assignments: dict[str, str] = {}

    for row in metadata:
        image_id = row["image_id"].strip()
        if not image_id or image_id in metadata_by_id:
            errors.append(f"image_id trống hoặc trùng: {image_id!r}")
        else:
            metadata_by_id[image_id] = row

    for row in assignment_rows:
        image_id = row["image_id"].strip()
        split = row["split"].strip().lower()
        if not image_id or image_id in assignments:
            errors.append(f"image_id phân tập trống hoặc trùng: {image_id!r}")
            continue
        if split not in ALL_SPLITS:
            errors.append(f"{image_id}: split không hợp lệ: {split!r}")
        assignments[image_id] = split

    missing_assignments = sorted(set(metadata_by_id) - set(assignments))
    extra_assignments = sorted(set(assignments) - set(metadata_by_id))
    if missing_assignments:
        errors.append(f"Ảnh thiếu phân tập: {missing_assignments[:5]}")
    if extra_assignments:
        errors.append(f"Phân tập có ảnh không nằm trong metadata: {extra_assignments[:5]}")

    active_rows = [
        row for row in metadata if assignments.get(row["image_id"]) in ACTIVE_SPLITS
    ]
    for row in active_rows:
        image_id = row["image_id"].strip()
        split = assignments[image_id]
        for field in required_fields:
            if row[field].strip().lower() in UNSET:
                errors.append(f"{image_id}: thiếu {field}")
        if row["capture_role"].strip() != "dataset_candidate":
            errors.append(f"{image_id}: ảnh tham chiếu/QC không được vào {split}")
        if row["consent_or_data_status"].strip() not in APPROVED_STATUSES:
            errors.append(f"{image_id}: chưa có trạng thái dữ liệu được phép sử dụng")
        image_hash = row["image_sha256"].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            errors.append(f"{image_id}: image_sha256 không hợp lệ")
        label = row["label"].strip().upper()
        if split in {"model", "calibration"} and label != "OK":
            errors.append(f"{image_id}: {split} chỉ được chứa ảnh OK")
        if split == "test" and label not in {"OK", "NG"}:
            errors.append(f"{image_id}: test phải có nhãn OK hoặc NG đã xác nhận")

        parent_id = row["parent_image_id"].strip()
        if parent_id.lower() not in UNSET:
            if parent_id not in metadata_by_id:
                errors.append(f"{image_id}: thiếu ảnh nguồn {parent_id}")
            elif assignments.get(parent_id) != split:
                errors.append(f"{image_id}: ảnh dẫn xuất khác split với ảnh nguồn {parent_id}")

    for field in ("sample_id", "coupon_id", "surface_region_id"):
        overlaps = group_overlap(active_rows, assignments, field)
        if overlaps:
            errors.append(f"{field} xuất hiện ở nhiều split: {list(overlaps.items())[:5]}")

    active_hashes: dict[str, set[str]] = defaultdict(set)
    for row in active_rows:
        active_hashes[row["image_sha256"].strip().lower()].add(assignments[row["image_id"]])
    hash_overlap = [value for value, splits in active_hashes.items() if len(splits) > 1]
    if hash_overlap:
        errors.append(f"Nội dung ảnh trùng giữa các split: {hash_overlap[:5]}")

    # The test session is held out. Model and calibration may share a session,
    # but only when their physical sample IDs are disjoint.
    sessions_by_split: dict[str, set[str]] = defaultdict(set)
    for row in active_rows:
        sessions_by_split[assignments[row["image_id"]]].add(row["capture_session"].strip())
    test_session_overlap = sessions_by_split["test"] & (
        sessions_by_split["model"] | sessions_by_split["calibration"]
    )
    if test_session_overlap:
        errors.append(f"Phiên chụp test cũng có ở model/calibration: {sorted(test_session_overlap)}")

    sources = {row["data_source"].strip() for row in active_rows}
    if len(sources) > 1:
        errors.append(f"Không trộn nguồn dữ liệu trong một phép chia: {sorted(sources)}")

    split_counts = Counter(assignments.get(row["image_id"]) for row in metadata)
    label_counts = {
        split: dict(sorted(Counter(
            row["label"].strip().upper()
            for row in active_rows if assignments[row["image_id"]] == split
        ).items()))
        for split in sorted(ACTIVE_SPLITS)
    }
    test_ng = [
        row for row in active_rows
        if assignments[row["image_id"]] == "test" and row["label"].strip().upper() == "NG"
    ]
    masks_available = sum(bool(row["annotation_path"].strip()) for row in test_ng)
    if masks_available < len(test_ng):
        warnings.append("Một số ảnh NG test chưa có đường dẫn mask; chưa thể đánh giá định vị đầy đủ.")

    integrity_status = "PASS" if not errors else "FAIL"
    experiment_ready = (
        integrity_status == "PASS"
        and all(split_counts[split] > 0 for split in ACTIVE_SPLITS)
        and label_counts["test"].get("OK", 0) > 0
        and label_counts["test"].get("NG", 0) > 0
    )
    if not experiment_ready and integrity_status == "PASS":
        warnings.append("Phép chia chưa đủ model, calibration và test OK/NG để tính chỉ số.")

    return {
        "integrity_status": integrity_status,
        "experiment_ready": experiment_ready,
        "split_counts": {split: split_counts[split] for split in sorted(ALL_SPLITS)},
        "label_counts": label_counts,
        "unique_active_sample_ids": len({row["sample_id"].strip() for row in active_rows}),
        "unique_active_surface_regions": len(
            {row["surface_region_id"].strip() for row in active_rows}
        ),
        "test_ng_with_mask_path": masks_available,
        "test_ng_total": len(test_ng),
        "errors": errors,
        "warnings": warnings,
        "limitations": [
            "Không chứng minh mã vật mẫu phản ánh vật thể độc lập ngoài thực địa.",
            "Không kiểm tra chồng lấp vùng bề mặt hoặc nội dung pixel của mask.",
            "PASS chỉ xác nhận quy tắc chia tập từ metadata; không xác nhận quyền dữ liệu hay hiệu năng mô hình.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    assignments_path = args.assignments.resolve()
    report_path = args.report.resolve() if args.report else assignments_path.with_suffix(".validation.json")
    metadata, metadata_errors = read_csv(metadata_path, METADATA_FIELDS)
    assignment_rows, assignment_errors = read_csv(assignments_path, ASSIGNMENT_FIELDS)
    if metadata_errors or assignment_errors:
        result: dict[str, object] = {
            "integrity_status": "FAIL",
            "experiment_ready": False,
            "errors": metadata_errors + assignment_errors,
        }
    else:
        result = validate(metadata, assignment_rows)

    result.update({
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_csv": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "assignments_csv": str(assignments_path),
        "assignments_sha256": sha256_file(assignments_path),
        "policy": "group_by_physical_sample_and_region; hold_out_test_session; normal_only_model_cal",
    })
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["integrity_status"])
    print(report_path)
    if result["integrity_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
