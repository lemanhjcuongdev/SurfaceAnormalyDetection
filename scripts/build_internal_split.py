"""Build the locked internal v1 split (protocol D-096) from the image registry.

Reads `Anh_OK` and `Anh_NG`, assigns every image to model/calibration/test by
physical `sample_id`, hashes the image files, writes metadata/assignment CSVs
and runs `validate_internal_split.validate`. Images are only hashed, never
modified or copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_internal_split import UNSET, validate  # noqa: E402

METADATA_COLUMNS = [
    "image_id", "parent_image_id", "sample_id", "coupon_id", "surface_region_id", "capture_session",
    "capture_role", "label", "data_source", "consent_or_data_status", "image_sha256", "annotation_path",
    "image_path", "registry_sheet", "capture_qc_status",
]


def is_unset(value: Any) -> bool:
    return value is None or str(value).strip().lower() in UNSET


def session_folder(session: str) -> str:
    """`20261029-S08` -> `RESOURCES/29102026` (folder naming used since S04)."""
    date = session.split("-", 1)[0]
    return f"RESOURCES/{date[6:8]}{date[4:6]}{date[0:4]}"


def read_registry(path: Path) -> list[dict[str, Any]]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows: list[dict[str, Any]] = []
    for sheet in ("Anh_OK", "Anh_NG"):
        values = list(workbook[sheet].iter_rows(values_only=True))
        header = [str(cell) if cell is not None else "" for cell in values[0]]
        for raw in values[1:]:
            record = dict(zip(header, raw))
            if is_unset(record.get("image_id")):
                continue
            record["registry_sheet"] = sheet
            rows.append(record)
    return rows


def assign_splits(rows: list[dict[str, Any]], split_config: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Return image_id -> split and a report. Pure function of the registry rows."""
    test_sessions = set(split_config["test_sessions"])
    development_sessions = set(split_config["development_sessions"])
    no_calibration = set(split_config["calibration_excluded_sessions"])
    assignments: dict[str, str] = {}
    report: dict[str, Any] = {"unknown_session_images": [], "conflict_excluded_images": [], "unresolved_sample_images": []}

    test_samples = {
        str(row["sample_id"]).strip()
        for row in rows
        if row["capture_session"] in test_sessions and not is_unset(row["sample_id"])
    }
    development_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row["image_id"])
        session = row["capture_session"]
        sample = None if is_unset(row["sample_id"]) else str(row["sample_id"]).strip()
        if session in test_sessions:
            assignments[image_id] = "test"
        elif session in development_sessions:
            if str(row["label"]).strip().upper() != "OK":
                assignments[image_id] = "excluded"  # NG outside test sessions cannot enter model/calibration
            elif sample is None:
                assignments[image_id] = "model"     # validator will fail on the missing sample_id
                report["unresolved_sample_images"].append(image_id)
            elif sample in test_samples:
                assignments[image_id] = "excluded"
                report["conflict_excluded_images"].append(image_id)
            else:
                development_by_sample[sample].append(row)
        else:
            assignments[image_id] = "excluded"
            report["unknown_session_images"].append(image_id)

    samples = sorted(development_by_sample)
    eligible = [s for s in samples if all(r["capture_session"] not in no_calibration for r in development_by_sample[s])]
    target = min(len(eligible), round(split_config["calibration_fraction_of_samples"] * len(samples)))
    shuffled = list(eligible)
    random.Random(int(split_config["seed"])).shuffle(shuffled)
    calibration = set(shuffled[:target])
    for sample in samples:
        for row in development_by_sample[sample]:
            assignments[str(row["image_id"])] = "calibration" if sample in calibration else "model"
    report.update({
        "development_sample_count": len(samples),
        "calibration_eligible_sample_count": len(eligible),
        "calibration_samples": sorted(calibration),
        "test_sample_count": len(test_samples),
    })
    return assignments, report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_row(row: dict[str, Any], data_root: Path, data_source: str) -> dict[str, str]:
    image_path = data_root / session_folder(row["capture_session"]) / str(row["file_name"])
    if not image_path.is_file():
        raise FileNotFoundError(f"{row['image_id']}: missing image {image_path}")
    def text(key: str) -> str:
        value = row.get(key)
        return "" if value is None else str(value).strip()
    return {
        "image_id": text("image_id"),
        "parent_image_id": "none",
        "sample_id": text("sample_id"),
        "coupon_id": "",
        "surface_region_id": text("surface_region_id"),
        "capture_session": text("capture_session"),
        "capture_role": "dataset_candidate",
        "label": text("label").upper(),
        "data_source": data_source,
        "consent_or_data_status": text("consent_or_data_status"),
        "image_sha256": sha256_file(image_path),
        "annotation_path": text("annotation_path"),
        "image_path": str(image_path),
        "registry_sheet": text("registry_sheet"),
        "capture_qc_status": text("capture_qc_status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v1.yaml")
    parser.add_argument("--data-root", type=Path, required=True, help="HK4 folder containing RESOURCES/<ddmmyyyy>.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    registry_path = (ROOT / config["registry"]).resolve()
    rows = read_registry(registry_path)
    assignments, report = assign_splits(rows, config["split"])
    metadata = [metadata_row(row, args.data_root.resolve(), config["data_source"]) for row in rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.csv"
    assignments_path = args.out_dir / "assignments.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata)
    with assignments_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows({"image_id": row["image_id"], "split": assignments[row["image_id"]]} for row in metadata)

    result = validate(metadata, [{"image_id": k, "split": v} for k, v in assignments.items()])
    result.update({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "D-096",
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(registry_path),
        "metadata_sha256": sha256_file(metadata_path),
        "assignments_sha256": sha256_file(assignments_path),
        "build_report": report,
    })
    (args.out_dir / "split_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    missing = defaultdict(int)
    for error in result["errors"]:
        if ": thiếu " in error:
            missing[error.split(": thiếu ", 1)[1]] += 1
    print(json.dumps({
        "integrity_status": result["integrity_status"],
        "experiment_ready": result["experiment_ready"],
        "split_counts": result["split_counts"],
        "label_counts": result["label_counts"],
        "missing_field_counts": dict(missing),
        "other_errors": [e for e in result["errors"] if ": thiếu " not in e][:5],
        "calibration_samples": report["calibration_samples"],
        "conflict_excluded": len(report["conflict_excluded_images"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
