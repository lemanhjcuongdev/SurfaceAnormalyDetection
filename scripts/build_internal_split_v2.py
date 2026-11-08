"""Build the internal v2 split (protocol D-126): the v1 assignments minus relabelled NG images.

Sample assignments are pinned to v1, so nothing is reshuffled when a sample loses
all its images. Images on the relabelled list become `excluded`; every other image
must keep its v1 split, label, sample, session and file hash. Labels on the list
win over the registry, so the result is the same before and after the researcher
moves those rows from `Anh_OK` to `Anh_NG`. Images are only hashed, never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_internal_split import METADATA_COLUMNS, metadata_row, read_registry, sha256_file  # noqa: E402
from validate_internal_split import validate  # noqa: E402

COMPARED_WITH_V1 = ("sample_id", "capture_session", "label", "image_sha256")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relabel_rows(rows: list[dict[str, Any]], relabelled: set[str]) -> list[dict[str, Any]]:
    """Apply the relabelled list; the sheet-specific fields become those of an `Anh_NG` row."""
    out = []
    for row in rows:
        record = dict(row)
        if str(record["image_id"]) in relabelled:
            record.update({"label": "NG", "registry_sheet": "Anh_NG", "capture_qc_status": ""})
        out.append(record)
    return out


def pin_assignments(
    rows: list[dict[str, Any]],
    v1_splits: dict[str, str],
    relabelled: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return image_id -> split and a report. `rows` must already carry the relabelled labels."""
    ids = {str(row["image_id"]) for row in rows}
    if ids != set(v1_splits):
        raise SystemExit(f"Registry and v1 differ: {sorted(ids ^ set(v1_splits))[:5]}")
    unknown = sorted(relabelled - ids)
    if unknown:
        raise SystemExit(f"Relabelled images are not in the registry: {unknown[:5]}")
    outside = sorted(i for i in relabelled if v1_splits[i] not in {"model", "calibration"})
    if outside:
        raise SystemExit(f"Relabelled images must come from model/calibration: {outside[:5]}")

    assignments = {}
    for row in rows:
        image_id, split = str(row["image_id"]), v1_splits[str(row["image_id"])]
        if image_id in relabelled:
            split = "excluded"
        elif split in {"model", "calibration"} and str(row["label"]).strip().upper() != "OK":
            raise SystemExit(f"{image_id}: NG in {split} but not on the relabelled list; update the list first.")
        assignments[image_id] = split
    samples = {
        name: sorted({str(row["sample_id"]).strip() for row in rows if assignments[str(row["image_id"])] == name})
        for name in ("model", "calibration", "test")
    }
    report = {
        "excluded_images": sorted(relabelled),
        "calibration_samples": samples["calibration"],
        "model_sample_count": len(samples["model"]),
        "test_sample_count": len(samples["test"]),
    }
    return assignments, report


def check_against_v1(metadata: list[dict[str, str]], v1_metadata: list[dict[str, str]], relabelled: set[str]) -> list[str]:
    """Everything except the labels of relabelled images must equal the v1 manifest."""
    v1 = {row["image_id"]: row for row in v1_metadata}
    differences = []
    for row in metadata:
        for field in COMPARED_WITH_V1:
            if field == "label" and row["image_id"] in relabelled:
                continue
            if row[field].strip() != v1[row["image_id"]][field].strip():
                differences.append(f"{row['image_id']}: {field} differs from v1")
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v2.yaml")
    parser.add_argument("--data-root", type=Path, required=True, help="HK4 folder containing RESOURCES/<ddmmyyyy>.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v2")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pinned = ROOT / config["split"]["pinned_from"]
    v1_validation = json.loads((pinned / "split_validation.json").read_text(encoding="utf-8"))
    if v1_validation["assignments_sha256"] != sha256_file(pinned / "assignments.csv") or \
            v1_validation["metadata_sha256"] != sha256_file(pinned / "metadata.csv"):
        raise SystemExit("v1 manifest changed after validation.")
    v1_splits = {row["image_id"]: row["split"] for row in read_csv_rows(pinned / "assignments.csv")}
    list_path = ROOT / config["split"]["relabelled_list"]
    relabelled = {row["image_id"] for row in read_csv_rows(list_path)}

    registry_path = (ROOT / config["registry"]).resolve()
    rows = relabel_rows(read_registry(registry_path), relabelled)
    assignments, report = pin_assignments(rows, v1_splits, relabelled)
    metadata = [metadata_row(row, args.data_root.resolve(), config["data_source"]) for row in rows]
    differences = check_against_v1(metadata, read_csv_rows(pinned / "metadata.csv"), relabelled)
    if differences:
        raise SystemExit("Manifest differs from v1:\n" + "\n".join(differences[:20]))
    metadata.sort(key=lambda row: row["image_id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, assignments_path = args.out_dir / "metadata.csv", args.out_dir / "assignments.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata)
    with assignments_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "split"])
        writer.writeheader()
        writer.writerows({"image_id": row["image_id"], "split": assignments[row["image_id"]]} for row in metadata)

    required = tuple(dict.fromkeys([*config["split"]["required_fields"], "data_source"]))
    result = validate(metadata, [{"image_id": k, "split": v} for k, v in assignments.items()], required)
    result.update({
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"],
        "config_sha256": sha256_file(args.config),
        "registry_sha256": sha256_file(registry_path),
        "metadata_sha256": sha256_file(metadata_path),
        "assignments_sha256": sha256_file(assignments_path),
        "pinned_v1_assignments_sha256": v1_validation["assignments_sha256"],
        "relabelled_list_sha256": sha256_file(list_path),
        "build_report": report,
    })
    (args.out_dir / "split_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("integrity_status", "experiment_ready", "split_counts", "label_counts",
                                                   "test_ng_with_mask_path", "errors", "warnings", "metadata_sha256")},
                     ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ("calibration_samples", "model_sample_count", "test_sample_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
