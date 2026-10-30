"""Build the technical input manifest for the pooled S04/S05/S06 pilot run.

The script reads image paths and hashes only. It does not copy raw image pixels.
S04 contributes exactly one preselected representative from each three-photo
capture group; S05 and S06 contribute one record per supplied source image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    files = sorted(
        {
            path.resolve()
            for pattern in IMAGE_PATTERNS
            for path in directory.glob(pattern)
            if path.is_file()
        }
    )
    if not files:
        raise ValueError(f"Không có JPEG trong: {directory}")
    return files


def source_folder_name(directory: Path) -> str:
    return f"RESOURCES/{directory.name}"


def row(
    *,
    record_id: str,
    session: str,
    directory: Path,
    source_path: Path,
    group_id: str,
    independence_status: str,
    selection_method: str,
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "capture_session": session,
        "source_folder": source_folder_name(directory),
        "source_image_path": str(source_path),
        "source_image_sha256": sha256_file(source_path),
        "capture_group_id": group_id,
        "independence_status": independence_status,
        "capture_role": "provisional_normal_candidate",
        "label_assumption": "normal_candidate_user_directed",
        "selection_method": selection_method,
        "roi_version": "roi-v0.1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s04-representatives", type=Path, required=True)
    parser.add_argument("--s04-dir", type=Path, required=True)
    parser.add_argument("--s05-dir", type=Path, required=True)
    parser.add_argument("--s06-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    s04_csv = args.s04_representatives.resolve()
    s04_dir = args.s04_dir.resolve()
    s05_dir = args.s05_dir.resolve()
    s06_dir = args.s06_dir.resolve()
    output = args.output.resolve()
    for path in (s04_csv, s04_dir, s05_dir, s06_dir):
        if not path.exists():
            raise FileNotFoundError(f"Không thấy input: {path}")

    with s04_csv.open(newline="", encoding="utf-8") as stream:
        s04_rows = list(csv.DictReader(stream))
    required_s04 = {"capture_group_id", "capture_session", "representative_file"}
    if not s04_rows or not required_s04.issubset(s04_rows[0]):
        raise ValueError(
            "CSV nhóm ảnh thiếu capture_group_id, capture_session hoặc representative_file."
        )
    s04_rows = [
        source
        for source in s04_rows
        if source["capture_session"].strip() == "20261009-S04"
    ]
    if len(s04_rows) != 38:
        raise ValueError(
            f"CSV phải có đúng 38 cụm thuộc phiên 20261009-S04; nhận được {len(s04_rows)}."
        )

    records: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for source in sorted(s04_rows, key=lambda item: item["capture_group_id"]):
        capture_group_id = source["capture_group_id"].strip()
        source_path = (s04_dir / source["representative_file"].strip()).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Không thấy ảnh đại diện S04: {source_path}")
        if source_path in seen_paths:
            raise ValueError(f"Ảnh S04 đại diện bị lặp: {source_path}")
        records.append(
            row(
                record_id=capture_group_id,
                session="20261009-S04",
                directory=s04_dir,
                source_path=source_path,
                group_id=capture_group_id,
                independence_status="user_confirmed_distinct_nonoverlap_region",
                selection_method="one_representative_from_three_repeat_photos",
            )
        )
        seen_paths.add(source_path)

    s05_files = image_files(s05_dir)
    if len(s05_files) != 10:
        raise ValueError(f"S05 phải có 10 ảnh; nhận được {len(s05_files)}.")
    for index, source_path in enumerate(s05_files, start=1):
        record_id = f"S05-R{index:03d}"
        records.append(
            row(
                record_id=record_id,
                session="20261013-S05",
                directory=s05_dir,
                source_path=source_path,
                group_id=record_id,
                independence_status="user_confirmed_distinct_nonoverlap_region",
                selection_method="one_photo_per_user_confirmed_region",
            )
        )
        seen_paths.add(source_path)

    s06_files = image_files(s06_dir)
    if len(s06_files) != 35:
        raise ValueError(f"S06 phải có 35 ảnh; nhận được {len(s06_files)}.")
    for source_path in s06_files:
        record_id = f"S06-{source_path.stem}"
        records.append(
            row(
                record_id=record_id,
                session="20261016-S06",
                directory=s06_dir,
                source_path=source_path,
                group_id="unconfirmed",
                independence_status="unverified_physical_relationship",
                selection_method="all_supplied_images_user_directed",
            )
        )
        seen_paths.add(source_path)

    if len(records) != 83:
        raise RuntimeError(f"Manifest phải có 83 ảnh; nhận được {len(records)}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "raw_images_copied": False,
        "output_manifest": str(output),
        "output_manifest_sha256": sha256_file(output),
        "record_count": len(records),
        "counts_by_session": {"S04": 38, "S05": 10, "S06": 35},
        "selection_policy": {
            "S04": "one representative from each three-photo capture group",
            "S05": "all ten user-confirmed distinct non-overlap regions",
            "S06": "all 35 user-directed normal candidates; physical relation unverified",
        },
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
