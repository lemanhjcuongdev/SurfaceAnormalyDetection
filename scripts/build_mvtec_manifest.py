"""Build the fixed W1 MVTec AD tile manifest without copying image data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"
MANIFEST_FIELDS = [
    "record_id",
    "dataset",
    "category",
    "split",
    "image_relative_path",
    "image_sha256",
    "label",
    "label_binary",
    "defect_type",
    "mask_relative_path",
    "mask_sha256",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_tile_root() -> Path:
    return project_root().parents[1] / "RESOURCES" / "MVTec AD" / "tile"


def relative_to_tile(path: Path, tile_root: Path) -> str:
    return path.relative_to(tile_root).as_posix()


def make_record(
    tile_root: Path,
    image_path: Path,
    split: str,
    label: str,
    defect_type: str,
    mask_path: Path | None,
) -> dict[str, str]:
    image_relative_path = relative_to_tile(image_path, tile_root)
    image_hash = sha256_file(image_path)
    record_key = "|".join([split, image_relative_path, image_hash])
    return {
        "record_id": hashlib.sha256(record_key.encode("utf-8")).hexdigest()[:16],
        "dataset": "mvtec_ad",
        "category": "tile",
        "split": split,
        "image_relative_path": image_relative_path,
        "image_sha256": image_hash,
        "label": label,
        "label_binary": "0" if label == "OK" else "1",
        "defect_type": defect_type,
        "mask_relative_path": "" if mask_path is None else relative_to_tile(mask_path, tile_root),
        "mask_sha256": "" if mask_path is None else sha256_file(mask_path),
    }


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )


def build_manifest(tile_root: Path, seed: int) -> tuple[list[dict[str, str]], dict[str, Any]]:
    train_good_dir = tile_root / "train" / "good"
    test_dir = tile_root / "test"
    ground_truth_dir = tile_root / "ground_truth"
    if not train_good_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Expected MVTec tile directories were not found under "
            f"{tile_root}. Expected train/good and test."
        )

    train_good = image_files(train_good_dir)
    shuffled = train_good.copy()
    random.Random(seed).shuffle(shuffled)
    calibration_count = round(len(shuffled) * 0.20)
    calibration = sorted(shuffled[:calibration_count])
    model = sorted(shuffled[calibration_count:])

    rows: list[dict[str, str]] = []
    for image_path in model:
        rows.append(make_record(tile_root, image_path, "model", "OK", "good", None))
    for image_path in calibration:
        rows.append(make_record(tile_root, image_path, "calibration", "OK", "good", None))

    for defect_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        for image_path in image_files(defect_dir):
            if defect_type == "good":
                rows.append(make_record(tile_root, image_path, "test", "OK", "good", None))
                continue

            mask_path = ground_truth_dir / defect_type / f"{image_path.stem}_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing expected ground-truth mask for {image_path}: {mask_path}"
                )
            rows.append(make_record(tile_root, image_path, "test", "NG", defect_type, mask_path))

    rows.sort(key=lambda row: (row["split"], row["image_relative_path"]))
    split_counts = Counter(row["split"] for row in rows)
    test_label_counts = Counter(
        row["defect_type"] for row in rows if row["split"] == "test"
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "mvtec_ad",
        "category": "tile",
        "tile_root": str(tile_root.resolve()),
        "seed": seed,
        "split_counts": dict(sorted(split_counts.items())),
        "test_defect_counts": dict(sorted(test_label_counts.items())),
        "row_count": len(rows),
        "split_algorithm": (
            "Sort train/good filenames; deterministic random.Random(seed) shuffle; "
            "first round(20 percent) are calibration; remaining images are model."
        ),
        "test_policy": "Official test paths are copied to manifest only and remain locked.",
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-root", type=Path, default=default_tile_root())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "manifests" / "mvtec_tile_w1_seed42.csv",
    )
    args = parser.parse_args()

    tile_root = args.tile_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, summary = build_manifest(tile_root, args.seed)
    write_csv(output, rows)
    manifest_hash = sha256_file(output)
    summary["manifest_file"] = str(output)
    summary["manifest_sha256"] = manifest_hash

    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

