"""Create provisional visual groups for locally stored capture images.

This utility never copies raw images and never assigns physical ``sample_id``
values or data splits.  It only compares image appearance so that a researcher
can review likely surface families and select one representative from a burst.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


DEFAULT_ROI = (0, 0, 3000, 1957)
OUTPUT_FIELDS = [
    "capture_group_id",
    "capture_session",
    "source_folder",
    "representative_file",
    "files_in_capture_group",
    "representative_selection_method",
    "inferred_visual_family",
    "visual_family_size",
    "nearest_capture_group_id",
    "nearest_visual_similarity",
    "roi_proxy",
    "median_luminance",
    "texture_strength",
    "inference_status",
    "review_note",
]


def parse_roi(value: str) -> tuple[int, int, int, int]:
    """Parse x,y,width,height and reject invalid ROI values."""
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ROI phải có dạng x,y,width,height"
        ) from error
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI phải có tọa độ và kích thước dương")
    return x, y, width, height


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Return a finite unit vector, preserving an all-zero vector."""
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64, copy=False) / norm


def crop_rgb(path: Path, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Read an image locally and return a resized RGB ROI as float data."""
    x, y, width, height = roi
    with Image.open(path) as image:
        # ROI-v0.1 is defined after applying the camera's EXIF orientation.
        # Galaxy S23+ portrait JPEGs are physically stored landscape with
        # Orientation=6, so cropping before this conversion selects wrong pixels.
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        right = min(rgb.width, x + width)
        bottom = min(rgb.height, y + height)
        if right <= x or bottom <= y:
            raise ValueError(f"{path.name}: ROI nằm ngoài ảnh {rgb.width}×{rgb.height}")
        crop = rgb.crop((x, y, right, bottom))
        resized = crop.resize((96, 64), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64) / 255.0


def image_descriptor(path: Path, roi: tuple[int, int, int, int]) -> dict[str, object]:
    """Build an interpretable low-level appearance descriptor without a model."""
    rgb = crop_rgb(path, roi)
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    centered = gray - float(np.median(gray))
    structure = l2_normalize(centered.ravel())

    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dx, dy)
    angle = (np.arctan2(dy, dx) + np.pi) % np.pi
    histogram, _ = np.histogram(
        angle,
        bins=8,
        range=(0.0, np.pi),
        weights=magnitude,
    )
    texture_values = np.concatenate(
        [
            histogram,
            np.array(
                [
                    float(np.mean(magnitude)),
                    float(np.std(magnitude)),
                    float(np.quantile(magnitude, 0.90)),
                ]
            ),
        ]
    )
    texture = l2_normalize(texture_values)

    color_values = np.concatenate(
        [
            np.mean(rgb, axis=(0, 1)),
            np.std(rgb, axis=(0, 1)),
            np.quantile(gray, [0.10, 0.50, 0.90]),
        ]
    )
    appearance = l2_normalize(color_values)

    feature = np.concatenate([0.55 * structure, 0.30 * texture, 0.15 * appearance])
    feature = l2_normalize(feature)
    return {
        "feature": feature,
        "median_luminance": float(np.median(gray) * 255.0),
        "texture_strength": float(np.mean(magnitude)),
        "sharpness": float(np.mean(magnitude**2)),
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity clamped to a stable [0, 1] range."""
    similarity = float(np.dot(left, right))
    return max(0.0, min(1.0, similarity))


def choose_representative(descriptors: list[dict[str, object]]) -> int:
    """Choose the image most central to its burst; use sharpness for ties."""
    if len(descriptors) == 1:
        return 0
    scores: list[tuple[float, float, int]] = []
    for index, descriptor in enumerate(descriptors):
        feature = descriptor["feature"]
        assert isinstance(feature, np.ndarray)
        mean_similarity = float(
            np.mean(
                [
                    cosine_similarity(feature, other["feature"])
                    for other in descriptors
                    if other is not descriptor
                ]
            )
        )
        scores.append((mean_similarity, float(descriptor["sharpness"]), -index))
    return max(range(len(scores)), key=lambda index: scores[index])


def connected_components(
    identifiers: list[str], similarity: np.ndarray, threshold: float
) -> list[list[int]]:
    """Join only groups whose visual similarity clears the declared threshold."""
    del identifiers  # IDs are carried by the caller; positions determine edges.
    parent = list(range(similarity.shape[0]))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(similarity.shape[0]):
        for right in range(left + 1, similarity.shape[0]):
            if float(similarity[left, right]) >= threshold:
                union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(similarity.shape[0]):
        components[find(index)].append(index)
    return sorted(components.values(), key=lambda items: items[0])


def read_capture_groups(
    mapping_paths: list[Path], image_directories: list[Path], roi: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    """Read group mappings and compute a representative descriptor per group."""
    if len(mapping_paths) != len(image_directories):
        raise ValueError("Số --mapping phải bằng số --images-dir")

    groups: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for mapping_path, image_directory in zip(mapping_paths, image_directories):
        with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"capture_group_id", "capture_session", "source_folder", "source_file_1"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{mapping_path.name}: thiếu cột {sorted(missing)}")
            for row in reader:
                group_id = (row.get("capture_group_id") or "").strip()
                if not group_id or group_id in seen_ids:
                    raise ValueError(f"capture_group_id trống hoặc trùng: {group_id!r}")
                filenames = [
                    (row.get(column) or "").strip()
                    for column in ("source_file_1", "source_file_2", "source_file_3")
                    if (row.get(column) or "").strip()
                ]
                if not filenames:
                    raise ValueError(f"{group_id}: không có file ảnh")
                paths = [image_directory / filename for filename in filenames]
                missing_files = [path.name for path in paths if not path.is_file()]
                if missing_files:
                    raise FileNotFoundError(f"{group_id}: thiếu ảnh {missing_files}")
                descriptors = [image_descriptor(path, roi) for path in paths]
                selected_index = choose_representative(descriptors)
                selected_descriptor = descriptors[selected_index]
                groups.append(
                    {
                        "capture_group_id": group_id,
                        "capture_session": (row.get("capture_session") or "").strip(),
                        "source_folder": (row.get("source_folder") or "").strip(),
                        "representative_file": paths[selected_index].name,
                        "files_in_capture_group": len(paths),
                        "feature": selected_descriptor["feature"],
                        "median_luminance": selected_descriptor["median_luminance"],
                        "texture_strength": selected_descriptor["texture_strength"],
                        "selection_method": (
                            "single_image" if len(paths) == 1
                            else "highest_mean_similarity_then_sharpness"
                        ),
                    }
                )
                seen_ids.add(group_id)
    return groups


def infer_rows(groups: list[dict[str, object]], threshold: float, roi: tuple[int, int, int, int]) -> list[dict[str, object]]:
    """Create provisional visual-family rows with nearest-neighbour evidence."""
    if not groups:
        return []
    features = np.vstack([group["feature"] for group in groups])
    similarity = np.clip(features @ features.T, 0.0, 1.0)
    components = connected_components(
        [str(group["capture_group_id"]) for group in groups], similarity, threshold
    )
    family_by_index: dict[int, tuple[str, int]] = {}
    for family_index, component in enumerate(components, start=1):
        family_id = f"VF-{family_index:02d}"
        for index in component:
            family_by_index[index] = (family_id, len(component))

    rows: list[dict[str, object]] = []
    roi_label = ",".join(str(value) for value in roi)
    for index, group in enumerate(groups):
        other_indices = [other for other in range(len(groups)) if other != index]
        if other_indices:
            nearest_index = max(other_indices, key=lambda other: float(similarity[index, other]))
            nearest_id = str(groups[nearest_index]["capture_group_id"])
            nearest_similarity = float(similarity[index, nearest_index])
        else:
            nearest_id = ""
            nearest_similarity = 0.0
        family_id, family_size = family_by_index[index]
        rows.append(
            {
                "capture_group_id": group["capture_group_id"],
                "capture_session": group["capture_session"],
                "source_folder": group["source_folder"],
                "representative_file": group["representative_file"],
                "files_in_capture_group": group["files_in_capture_group"],
                "representative_selection_method": group["selection_method"],
                "inferred_visual_family": family_id,
                "visual_family_size": family_size,
                "nearest_capture_group_id": nearest_id,
                "nearest_visual_similarity": f"{nearest_similarity:.6f}",
                "roi_proxy": roi_label,
                "median_luminance": f"{float(group['median_luminance']):.3f}",
                "texture_strength": f"{float(group['texture_strength']):.6f}",
                "inference_status": "provisional_visual_inference_not_for_split",
                "review_note": (
                    "Chỉ là nhóm tương đồng hình ảnh; không chứng minh cùng vật mẫu."
                ),
            }
        )
    return rows


def write_outputs(
    rows: list[dict[str, object]], output_directory: Path, threshold: float, roi: tuple[int, int, int, int]
) -> None:
    """Write only CSV/JSON metadata; raw image pixels are never written."""
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "visual_group_inference.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    families = sorted({str(row["inferred_visual_family"]) for row in rows})
    summary = {
        "artifact_type": "provisional_visual_group_inference",
        "created_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PROVISIONAL",
        "raw_images_copied": False,
        "capture_group_count": len(rows),
        "visual_family_count": len(families),
        "visual_family_threshold": threshold,
        "threshold_policy": (
            "Ngưỡng cao, bảo thủ: chỉ gộp khi đặc trưng hình ảnh rất tương đồng; "
            "không đại diện cho nhận dạng vật mẫu."
        ),
        "roi_proxy": {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]},
        "outputs": [csv_path.name],
        "limitations": [
            "Tương đồng hình ảnh không chứng minh ảnh thuộc cùng một vật mẫu vật lý.",
            "Không dùng nhóm suy luận làm sample_id hoặc để chia train/calibration/test.",
            "Vật liệu, quyền sử dụng, ROI cuối và danh tính vật mẫu cần xác nhận của người chụp.",
        ],
    }
    (output_directory / "visual_group_inference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Suy luận nhóm hình ảnh tương đồng cục bộ; không gán sample_id/split."
    )
    parser.add_argument("--mapping", action="append", type=Path, required=True)
    parser.add_argument("--images-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roi", type=parse_roi, default=DEFAULT_ROI)
    parser.add_argument("--family-threshold", type=float, default=0.98)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.family_threshold <= 1.0:
        raise ValueError("--family-threshold phải thuộc (0, 1]")
    groups = read_capture_groups(args.mapping, args.images_dir, args.roi)
    rows = infer_rows(groups, args.family_threshold, args.roi)
    write_outputs(rows, args.output_dir, args.family_threshold, args.roi)
    print(
        f"Visual analysis complete for {len(rows)} capture groups; output: {args.output_dir}. "
        "Result is provisional and must not be used as sample_id or split."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
