"""Phân tích EXIF và độ ổn định ánh sáng cho một burst ảnh coupon cục bộ.

Script chỉ đọc ảnh đầu vào và ghi các artifact đo lường vào --output-dir. Nó
không sao chép hay di chuyển ảnh gốc. Mặc định, 10 ảnh đầu theo tên tệp là
burst tham chiếu; hãy đặt tên theo thời gian chụp hoặc chỉ truyền một thư mục
chứa đúng burst cần kiểm tra.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import ExifTags, Image, ImageOps, __version__ as pillow_version

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_VERSION = "0.1.0"
HISTOGRAM_BINS = 64
GRID_SIZE = 8
CONFIGURATION_FIELDS = (
    "exif_exposure_mode",
    "exif_white_balance",
    "exif_iso",
    "exif_exposure_time_s",
    "exif_f_number",
    "exif_focal_length_mm",
    "exif_digital_zoom_ratio",
    "width_px",
    "height_px",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> str | int | float | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_exif(image: Image.Image) -> dict[str, Any]:
    exif = image.getexif()
    tags = {ExifTags.TAGS.get(tag_id, str(tag_id)): value for tag_id, value in exif.items()}
    try:
        exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    except (AttributeError, KeyError, TypeError, ValueError):
        exif_ifd = {}
    tags.update(
        {ExifTags.TAGS.get(tag_id, str(tag_id)): value for tag_id, value in exif_ifd.items()}
    )
    return tags


def exif_first(tags: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in tags:
            return tags[name]
    return None


def parse_roi(value: str, width: int, height: int) -> tuple[int, int, int, int]:
    if value == "center80":
        x0, y0 = int(width * 0.1), int(height * 0.1)
        x1, y1 = int(width * 0.9), int(height * 0.9)
        return x0, y0, x1 - x0, y1 - y0
    try:
        x, y, roi_width, roi_height = (int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("--roi phải là center80 hoặc x,y,width,height theo pixel.") from error
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI phải có tọa độ không âm và kích thước dương.")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError("ROI vượt ngoài kích thước ảnh sau khi áp orientation EXIF.")
    return x, y, roi_width, roi_height


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def grid_statistics(values: np.ndarray) -> tuple[float, float, float]:
    height, width = values.shape
    means = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            y0, y1 = row * height // GRID_SIZE, (row + 1) * height // GRID_SIZE
            x0, x1 = column * width // GRID_SIZE, (column + 1) * width // GRID_SIZE
            means.append(float(values[y0:y1, x0:x1].mean()))
    grid = np.asarray(means, dtype=np.float64)
    cv = float(100 * grid.std(ddof=0) / grid.mean()) if grid.mean() else float("nan")
    return float(grid.min()), float(grid.max()), cv


def manual_exif_status(record: dict[str, Any]) -> str:
    exposure_mode = record["exif_exposure_mode"]
    white_balance = record["exif_white_balance"]
    if exposure_mode == 1 and white_balance == 1:
        return "manual_confirmed"
    if exposure_mode == 0 or white_balance == 0:
        return "automatic_detected"
    if exposure_mode is None or white_balance is None:
        return "not_recorded"
    return "other_value"


def analyse_image(path: Path, roi_text: str, expected_size: tuple[int, int] | None) -> dict[str, Any]:
    with Image.open(path) as source:
        tags = extract_exif(source)
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if expected_size is not None and (width, height) != expected_size:
            raise ValueError(
                f"{path.name} có kích thước {(width, height)}, khác ảnh đầu {expected_size}."
            )
        x, y, roi_width, roi_height = parse_roi(roi_text, width, height)
        rgb = np.asarray(image, dtype=np.float32)

    roi_rgb = rgb[y : y + roi_height, x : x + roi_width]
    values = luminance(roi_rgb)
    histogram, edges = np.histogram(values, bins=HISTOGRAM_BINS, range=(0, 256), density=False)
    grid_min, grid_max, grid_cv = grid_statistics(values)
    record: dict[str, Any] = {
        "image_file": path.name,
        "image_sha256": sha256_file(path),
        "width_px": width,
        "height_px": height,
        "roi_x": x,
        "roi_y": y,
        "roi_width": roi_width,
        "roi_height": roi_height,
        "luminance_mean": float(values.mean()),
        "luminance_median": float(np.median(values)),
        "luminance_std": float(values.std(ddof=0)),
        "luminance_p05": float(np.percentile(values, 5)),
        "luminance_p95": float(np.percentile(values, 95)),
        "grid_min_luminance": grid_min,
        "grid_max_luminance": grid_max,
        "grid_cv_percent": grid_cv,
        "clip_below_5_percent": float(100 * (values < 5).mean()),
        "clip_above_250_percent": float(100 * (values > 250).mean()),
        "exif_exposure_mode": numeric_value(exif_first(tags, "ExposureMode")),
        "exif_white_balance": numeric_value(exif_first(tags, "WhiteBalance")),
        "exif_iso": numeric_value(exif_first(tags, "PhotographicSensitivity", "ISOSpeedRatings")),
        "exif_exposure_time_s": numeric_value(exif_first(tags, "ExposureTime")),
        "exif_f_number": numeric_value(exif_first(tags, "FNumber")),
        "exif_focal_length_mm": numeric_value(exif_first(tags, "FocalLength")),
        "exif_digital_zoom_ratio": numeric_value(exif_first(tags, "DigitalZoomRatio")),
        "exif_software": json_value(exif_first(tags, "Software")),
        "exif_datetime_original": json_value(exif_first(tags, "DateTimeOriginal")),
        "histogram": histogram.astype(int).tolist(),
        "histogram_edges": edges.astype(float).tolist(),
    }
    record["manual_exif_status"] = manual_exif_status(record)
    return record


def configuration_signature(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[name] for name in CONFIGURATION_FIELDS)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [key for key in records[0] if key not in {"histogram", "histogram_edges"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in record.items()
                if key not in {"histogram", "histogram_edges"}
            }
            for record in records
        )


def plot_histograms(records: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for record in records:
        edges = np.asarray(record["histogram_edges"], dtype=float)
        histogram = np.asarray(record["histogram"], dtype=float)
        normalized = histogram / histogram.sum() if histogram.sum() else histogram
        axis.step(edges[:-1], normalized, where="post", label=record["image_file"])
    axis.set_title("Histogram luminance ROI")
    axis.set_xlabel("Luminance JPEG (0–255)")
    axis.set_ylabel("Tỷ lệ pixel")
    axis.set_xlim(0, 255)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_medians(
    records: list[dict[str, Any]], lower: float, upper: float, path: Path
) -> None:
    indices = np.arange(1, len(records) + 1)
    medians = [record["luminance_median"] for record in records]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(indices, medians, marker="o", label="Median luminance")
    axis.axhspan(lower, upper, alpha=0.18, color="tab:green", label="Dải tham chiếu ±3s")
    axis.set_title("Median luminance theo thứ tự ảnh")
    axis.set_xlabel("Thứ tự ảnh")
    axis.set_ylabel("Luminance JPEG (0–255)")
    axis.set_xticks(indices)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_grid_cv(records: list[dict[str, Any]], path: Path) -> None:
    indices = np.arange(1, len(records) + 1)
    values = [record["grid_cv_percent"] for record in records]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(indices, values, color="tab:blue")
    axis.set_title("Độ không đồng đều sáng ROI theo lưới 8 × 8")
    axis.set_xlabel("Thứ tự ảnh")
    axis.set_ylabel("Grid CV (%)")
    axis.set_xticks(indices)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--glob", default="*.jpg", help="Mẫu tệp, mặc định *.jpg.")
    parser.add_argument("--reference-count", type=int, default=10)
    parser.add_argument("--roi", default="center80", help="center80 hoặc x,y,width,height theo pixel.")
    parser.add_argument("--baseline-id", default=None)
    parser.add_argument(
        "--allow-nonmanual-exif",
        action="store_true",
        help="Chỉ dùng khi đã ghi rõ vì sao thiết bị không thể lưu ExposureMode/WB manual.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    image_paths = sorted(path for path in input_dir.glob(args.glob) if path.is_file())
    if args.reference_count < 2:
        raise ValueError("--reference-count phải ít nhất là 2.")
    if len(image_paths) < args.reference_count:
        raise ValueError(
            f"Cần ít nhất {args.reference_count} ảnh, nhưng chỉ tìm thấy {len(image_paths)} ảnh."
        )

    records: list[dict[str, Any]] = []
    expected_size: tuple[int, int] | None = None
    for image_path in image_paths:
        record = analyse_image(image_path, args.roi, expected_size)
        expected_size = (record["width_px"], record["height_px"])
        records.append(record)

    reference = records[: args.reference_count]
    signatures = {configuration_signature(record) for record in reference}
    settings_consistent = len(signatures) == 1
    reference_medians = np.asarray([record["luminance_median"] for record in reference])
    baseline_mean = float(reference_medians.mean())
    baseline_std = float(reference_medians.std(ddof=1))
    lower, upper = baseline_mean - 3 * baseline_std, baseline_mean + 3 * baseline_std
    baseline_id = args.baseline_id or f"{input_dir.name}-reference-{args.reference_count}"
    reference_signature = configuration_signature(reference[0])

    for index, record in enumerate(records):
        record["illumination_baseline_id"] = baseline_id
        record["configuration_matches_reference"] = configuration_signature(record) == reference_signature
        if index < args.reference_count:
            record["capture_role"] = "stability_reference"
            record["illumination_qc_status"] = "reference"
        elif not record["configuration_matches_reference"]:
            record["capture_role"] = "dataset_candidate"
            record["illumination_qc_status"] = "configuration_drift"
        elif lower <= record["luminance_median"] <= upper:
            record["capture_role"] = "dataset_candidate"
            record["illumination_qc_status"] = "within_reference_band"
        else:
            record["capture_role"] = "dataset_candidate"
            record["illumination_qc_status"] = "illumination_drift"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "capture_stability.csv", records)
    (output_dir / "capture_stability.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manual_ok = all(record["manual_exif_status"] == "manual_confirmed" for record in records)
    passed = settings_consistent and (manual_ok or args.allow_nonmanual_exif)
    metadata = {
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "analysed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "input_dir_name": input_dir.name,
        "input_image_count": len(records),
        "input_images": [
            {"file": record["image_file"], "sha256": record["image_sha256"]}
            for record in records
        ],
        "arguments": {
            "glob": args.glob,
            "reference_count": args.reference_count,
            "roi": args.roi,
            "baseline_id": baseline_id,
            "allow_nonmanual_exif": args.allow_nonmanual_exif,
        },
        "baseline": {
            "median_mean": baseline_mean,
            "median_sample_standard_deviation": baseline_std,
            "lower": lower,
            "upper": upper,
            "settings_consistent": settings_consistent,
        },
        "manual_exif_all_confirmed": manual_ok,
        "require_manual_exif": not args.allow_nonmanual_exif,
        "status": "PASS" if passed else "FAIL",
        "output_artifacts": [
            "capture_stability.csv",
            "capture_stability.json",
            "analysis_metadata.json",
            "histogram_luminance.png",
            "median_luminance.png",
            "uniformity_grid_cv.png",
        ],
        "packages": {
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": pillow_version,
        },
        "raw_images_copied": False,
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_histograms(records, output_dir / "histogram_luminance.png")
    plot_medians(records, lower, upper, output_dir / "median_luminance.png")
    plot_grid_cv(records, output_dir / "uniformity_grid_cv.png")

    print("PASS" if passed else "FAIL")
    print(output_dir)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
