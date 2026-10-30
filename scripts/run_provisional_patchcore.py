"""Run a local PatchCore-compatible self-inspection on normal-candidate images.

This runner is intentionally separate from the locked MVTec/Anomalib baselines.
It fits a feature memory bank from every supplied image, then scores those same
images only to inspect raw score and anomaly-map outputs. It never calculates
AUROC, chooses a threshold, or presents a self-inspection result as evaluation.

It uses timm + PyTorch directly because the current Windows policy interrupts
the Anomalib PatchCore-model import. It does not disable or alter that policy.
Raw source images are read in place and never copied to the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Must be set before importing torch so CUDA cuBLAS honors deterministic mode.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import timm
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCRIPT_VERSION = "0.2.0"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config phải là YAML mapping.")
    return config


def validate_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    model = config["model"]
    run = config["run"]
    evaluation = config["evaluation"]
    if dataset["input_size"] != [256, 256]:
        raise ValueError("Run thăm dò này khóa input_size = [256, 256].")
    if dataset["roi"] != [0, 0, 3000, 1957]:
        raise ValueError("Run thăm dò này khóa ROI-v0.1 = [0, 0, 3000, 1957].")
    if model["backbone"] != "wide_resnet50_2.racm_in1k":
        raise ValueError("Backbone phải là wide_resnet50_2.racm_in1k.")
    if model["feature_layers"] != ["layer2", "layer3"]:
        raise ValueError("Feature layers phải là layer2 và layer3.")
    if float(model["coreset_sampling_ratio"]) != 0.10:
        raise ValueError("Coreset ratio phải là 0.10.")
    if int(model["num_neighbors"]) != 9:
        raise ValueError("Số láng giềng phải là 9.")
    if int(run["batch_size"]) != 1:
        raise ValueError("Batch size phải là 1 để phù hợp GPU 4 GB.")
    if evaluation["automatic_threshold"] is not False:
        raise ValueError("Không bật automatic threshold trong run thăm dò.")
    if evaluation["mode"] != "self_inspection_only":
        raise ValueError("Run này chỉ hỗ trợ self_inspection_only.")


def configured_input_limitations(config: dict[str, Any]) -> list[str]:
    configured = config["dataset"].get("input_limitations")
    if configured is None:
        return [
            "Capture EXIF records automatic exposure for this batch.",
            "Normal status is an explicit researcher-directed assumption, not independently verified here.",
            "No independent test split, confirmed NG label, defect mask, or threshold is present.",
            "ROI-v0.1 is reused only as a fixed pre-model crop and is not validated for every surface geometry.",
        ]
    if not isinstance(configured, list) or not all(
        isinstance(item, str) and item.strip() for item in configured
    ):
        raise ValueError("dataset.input_limitations phải là danh sách chuỗi không rỗng.")
    return configured


def image_paths(input_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
        for path in input_dir.glob(pattern)
        if path.is_file()
    )
    unique_paths = list(dict.fromkeys(paths))
    if len(unique_paths) < 2:
        raise ValueError("Cần ít nhất hai ảnh để tạo memory bank.")
    return unique_paths


def parse_roi(values: list[int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = (int(value) for value in values)
    image_width, image_height = image_size
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI phải có tọa độ không âm và kích thước dương.")
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"ROI {values} vượt ảnh đã xoay EXIF có kích thước {image_size}."
        )
    return x, y, width, height


def image_details(
    path: Path,
    roi: list[int],
    *,
    record_id: str | None = None,
    capture_role: str = "provisional_normal_candidate",
    label_assumption: str = "normal_candidate_user_directed",
) -> dict[str, Any]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
    parse_roi(roi, (width, height))
    return {
        "record_id": record_id or f"S06-{path.stem}",
        "source_image_path": str(path.resolve()),
        "source_image_sha256": sha256_file(path),
        "source_width_px": width,
        "source_height_px": height,
        "capture_role": capture_role,
        "label_assumption": label_assumption,
        "roi_version": "roi-v0.1",
        "roi_x": roi[0],
        "roi_y": roi[1],
        "roi_width": roi[2],
        "roi_height": roi[3],
    }


def records_from_manifest(path: Path, roi: list[int]) -> list[dict[str, Any]]:
    """Read a technical candidate manifest without copying source pixels."""
    required = {
        "record_id",
        "source_image_path",
        "source_image_sha256",
        "capture_role",
        "label_assumption",
    }
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest đầu vào không có bản ghi nào.")
    if not required.issubset(set(rows[0] or {})):
        missing = sorted(required.difference(set(rows[0] or {})))
        raise ValueError(f"Manifest thiếu cột bắt buộc: {', '.join(missing)}")

    records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    source_paths: set[str] = set()
    for row in rows:
        record_id = str(row["record_id"]).strip()
        source_path = Path(str(row["source_image_path"]).strip()).resolve()
        if not record_id:
            raise ValueError("Manifest có record_id rỗng.")
        if record_id in record_ids:
            raise ValueError(f"record_id bị lặp: {record_id}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Không thấy ảnh trong manifest: {source_path}")
        if str(source_path) in source_paths:
            raise ValueError(f"Ảnh nguồn bị lặp trong manifest: {source_path}")

        details = image_details(
            source_path,
            roi,
            record_id=record_id,
            capture_role=str(row["capture_role"]).strip(),
            label_assumption=str(row["label_assumption"]).strip(),
        )
        expected_hash = str(row["source_image_sha256"]).strip().lower()
        if expected_hash and expected_hash != details["source_image_sha256"]:
            raise ValueError(f"SHA-256 không khớp ảnh nguồn: {source_path}")
        for field in (
            "capture_session",
            "source_folder",
            "capture_group_id",
            "independence_status",
            "selection_method",
        ):
            details[field] = str(row.get(field, "")).strip()
        records.append(details)
        record_ids.add(record_id)
        source_paths.add(str(source_path))
    if len(records) < 2:
        raise ValueError("Cần ít nhất hai ảnh để tạo memory bank.")
    return records


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    fields = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def transform_for(input_size: list[int]) -> Compose:
    return Compose(
        [
            Resize((int(input_size[0]), int(input_size[1])), antialias=True),
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_tensor(path: Path, roi: list[int], transform: Compose) -> torch.Tensor:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        x, y, width, height = parse_roi(roi, image.size)
        cropped = image.crop((x, y, x + width, y + height))
    return transform(cropped)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(config: dict[str, Any], forced_device: str | None) -> torch.device:
    requested = forced_device or str(config["run"]["preferred_device"])
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda" and not bool(config["run"]["allow_cpu_fallback"]):
        raise RuntimeError("CUDA không khả dụng và CPU fallback đang tắt.")
    return torch.device("cpu")


def create_backbone(device: torch.device) -> torch.nn.Module:
    model = timm.create_model(
        "wide_resnet50_2.racm_in1k",
        pretrained=True,
        features_only=True,
        out_indices=(2, 3),
    )
    model.eval().to(device)
    return model


def patch_embeddings(model: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    features = model(image)
    if not isinstance(features, (list, tuple)) or len(features) != 2:
        raise RuntimeError("Backbone không trả về đúng hai feature maps.")
    fine, coarse = features
    if fine.ndim != 4 or coarse.ndim != 4:
        raise RuntimeError("Feature map phải có bốn chiều.")
    coarse = functional.interpolate(
        coarse,
        size=fine.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    embedding = torch.cat([fine, coarse], dim=1)
    grid_shape = (int(embedding.shape[-2]), int(embedding.shape[-1]))
    patches = embedding.permute(0, 2, 3, 1).reshape(-1, embedding.shape[1])
    return functional.normalize(patches, p=2, dim=1), grid_shape


def select_uniform_coreset(
    patches: torch.Tensor,
    ratio: float,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    count = max(1, int(np.ceil(len(patches) * ratio)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(len(patches), generator=generator)[:count]
    return patches[selected].contiguous(), selected.numpy()


def infer_one(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    memory_bank: torch.Tensor,
    num_neighbors: int,
) -> tuple[float, np.ndarray, tuple[int, int]]:
    patches, grid_shape = patch_embeddings(model, tensor)
    distances = torch.cdist(patches.unsqueeze(0), memory_bank.unsqueeze(0)).squeeze(0)
    nearest = torch.topk(
        distances,
        k=min(num_neighbors, memory_bank.shape[0]),
        largest=False,
        dim=1,
    ).values
    anomaly_map = nearest[:, 0].reshape(grid_shape).detach().float().cpu().numpy()
    return float(anomaly_map.max()), anomaly_map, grid_shape


def colorize_map(anomaly_map: np.ndarray, lower: float, upper: float) -> Image.Image:
    scaled = np.clip((anomaly_map - lower) / max(upper - lower, 1e-12), 0.0, 1.0)
    red = 255 * scaled
    green = 255 * (1 - np.abs(2 * scaled - 1))
    blue = 255 * (1 - scaled)
    rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def render_heatmap(
    anomaly_map: np.ndarray,
    lower: float,
    upper: float,
    side_px: int,
    output_path: Path,
) -> None:
    image = colorize_map(anomaly_map, lower, upper)
    image = image.resize((side_px, side_px), Image.Resampling.NEAREST)
    image.save(output_path, format="PNG", optimize=True)


def environment_record(device: torch.device) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torchvision", "timm", "numpy", "Pillow", "PyYAML")
        },
        "torch_cuda_build": torch.version.cuda,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_available": cuda_available,
        "selected_device": str(device),
        "cuda_device": (
            {
                "name": torch.cuda.get_device_name(0),
                "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "capability": list(torch.cuda.get_device_capability(0)),
            }
            if cuda_available and device.type == "cuda"
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-dir", type=Path)
    source_group.add_argument("--input-manifest", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--force-device", choices=("cuda", "cpu"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = project_root()
    config_path = (
        args.config if args.config.is_absolute() else root / args.config
    ).resolve()
    config = load_config(config_path)
    validate_config(config)
    input_dir = args.input_dir.resolve() if args.input_dir else None
    input_source_manifest = (
        args.input_manifest.resolve() if args.input_manifest else None
    )
    if input_dir is not None and not input_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục ảnh: {input_dir}")
    if input_source_manifest is not None and not input_source_manifest.is_file():
        raise FileNotFoundError(f"Không thấy manifest ảnh: {input_source_manifest}")
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Thư mục run phải trống: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = run_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")

    records = (
        records_from_manifest(input_source_manifest, list(config["dataset"]["roi"]))
        if input_source_manifest is not None
        else [
            image_details(path, list(config["dataset"]["roi"]))
            for path in image_paths(input_dir)
        ]
    )
    manifest_path = run_dir / "input_manifest.csv"
    write_manifest(manifest_path, records)
    manifest_hash = sha256_file(manifest_path)
    config_hash = sha256_file(run_dir / "config.snapshot.yaml")
    device = select_device(config, args.force_device)
    metadata: dict[str, Any] = {
        "run_id": run_dir.name,
        "status": "running",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "config_source": str(config_path),
        "config_sha256": config_hash,
        "input_source_type": "manifest" if input_source_manifest else "directory",
        "input_dir": str(input_dir) if input_dir else None,
        "input_source_manifest": (
            str(input_source_manifest) if input_source_manifest else None
        ),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": manifest_hash,
        "input_image_count": len(records),
        "raw_images_copied": False,
        "scope": (
            "Provisional normal-only self-inspection. All inputs construct the memory "
            "bank and are then scored by that same fitted model; outputs are not an "
            "independent test, AUROC, threshold, or production claim."
        ),
        "input_limitations": configured_input_limitations(config),
        "environment": environment_record(device),
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        metadata.update(
            {
                "status": "dry_run_complete",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "dry_run": True,
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"run_dir": str(run_dir), "input_image_count": len(records)}))
        return 0

    try:
        set_seed(int(config["run"]["seed"]))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        transform = transform_for(list(config["dataset"]["input_size"]))
        model = create_backbone(device)

        fit_started = time.perf_counter()
        patch_chunks: list[torch.Tensor] = []
        grid_shape: tuple[int, int] | None = None
        with torch.inference_mode():
            for record in records:
                tensor = load_tensor(
                    Path(record["source_image_path"]),
                    list(config["dataset"]["roi"]),
                    transform,
                ).unsqueeze(0).to(device)
                patches, observed_grid_shape = patch_embeddings(model, tensor)
                if grid_shape is None:
                    grid_shape = observed_grid_shape
                elif grid_shape != observed_grid_shape:
                    raise RuntimeError("Kích thước lưới patch không nhất quán giữa các ảnh.")
                patch_chunks.append(patches.detach().float().cpu())
        all_patches = torch.cat(patch_chunks, dim=0)
        memory_bank_cpu, selected_indices = select_uniform_coreset(
            all_patches,
            float(config["model"]["coreset_sampling_ratio"]),
            int(config["run"]["seed"]),
        )
        fit_seconds = time.perf_counter() - fit_started
        np.savez_compressed(
            run_dir / "memory_bank.npz",
            embedding=memory_bank_cpu.numpy().astype(np.float32),
            selected_patch_indices=selected_indices,
        )
        memory_bank = memory_bank_cpu.to(device)

        predict_started = time.perf_counter()
        score_records: list[dict[str, Any]] = []
        anomaly_maps: list[np.ndarray] = []
        with torch.inference_mode():
            for record in records:
                tensor = load_tensor(
                    Path(record["source_image_path"]),
                    list(config["dataset"]["roi"]),
                    transform,
                ).unsqueeze(0).to(device)
                image_score, anomaly_map, observed_grid_shape = infer_one(
                    model,
                    tensor,
                    memory_bank,
                    int(config["model"]["num_neighbors"]),
                )
                if observed_grid_shape != grid_shape:
                    raise RuntimeError("Lưới patch dự đoán không khớp memory bank.")
                score_records.append(
                    {
                        "record_id": record["record_id"],
                        "source_image_path": record["source_image_path"],
                        "source_image_sha256": record["source_image_sha256"],
                        "capture_role": record["capture_role"],
                        "label_assumption": record["label_assumption"],
                        "capture_session": record.get("capture_session", ""),
                        "capture_group_id": record.get("capture_group_id", ""),
                        "selection_method": record.get("selection_method", ""),
                        "self_inspection_raw_image_score": image_score,
                        "anomaly_map_height": anomaly_map.shape[0],
                        "anomaly_map_width": anomaly_map.shape[1],
                    }
                )
                anomaly_maps.append(anomaly_map.astype(np.float32))
        predict_seconds = time.perf_counter() - predict_started
        with (run_dir / "raw_image_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(score_records[0].keys()))
            writer.writeheader()
            writer.writerows(score_records)
        map_array = np.stack(anomaly_maps).astype(np.float32)
        np.savez_compressed(
            run_dir / "raw_anomaly_maps.npz",
            record_ids=np.asarray([record["record_id"] for record in score_records]),
            anomaly_maps=map_array,
        )
        lower, upper = (float(value) for value in np.percentile(map_array, [1, 99]))
        for score_record, anomaly_map in zip(score_records, anomaly_maps, strict=True):
            render_heatmap(
                anomaly_map,
                lower,
                upper,
                int(config["run"]["heatmap_side_px"]),
                heatmap_dir / f"{score_record['record_id']}.png",
            )
        summary = {
            "run_id": run_dir.name,
            "scope": metadata["scope"],
            "input_image_count": len(records),
            "fit_memory_bank_image_count": len(records),
            "score_image_count": len(score_records),
            "test_image_count": 0,
            "confirmed_ng_image_count": 0,
            "metrics": "not_computed",
            "threshold": "not_computed",
            "anomaly_map_shape": list(map_array.shape[1:]),
            "heatmap_scale_percentiles": {"p01": lower, "p99": upper},
            "score_range": {
                "minimum": float(min(item["self_inspection_raw_image_score"] for item in score_records)),
                "maximum": float(max(item["self_inspection_raw_image_score"] for item in score_records)),
            },
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        metadata.update(
            {
                "status": "complete",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": fit_seconds + predict_seconds,
                "feature_grid_shape": list(grid_shape or ()),
                "feature_patch_count": int(all_patches.shape[0]),
                "feature_dimension": int(all_patches.shape[1]),
                "memory_bank_patch_count": int(memory_bank_cpu.shape[0]),
                "memory_bank_method": config["model"]["coreset_method"],
                "raw_score_csv": str(run_dir / "raw_image_scores.csv"),
                "raw_map_archive": str(run_dir / "raw_anomaly_maps.npz"),
                "heatmap_count": len(score_records),
                "summary_path": str(run_dir / "summary.json"),
                "cuda_peak_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device.type == "cuda"
                    else None
                ),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False))
        return 0
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
