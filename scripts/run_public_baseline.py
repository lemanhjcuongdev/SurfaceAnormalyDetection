"""Run one locked W1 MVTec tile baseline and preserve raw outputs.

The script intentionally reads only the manifest model split during fitting and
the locked test split during prediction. The calibration split is materialized
for later work but never passed to a model or thresholding method in W1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

from anomalib.data import ImageBatch
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from iad_w1.manifest_data import ManifestImageDataset


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("w1_baseline")


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


def git_revision(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def read_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


def resolve_project_path(path_value: str) -> Path:
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else project_root() / candidate


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    name = model["name"]
    input_size = config["dataset"]["input_size"]
    if input_size != [256, 256]:
        raise ValueError("W1 protocol requires input_size [256, 256].")
    if config["run"]["batch_size"] != 1:
        raise ValueError("W1 protocol requires batch_size 1.")
    if config["evaluation"]["automatic_threshold"] is not False:
        raise ValueError("W1 requires automatic_threshold false.")
    if name == "Patchcore":
        expected = {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.10,
            "num_neighbors": 9,
        }
    elif name == "Padim":
        expected = {
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"],
        }
    else:
        raise ValueError(f"Unsupported W1 model: {name}.")
    for key, expected_value in expected.items():
        if model.get(key) != expected_value:
            raise ValueError(
                f"W1 protocol requires model.{key}={expected_value!r}, "
                f"found {model.get(key)!r}."
            )


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def manifest_tile_root(manifest_path: Path) -> Path:
    summary_path = manifest_path.with_suffix(".summary.json")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    tile_root = Path(summary["tile_root"])
    if not tile_root.is_dir():
        raise FileNotFoundError(f"Tile root from summary does not exist: {tile_root}")
    return tile_root


def assert_manifest_policy(rows: list[dict[str, str]]) -> None:
    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    expected_counts = {"model": 184, "calibration": 46, "test": 117}
    if split_counts != expected_counts:
        raise ValueError(
            f"Manifest split counts must be {expected_counts}, got {split_counts}."
        )
    for row in rows:
        if row["split"] in {"model", "calibration"} and row["label_binary"] != "0":
            raise ValueError("Model or calibration rows must be normal-only.")
    paths_by_split = {
        split: {row["image_relative_path"] for row in rows if row["split"] == split}
        for split in expected_counts
    }
    for first, second in [
        ("model", "calibration"),
        ("model", "test"),
        ("calibration", "test"),
    ]:
        overlap = paths_by_split[first] & paths_by_split[second]
        if overlap:
            raise ValueError(f"Manifest overlap between {first} and {second}: {sorted(overlap)[:3]}")


def select_device(config: dict[str, Any], forced_device: str | None) -> str:
    requested = forced_device or config["run"]["preferred_device"]
    cuda_ready = bool(torch.cuda.is_available())
    if requested == "cuda" and cuda_ready:
        return "gpu"
    if requested == "cuda" and not config["run"]["allow_cpu_fallback"]:
        raise RuntimeError("CUDA was requested but is unavailable and CPU fallback is disabled.")
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def create_model(config: dict[str, Any]):
    model_config = config["model"]
    common = {
        "pre_processor": False,
        "post_processor": False,
        "evaluator": False,
        "visualizer": False,
    }
    if model_config["name"] == "Patchcore":
        return Patchcore(
            backbone=model_config["backbone"],
            layers=model_config["layers"],
            pre_trained=True,
            coreset_sampling_ratio=model_config["coreset_sampling_ratio"],
            num_neighbors=model_config["num_neighbors"],
            precision="float32",
            **common,
        )
    if model_config["name"] == "Padim":
        return Padim(
            backbone=model_config["backbone"],
            layers=model_config["layers"],
            pre_trained=True,
            **common,
        )
    raise ValueError(f"Unsupported W1 model {model_config['name']}.")


def batch_to_records(
    batch: ImageBatch,
    manifest_by_absolute_path: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if batch.pred_score is None or batch.anomaly_map is None:
        raise RuntimeError("Model did not return raw pred_score and anomaly_map.")
    scores = torch.as_tensor(batch.pred_score).detach().cpu().reshape(-1).numpy()
    maps = torch.as_tensor(batch.anomaly_map).detach().cpu().numpy()
    labels = torch.as_tensor(batch.gt_label).detach().cpu().reshape(-1).numpy()
    masks = torch.as_tensor(batch.gt_mask).detach().cpu().numpy()
    image_paths = batch.image_path or []
    if len(scores) != len(image_paths):
        raise RuntimeError("Prediction score count does not match image path count.")

    records: list[dict[str, Any]] = []
    raw_maps: list[np.ndarray] = []
    raw_masks: list[np.ndarray] = []
    raw_labels: list[np.ndarray] = []
    for index, image_path in enumerate(image_paths):
        manifest_row = manifest_by_absolute_path.get(str(Path(image_path).resolve()))
        if manifest_row is None:
            raise RuntimeError(f"Prediction path is absent from locked manifest: {image_path}")
        anomaly_map = np.squeeze(maps[index]).astype(np.float32)
        gt_mask = np.squeeze(masks[index]).astype(np.uint8)
        if anomaly_map.ndim != 2 or gt_mask.ndim != 2:
            raise RuntimeError(
                "Expected a 2D anomaly map and 2D ground-truth mask after prediction."
            )
        if anomaly_map.shape != gt_mask.shape:
            raise RuntimeError(
                f"Map/mask shape mismatch for {image_path}: "
                f"{anomaly_map.shape} versus {gt_mask.shape}."
            )
        observed_label = int(labels[index])
        manifest_label = int(manifest_row["label_binary"])
        if observed_label != manifest_label:
            raise RuntimeError(
                f"Label mismatch for {image_path}: batch={observed_label}, "
                f"manifest={manifest_label}."
            )
        records.append(
            {
                "record_id": manifest_row["record_id"],
                "image_relative_path": manifest_row["image_relative_path"],
                "image_sha256": manifest_row["image_sha256"],
                "label": manifest_row["label"],
                "label_binary": manifest_label,
                "defect_type": manifest_row["defect_type"],
                "raw_image_score": float(scores[index]),
                "anomaly_map_height": int(anomaly_map.shape[0]),
                "anomaly_map_width": int(anomaly_map.shape[1]),
            }
        )
        raw_maps.append(anomaly_map)
        raw_masks.append(gt_mask)
        raw_labels.append(np.asarray(manifest_label, dtype=np.uint8))
    return records, raw_maps, raw_masks, raw_labels


def write_scores(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "record_id",
        "image_relative_path",
        "image_sha256",
        "label",
        "label_binary",
        "defect_type",
        "raw_image_score",
        "anomaly_map_height",
        "anomaly_map_width",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def render_heatmap(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize(
            (anomaly_map.shape[1], anomaly_map.shape[0]),
            Image.Resampling.BILINEAR,
        )
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=160)
    axes[0].imshow(image)
    axes[0].set_title("Ảnh đầu vào")
    axes[0].axis("off")
    axes[1].imshow(image)
    heatmap = axes[1].imshow(anomaly_map, cmap="jet", alpha=0.50)
    axes[1].set_title("Bản đồ bất thường thô")
    axes[1].axis("off")
    figure.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def representative_ids(records: list[dict[str, Any]]) -> list[str]:
    """Select figures before inspecting scores: lexical first path per class."""
    selected: list[str] = []
    seen_types: set[str] = set()
    for record in sorted(records, key=lambda item: item["image_relative_path"]):
        defect_type = str(record["defect_type"])
        if defect_type not in seen_types:
            seen_types.add(defect_type)
            selected.append(str(record["record_id"]))
    return selected


def environment_record() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in [
                "anomalib",
                "torch",
                "torchvision",
                "lightning",
                "numpy",
                "pandas",
                "scikit-learn",
            ]
        },
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_devices": (
            [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else []
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Project-relative or absolute YAML config path.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Optional empty output directory. Default is a timestamped runs directory.",
    )
    parser.add_argument(
        "--force-device",
        choices=["cuda", "cpu"],
        help="Use only for a documented fallback or diagnostic run.",
    )
    args = parser.parse_args()

    root = project_root()
    config_path = resolve_project_path(str(args.config)).resolve()
    config = read_config(config_path)
    validate_config(config)
    run_id = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{config['experiment_id']}"
    )
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else root / "runs" / config["experiment_id"] / run_id
    )
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "heatmaps").mkdir(exist_ok=True)

    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    snapshot_path = run_dir / "config.snapshot.yaml"
    shutil.copy2(config_path, snapshot_path)
    manifest_path = resolve_project_path(config["dataset"]["manifest"]).resolve()
    rows = load_rows(manifest_path)
    assert_manifest_policy(rows)
    tile_root = manifest_tile_root(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    config_hash = sha256_file(snapshot_path)
    device = select_device(config, args.force_device)
    seed = int(config["run"]["seed"])
    set_seed(seed)
    if device == "gpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    started_at = datetime.now(timezone.utc)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "command": [sys.executable, *sys.argv],
        "config_source": str(config_path),
        "config_snapshot": str(snapshot_path),
        "config_sha256": config_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ["model", "calibration", "test"]
        },
        "calibration_policy": "Present in manifest; not passed to model or thresholding in W1.",
        "model_config": config["model"],
        "input_size": config["dataset"]["input_size"],
        "seed": seed,
        "selected_device": device,
        "automatic_threshold": False,
        "environment": environment_record(),
        "git_revision": git_revision(root),
        "figure_selection_rule": "First lexical test path per defect_type; score-independent.",
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        model_rows = [row for row in rows if row["split"] == "model"]
        test_rows = [row for row in rows if row["split"] == "test"]
        input_size = tuple(int(value) for value in config["dataset"]["input_size"])
        train_dataset = ManifestImageDataset(
            model_rows,
            tile_root,
            "train",
            input_size,
        )
        test_dataset = ManifestImageDataset(
            test_rows,
            tile_root,
            "test",
            input_size,
        )
        batch_size = int(config["run"]["batch_size"])
        num_workers = int(config["run"]["num_workers"])
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=device == "gpu",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=device == "gpu",
        )
        model = create_model(config)
        engine = Engine(
            default_root_dir=run_dir / "engine",
            logger=False,
            accelerator=device,
            devices=1,
            deterministic="warn",
            enable_progress_bar=True,
        )

        fit_start = time.perf_counter()
        engine.fit(model=model, train_dataloaders=train_loader)
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        predictions = engine.predict(
            model=model,
            dataloaders=test_loader,
            return_predictions=True,
        )
        predict_seconds = time.perf_counter() - predict_start
        if predictions is None:
            raise RuntimeError("Engine.predict returned no predictions.")

        manifest_by_absolute_path = {
            str((tile_root / row["image_relative_path"]).resolve()): row
            for row in test_rows
        }
        score_records: list[dict[str, Any]] = []
        anomaly_maps: list[np.ndarray] = []
        ground_truth_masks: list[np.ndarray] = []
        image_labels: list[np.ndarray] = []
        for prediction in predictions:
            records, maps, masks, labels = batch_to_records(
                prediction,
                manifest_by_absolute_path,
            )
            score_records.extend(records)
            anomaly_maps.extend(maps)
            ground_truth_masks.extend(masks)
            image_labels.extend(labels)

        score_records.sort(key=lambda item: item["image_relative_path"])
        maps_by_record = {
            record["record_id"]: anomaly_map
            for record, anomaly_map in zip(
                sorted(score_records, key=lambda item: item["image_relative_path"]),
                [
                    anomaly_map
                    for _, anomaly_map in sorted(
                        zip(
                            [
                                row["image_relative_path"]
                                for row in test_rows
                            ],
                            anomaly_maps,
                            strict=True,
                        ),
                        key=lambda pair: pair[0],
                    )
                ],
                strict=True,
            )
        }

        score_path = run_dir / "raw_image_scores.csv"
        write_scores(score_path, score_records)
        sorted_records_for_arrays = sorted(
            score_records,
            key=lambda item: item["image_relative_path"],
        )
        sorted_map_mask_label = sorted(
            zip(
                [
                    row["image_relative_path"]
                    for row in test_rows
                ],
                anomaly_maps,
                ground_truth_masks,
                image_labels,
                strict=True,
            ),
            key=lambda item: item[0],
        )
        map_array = np.stack([item[1] for item in sorted_map_mask_label]).astype(np.float32)
        mask_array = np.stack([item[2] for item in sorted_map_mask_label]).astype(np.uint8)
        label_array = np.asarray([int(item[3]) for item in sorted_map_mask_label], dtype=np.uint8)
        if len(sorted_records_for_arrays) != len(map_array):
            raise RuntimeError("Score and raw-map counts disagree.")
        np.savez_compressed(
            run_dir / "raw_anomaly_maps.npz",
            record_ids=np.asarray(
                [record["record_id"] for record in sorted_records_for_arrays],
                dtype=str,
            ),
            anomaly_maps=map_array,
            gt_masks=mask_array,
        )
        image_scores = np.asarray(
            [record["raw_image_score"] for record in sorted_records_for_arrays],
            dtype=np.float64,
        )
        metrics = {
            "run_id": run_id,
            "manifest_sha256": manifest_hash,
            "selected_device": device,
            "automatic_threshold_enabled": False,
            "reported_outputs": ["raw_score", "anomaly_map"],
            "image_auroc": float(roc_auc_score(label_array, image_scores)),
            "pixel_auroc": float(
                roc_auc_score(mask_array.reshape(-1), map_array.reshape(-1))
            ),
            "test_image_count": int(len(label_array)),
            "test_anomalous_image_count": int(label_array.sum()),
            "test_normal_image_count": int((label_array == 0).sum()),
            "anomaly_map_shape": list(map_array.shape[1:]),
            "map_mask_alignment": "Both were resized to 256 by 256 through the same torchvision v2 transform.",
        }
        metrics_path = run_dir / "raw_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        selected_ids = set(representative_ids(sorted_records_for_arrays))
        for record in sorted_records_for_arrays:
            if record["record_id"] not in selected_ids:
                continue
            image_path = tile_root / record["image_relative_path"]
            stem = Path(record["image_relative_path"]).stem
            output_name = f"{record['defect_type']}__{stem}.png"
            render_heatmap(
                image_path=image_path,
                anomaly_map=maps_by_record[record["record_id"]],
                output_path=run_dir / "heatmaps" / output_name,
                title=(
                    f"{config['model']['name']} — {record['defect_type']} — "
                    f"{record['image_relative_path']}"
                ),
            )

        metadata.update(
            {
                "status": "complete",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": time.perf_counter() - fit_start,
                "raw_score_csv": str(score_path),
                "raw_map_archive": str(run_dir / "raw_anomaly_maps.npz"),
                "metrics": metrics,
                "representative_heatmap_count": len(selected_ids),
                "cuda_peak_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device == "gpu" else None
                ),
                "cuda_peak_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if device == "gpu" else None
                ),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"run_dir": str(run_dir), "metrics": metrics}, ensure_ascii=False))
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
        LOGGER.exception("W1 baseline failed; metadata was retained in %s", metadata_path)
        raise


if __name__ == "__main__":
    main()
