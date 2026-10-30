"""Measure locked PatchCore latency and memory on the public MVTec tile test images.

The protocol is fixed in `configs/public_tile_latency_patchcore.yaml` before any
measurement. The script reads only test image files (never labels or masks),
reuses the frozen W2 checkpoint and the frozen C/D thresholds, and never fits a
model or selects a parameter. Run one process per device so RAM peaks stay
separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import numpy as np
import yaml

from iad_w1.normal_calibration import smooth_map_3x3, upper_empirical_quantile


EXPECTED_MANIFEST_SHA256 = "f8e2ef28194944912d0790bf2b8bb42c2e13ebbee75061b3d92f5f730479fa3b"
TEMPLATE_PATH = ROOT / "data" / "metadata" / "latency_memory_template.csv"
LOG_PATH = ROOT / "data" / "metadata" / "latency_memory_log.csv"
LOCKED_PROTOCOL = {
    "input_split": "test",
    "input_order": "sorted_by_image_relative_path",
    "labels_read": False,
    "precision": "float32",
    "batch_size": 1,
    "input_size": [256, 256],
    "seed": 42,
    "warmup_image_count": 10,
    "measured_passes": 3,
    "percentile_method": "nearest_rank",
    "stages": [
        "read_decode",
        "preprocess_to_device",
        "model_forward",
        "postprocess_cd_decision",
        "heatmap_overlay_png_write",
    ],
}
STAGES = ("read", "preprocess", "model", "post", "heatmap")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_protocol(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Latency config must be a YAML mapping.")
    for key, expected in LOCKED_PROTOCOL.items():
        if config.get(key) != expected:
            raise ValueError(f"Locked latency protocol changed at {key!r}; record a new protocol first.")
    if config.get("validity", {}).get("max_abs_score_difference_to_baseline") != 0.01:
        raise ValueError("Locked score-parity tolerance changed; record a new protocol first.")
    return config


def summarize(values_ms: list[float]) -> dict[str, float | int]:
    """Summarize latencies with nearest-rank percentiles (no interpolation)."""
    sample = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": int(sample.size),
        "mean_ms": float(sample.mean()),
        "p50_ms": upper_empirical_quantile(sample, 0.50),
        "p95_ms": upper_empirical_quantile(sample, 0.95),
        "min_ms": float(sample.min()),
        "max_ms": float(sample.max()),
    }


def template_columns() -> list[str]:
    with TEMPLATE_PATH.open("r", newline="", encoding="utf-8") as source:
        return next(csv.reader(source))


def template_row(values: dict[str, Any]) -> dict[str, Any]:
    columns = template_columns()
    unknown = set(values) - set(columns)
    if unknown:
        raise ValueError(f"Values outside the latency template: {sorted(unknown)}")
    return {column: values.get(column, "") for column in columns}


def append_log_rows(rows: list[dict[str, Any]]) -> None:
    columns = template_columns()
    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def nvidia_smi_snapshot() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def measure(config_path: Path, device_name: str, baseline_dir: Path, run_dir: Path) -> dict[str, Any]:
    config = read_protocol(config_path)
    if device_name not in config["devices"]:
        raise ValueError(f"Device {device_name!r} is not part of the locked protocol.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")

    import cv2
    import psutil
    import torch
    from anomalib.data.utils import read_image
    from torchvision.transforms.v2 import Normalize, Resize

    from iad_w1.manifest_data import IMAGENET_MEAN, IMAGENET_STD
    from run_public_baseline import (
        create_model,
        environment_record,
        load_rows,
        manifest_tile_root,
        read_config,
        set_seed,
    )
    from run_public_w4 import checkpoint_path, digest, read_csv, read_json

    battery = psutil.sensors_battery()
    on_ac_power = battery is None or bool(battery.power_plugged)
    if config["validity"]["require_ac_power"] and not on_ac_power:
        raise RuntimeError("Protocol requires AC power; plug in the laptop and rerun.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA measurement requested but CUDA is unavailable.")

    manifest_path = resolve(config["manifest"])
    if digest(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Locked tile manifest hash does not match W1/W2.")
    baseline_meta = read_json(baseline_dir / "run_metadata.json")
    if baseline_meta.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Baseline run uses a different manifest.")
    checkpoint = checkpoint_path(baseline_dir)
    checkpoint_sha256 = digest(checkpoint)
    thresholds = read_json(resolve(config["cd_run"]) / "frozen_thresholds.json")
    if thresholds.get("status") != "frozen_before_test_evaluation":
        raise ValueError("C/D thresholds are not frozen.")
    if thresholds.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("C/D thresholds were frozen for a different checkpoint.")
    threshold_c = float(thresholds["threshold_c_raw_image_score"])
    threshold_d = float(thresholds["threshold_d_smoothed_map_max"])
    baseline_scores = {
        row["record_id"]: float(row["raw_image_score"])
        for row in read_csv(baseline_dir / "raw_image_scores.csv")
    }

    tile_root = manifest_tile_root(manifest_path)
    test_rows = sorted(
        (row for row in load_rows(manifest_path) if row["split"] == config["input_split"]),
        key=lambda row: row["image_relative_path"],
    )
    if len(test_rows) != 117:
        raise ValueError(f"Expected 117 locked test images, found {len(test_rows)}.")
    # Only image paths and record IDs are used below; labels and masks stay unread.
    inputs = [(row["record_id"], row["image_relative_path"], tile_root / row["image_relative_path"]) for row in test_rows]

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.snapshot.yaml")
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at_utc": now_utc(),
        "command": [sys.executable, *sys.argv],
        "device": device_name,
        "baseline_run": str(baseline_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "cd_thresholds": {"C": threshold_c, "D": threshold_d},
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "config_sha256": digest(run_dir / "config.snapshot.yaml"),
        "on_ac_power": on_ac_power,
        "battery_percent": None if battery is None else battery.percent,
        "cpu_logical_count": psutil.cpu_count(logical=True),
        "cpu_physical_count": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total,
        "environment": environment_record(),
        "test_access_policy": "Only test image files are read; labels and masks are never loaded.",
    }
    metadata_path = run_dir / "run_metadata.json"

    def save_metadata() -> None:
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_metadata()
    try:
        set_seed(int(config["seed"]))
        torch_device = torch.device("cuda" if device_name == "cuda" else "cpu")

        def synchronize() -> None:
            if torch_device.type == "cuda":
                torch.cuda.synchronize()

        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        load_started = time.perf_counter()
        baseline_config = read_config(baseline_dir / "config.snapshot.yaml")
        model = create_model(baseline_config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.model.memory_bank = torch.empty_like(state["model.memory_bank"])
        model.load_state_dict(state, strict=True)
        model.eval()
        model.to(torch_device)
        synchronize()
        load_seconds = time.perf_counter() - load_started
        memory_bank_shape = list(model.model.memory_bank.shape)
        del state

        resize = Resize(tuple(config["input_size"]), antialias=True)
        normalize = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        heatmap_path = run_dir / f"heatmap_overlay_last_{device_name}.png"
        display_max = 2.0 * threshold_d  # fixed colour scale; visual only

        def run_one(image_path: Path) -> tuple[dict[str, float], dict[str, Any]]:
            marks = [time.perf_counter()]
            image = read_image(image_path, as_tensor=True)
            marks.append(time.perf_counter())
            resized = resize(image)
            batch = normalize(resized).unsqueeze(0).to(torch_device)
            synchronize()
            marks.append(time.perf_counter())
            output = model.model(batch)
            synchronize()
            marks.append(time.perf_counter())
            raw_score = float(output.pred_score.reshape(-1)[0].item())
            anomaly_map = output.anomaly_map.detach().squeeze().cpu().numpy().astype(np.float32)
            smoothed = smooth_map_3x3(anomaly_map)
            d_score = float(smoothed.max())
            c_alert = raw_score > threshold_c
            d_alert = d_score > threshold_d
            marks.append(time.perf_counter())
            base = (resized.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            scaled = np.clip(smoothed / display_max, 0.0, 1.0)
            colour = cv2.applyColorMap((scaled * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.5, colour, 0.5, 0.0)
            if not cv2.imwrite(str(heatmap_path), overlay):
                raise RuntimeError(f"Could not write heatmap overlay: {heatmap_path}")
            marks.append(time.perf_counter())
            timings = {stage: (marks[i + 1] - marks[i]) * 1000.0 for i, stage in enumerate(STAGES)}
            timings["e2e"] = (marks[-1] - marks[0]) * 1000.0
            outputs = {"raw_score": raw_score, "d_smoothed_max": d_score, "c_alert": int(c_alert), "d_alert": int(d_alert)}
            return timings, outputs

        records: list[dict[str, Any]] = []
        with torch.inference_mode():
            first_timings, _ = run_one(inputs[0][2])
            for _, _, image_path in inputs[1 : config["warmup_image_count"]]:
                run_one(image_path)
            measured_started = time.perf_counter()
            for pass_index in range(1, config["measured_passes"] + 1):
                for order_index, (record_id, relative_path, image_path) in enumerate(inputs):
                    timings, outputs = run_one(image_path)
                    records.append({
                        "device": device_name,
                        "pass": pass_index,
                        "order_index": order_index,
                        "record_id": record_id,
                        "image_relative_path": relative_path,
                        **{f"{key}_ms": value for key, value in timings.items()},
                        **outputs,
                    })
            measured_seconds = time.perf_counter() - measured_started

        gpu_memory = None
        if torch_device.type == "cuda":
            gpu_memory = {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "nvidia_smi_while_loaded": nvidia_smi_snapshot(),
            }
        process_memory = psutil.Process().memory_info()
        peak_ram_bytes = int(getattr(process_memory, "peak_wset", process_memory.rss))

        log_path = run_dir / "per_image_latency.csv"
        with log_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

        differences = [abs(item["raw_score"] - baseline_scores[item["record_id"]]) for item in records]
        max_difference = float(max(differences))
        expected_count = config["measured_passes"] * len(inputs)
        valid = (
            len(records) == expected_count
            and max_difference <= config["validity"]["max_abs_score_difference_to_baseline"]
            and on_ac_power
        )
        summary = {
            "status": "valid" if valid else "invalid",
            "device": device_name,
            "model_load_seconds": load_seconds,
            "first_image_e2e_ms": first_timings["e2e"],
            "first_image_model_ms": first_timings["model"],
            "measured_image_count": len(records),
            "measured_total_seconds": measured_seconds,
            "stage_summary": {
                key: summarize([item[f"{key}_ms"] for item in records]) for key in (*STAGES, "e2e")
            },
            "max_abs_score_difference_to_baseline": max_difference,
            "memory_bank_shape": memory_bank_shape,
            "gpu_memory": gpu_memory,
            "process_peak_ram_bytes": peak_ram_bytes,
            "torch_num_threads": torch.get_num_threads(),
            "per_image_log_sha256": digest(log_path),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if valid:
            run_id = run_dir.name
            relative_log = log_path.relative_to(ROOT).as_posix() if log_path.is_relative_to(ROOT) else str(log_path)
            common = {
                "run_id": run_id,
                "model": "Patchcore",
                "seed": config["seed"],
                "hardware_id": config["hardware_id"],
                "device": device_name,
                "precision": config["precision"],
                "batch_size": config["batch_size"],
                "input_size_px": "x".join(str(value) for value in config["input_size"]),
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "split": config["input_split"],
                "warmup_image_count": config["warmup_image_count"],
                "measured_image_count": len(records),
                "total_elapsed_s": round(measured_seconds, 3),
                "peak_gpu_memory_mb": (
                    round(gpu_memory["max_allocated_bytes"] / 2**20, 1) if gpu_memory else ""
                ),
                "process_peak_ram_mb": round(peak_ram_bytes / 2**20, 1),
                "measurement_command": " ".join(sys.argv),
                "raw_log_relative_path": relative_log,
            }
            rows = []
            for variant, key, note in (
                ("model_forward", "model", "Feature extraction + memory-bank kNN + score/map; synchronized."),
                ("end_to_end_cd_heatmap", "e2e", "Read/decode, preprocess, model, C/D decision, heatmap PNG write."),
            ):
                stats = summary["stage_summary"][key]
                rows.append(template_row({
                    **common,
                    "variant": variant,
                    "mean_latency_ms": round(stats["mean_ms"], 2),
                    "p50_latency_ms": round(stats["p50_ms"], 2),
                    "p95_latency_ms": round(stats["p95_ms"], 2),
                    "notes": note,
                }))
            append_log_rows(rows)

        metadata.update({"status": "complete", "finished_at_utc": now_utc(), "summary_status": summary["status"]})
        save_metadata()
        print(json.dumps(summary, ensure_ascii=False))
        return summary
    except Exception as error:
        metadata.update({"status": "failed", "finished_at_utc": now_utc(), "error": repr(error)})
        save_metadata()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "public_tile_latency_patchcore.yaml")
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Frozen PatchCore baseline run stored outside this public repository.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    measure(args.config.resolve(), args.device, args.baseline_dir.resolve(), args.run_dir.resolve())


if __name__ == "__main__":
    main()
