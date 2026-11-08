"""Measure latency and memory of the locked internal pipelines on self-captured images.

Protocol D-129. One process per model and device keeps the memory peaks apart. A
separate `--prepare` process refits PatchCore or PaDiM exactly as the reference run
and saves the state, so fitting never counts towards the measured peaks. Only image
files are read from the test split; labels and masks are never used, and nothing
here selects a model, threshold, ROI or post-processing parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from iad_w1.normal_calibration import smooth_map_3x3
from measure_latency_memory import STAGES, append_log_rows, nvidia_smi_snapshot, summarize, template_row
from run_internal_experiment import load_view, read_csv, sha256_file

VARIANT_OF = {"patchcore": "D", "padim": "E", "efficientad": "F"}
RAW_FILE = {"patchcore": "raw_patchcore_roi_seed42.npz", "padim": "raw_padim_roi_seed42.npz"}


def load_state(model, state: dict[str, Any]) -> None:
    """Load a fitted state; PatchCore and PaDiM start with empty buffers, so resize those first."""
    import torch

    own = model.state_dict()
    for key, value in state.items():
        if key in own and own[key].shape != value.shape:
            path, _, name = key.rpartition(".")
            setattr(model.get_submodule(path) if path else model, name, torch.empty_like(value))
    model.load_state_dict(state, strict=True)


def parity(measured: np.ndarray, stored: np.ndarray, thresholds: list[tuple[np.ndarray, np.ndarray, float]]) -> dict[str, Any]:
    """Largest relative score difference across passes, and how many alert decisions differ."""
    denominator = np.maximum(np.abs(stored), 1e-12)
    relative = float((np.abs(measured - stored) / denominator).max())
    changed = sum(int(((new > limit) != (old > limit)).sum()) for new, old, limit in thresholds)
    return {"max_relative_difference": relative, "decisions_changed": changed}


def is_valid(check: dict[str, Any], limit: float, records: int, expected: int, on_ac_power: bool) -> bool:
    return records == expected and on_ac_power and check["max_relative_difference"] <= limit and check["decisions_changed"] == 0


def new_model(family: str, base: dict[str, Any], ead: dict[str, Any]):
    if family == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        params = base["patchcore"]
        return PatchcoreModel(backbone=params["backbone"], layers=params["layers"], pre_trained=True, num_neighbors=params["num_neighbors"])
    if family == "padim":
        from anomalib.models.image.padim.torch_model import PadimModel

        params = base["padim"]
        return PadimModel(backbone=params["backbone"], layers=params["layers"], pre_trained=True)
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize

    return EfficientAdModel(teacher_out_channels=ead["teacher_out_channels"], model_size=EfficientAdModelSize(ead["model_size"]),
                            padding=ead["padding"], pad_maps=ead["pad_maps"])


def prepare(family: str, base: dict[str, Any], ead: dict[str, Any], split_dir: Path, state_path: Path, seed: int) -> None:
    """Refit exactly as `fit_and_score` does and keep the state; the fit is not measured."""
    import torch

    from run_internal_experiment import to_tensor
    from run_public_baseline import set_seed

    metadata = {row["image_id"]: row for row in read_csv(split_dir / "metadata.csv")}
    model_ids = sorted(row["image_id"] for row in read_csv(split_dir / "assignments.csv") if row["split"] == "model")
    size, roi = tuple(base["input_size"]), base["variants"]["B"]["roi"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = [to_tensor(load_view(Path(metadata[i]["image_path"]), "roi", roi), size) for i in model_ids]
    set_seed(seed)
    model = new_model(family, base, ead).to(device)
    model.train()
    with torch.no_grad():
        for tensor in train:
            model(tensor.unsqueeze(0).to(device))
        if family == "patchcore":
            model.subsample_embedding(base["patchcore"]["coreset_sampling_ratio"])
        else:
            model.fit()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, state_path)
    print("prepared", family, state_path, flush=True)


def measure(config_path: Path, family: str, device_name: str, split_dir: Path, state_path: Path, run_dir: Path) -> dict[str, Any]:
    import cv2
    import psutil
    import torch
    from PIL import Image

    from run_internal_efficientad import to_unit_tensor
    from run_internal_experiment import to_tensor

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / config["base_config"]).read_text(encoding="utf-8"))
    ead = yaml.safe_load((ROOT / config["efficientad_config"]).read_text(encoding="utf-8"))["efficientad"]
    if device_name not in config["devices"]:
        raise ValueError(f"Device {device_name!r} is not part of the locked protocol.")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {run_dir}")
    battery = psutil.sensors_battery()
    on_ac_power = battery is None or bool(battery.power_plugged)
    if config["validity"]["require_ac_power"] and not on_ac_power:
        raise RuntimeError("Protocol requires AC power; plug in the laptop and rerun.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA measurement requested but CUDA is unavailable.")

    validation = json.loads((split_dir / "split_validation.json").read_text(encoding="utf-8"))
    if validation.get("integrity_status") != "PASS" or validation["assignments_sha256"] != sha256_file(split_dir / "assignments.csv"):
        raise SystemExit("Split is not valid for this measurement.")
    metadata = {row["image_id"]: row for row in read_csv(split_dir / "metadata.csv")}
    test_ids = sorted(row["image_id"] for row in read_csv(split_dir / "assignments.csv") if row["split"] == config["input_split"])
    size, roi = tuple(base["input_size"]), base["variants"]["B"]["roi"]
    x, y, w, h = roi

    reference, ead_run = ROOT / config["reference_run"], ROOT / config["efficientad_run"]
    frozen = json.loads((reference / "frozen_thresholds.json").read_text(encoding="utf-8"))[str(config["seed"])]
    frozen.update(json.loads((ead_run / "frozen_thresholds.json").read_text(encoding="utf-8"))[str(config["seed"])])
    raw = np.load(ead_run / "raw_efficientad_roi_seed42.npz") if family == "efficientad" else np.load(reference / RAW_FILE[family])
    if [str(i) for i in raw["test_ids"]] != test_ids:
        raise SystemExit("Image order differs from the reference run.")
    stored_scores = np.asarray(raw["test_scores"], dtype=np.float64)
    stored_d = np.array([smooth_map_3x3(m).max() for m in raw["test_maps"]], dtype=np.float64) if family == "patchcore" else None
    variant = VARIANT_OF[family]
    threshold = float(frozen[variant])
    threshold_c = float(frozen["C"]) if family == "patchcore" else None

    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    meta: dict[str, Any] = {
        "status": "running", "protocol": config["protocol"], "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "family": family, "variant": variant, "device": device_name,
        "config_sha256": sha256_file(config_path), "assignments_sha256": validation["assignments_sha256"],
        "state_path": str(state_path), "state_sha256": sha256_file(state_path), "state_bytes": state_path.stat().st_size,
        "on_ac_power": on_ac_power, "battery_percent": None if battery is None else battery.percent,
        "cpu_logical_count": psutil.cpu_count(logical=True), "cpu_physical_count": psutil.cpu_count(logical=False),
        "ram_total_bytes": psutil.virtual_memory().total, "torch": torch.__version__,
        "test_access_policy": "Only test image files are read; labels and masks are never loaded.",
    }
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    torch_device = torch.device("cuda" if device_name == "cuda" else "cpu")

    def synchronize() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = new_model(family, base, ead)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    load_state(model, state)
    del state
    model.eval()
    model.to(torch_device)
    synchronize()
    load_seconds = time.perf_counter() - started

    make_tensor = to_unit_tensor if family == "efficientad" else to_tensor
    heatmap_path = run_dir / f"heatmap_overlay_last_{family}_{device_name}.png"
    display_max = 2.0 * max(threshold, 1e-6)  # fixed colour scale; visual only

    def run_one(path: Path) -> tuple[dict[str, float], float, float]:
        marks = [time.perf_counter()]
        image = load_view(path, "full_frame", None)
        marks.append(time.perf_counter())
        crop = image.crop((x, y, min(image.width, x + w), min(image.height, y + h)))
        batch = make_tensor(crop, size).unsqueeze(0).to(torch_device)
        synchronize()
        marks.append(time.perf_counter())
        output = model(batch)
        synchronize()
        marks.append(time.perf_counter())
        raw_score = float(output.pred_score.reshape(-1)[0].item())
        anomaly_map = output.anomaly_map.detach().squeeze().cpu().numpy().astype(np.float32)
        final_map = smooth_map_3x3(anomaly_map) if family == "patchcore" else anomaly_map
        decision_score = float(final_map.max()) if family == "patchcore" else raw_score
        _ = decision_score > threshold
        marks.append(time.perf_counter())
        base_image = np.asarray(crop.resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.uint8)
        scaled = np.clip(final_map / display_max, 0.0, 1.0)
        colour = cv2.applyColorMap((scaled * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR), 0.5, colour, 0.5, 0.0)
        if not cv2.imwrite(str(heatmap_path), overlay):
            raise RuntimeError(f"Could not write heatmap overlay: {heatmap_path}")
        marks.append(time.perf_counter())
        timings = {stage: (marks[i + 1] - marks[i]) * 1000.0 for i, stage in enumerate(STAGES)}
        timings["e2e"] = (marks[-1] - marks[0]) * 1000.0
        return timings, raw_score, decision_score

    paths = [Path(metadata[i]["image_path"]) for i in test_ids]
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        first, _, _ = run_one(paths[0])
        for path in paths[1:config["warmup_image_count"]]:
            run_one(path)
        measured_started = time.perf_counter()
        for pass_index in range(1, config["measured_passes"] + 1):
            for order, (image_id, path) in enumerate(zip(test_ids, paths)):
                timings, raw_score, decision_score = run_one(path)
                records.append({"pass": pass_index, "order_index": order, "image_id": image_id,
                                **{f"{k}_ms": v for k, v in timings.items()},
                                "raw_score": raw_score, "decision_score": decision_score})
        measured_seconds = time.perf_counter() - measured_started

    gpu_memory = None
    if torch_device.type == "cuda":
        gpu_memory = {"max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                      "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                      "nvidia_smi_while_loaded": nvidia_smi_snapshot()}
    process_memory = psutil.Process().memory_info()
    peak_ram = int(getattr(process_memory, "peak_wset", process_memory.rss))

    log_path = run_dir / "per_image_latency.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    passes = config["measured_passes"]
    measured_raw = np.array([r["raw_score"] for r in records], dtype=np.float64)
    measured_decision = np.array([r["decision_score"] for r in records], dtype=np.float64)
    if family == "patchcore":
        limits = [(measured_raw, np.tile(stored_scores, passes), threshold_c),
                  (measured_decision, np.tile(stored_d, passes), threshold)]
    else:
        limits = [(measured_decision, np.tile(stored_scores, passes), threshold)]
    check = parity(measured_raw, np.tile(stored_scores, passes), limits)
    limit = config["validity"][f"{device_name}_max_relative_score_difference"]
    valid = is_valid(check, limit, len(records), passes * len(test_ids), on_ac_power)
    summary = {
        "status": "valid" if valid else "invalid", "family": family, "variant": variant, "device": device_name,
        "model_load_seconds": load_seconds, "first_image_e2e_ms": first["e2e"], "first_image_model_ms": first["model"],
        "measured_image_count": len(records), "measured_total_seconds": measured_seconds,
        "stage_summary": {key: summarize([r[f"{key}_ms"] for r in records]) for key in (*STAGES, "e2e")},
        "parity": check, "parity_limit": limit, "gpu_memory": gpu_memory, "process_peak_ram_bytes": peak_ram,
        "state_bytes": state_path.stat().st_size, "torch_num_threads": torch.get_num_threads(),
        "per_image_log_sha256": sha256_file(log_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if valid:
        log_rows = []
        relative_log = log_path.relative_to(ROOT).as_posix() if log_path.is_relative_to(ROOT) else str(log_path)
        for name, key, note in (("model_forward", "model", "Model forward; synchronized."),
                                ("end_to_end_decision_heatmap", "e2e", "JPEG read and decode, ROI crop, preprocess, model, decision, heatmap PNG write.")):
            stats = summary["stage_summary"][key]
            log_rows.append(template_row({
                "run_id": run_dir.name, "model": family, "variant": name, "seed": config["seed"], "hardware_id": config["hardware_id"],
                "device": device_name, "precision": config["precision"], "batch_size": config["batch_size"],
                "input_size_px": "x".join(str(v) for v in size), "manifest_sha256": validation["assignments_sha256"],
                "split": config["input_split"], "warmup_image_count": config["warmup_image_count"],
                "measured_image_count": len(records), "total_elapsed_s": round(measured_seconds, 3),
                "mean_latency_ms": round(stats["mean_ms"], 2), "p50_latency_ms": round(stats["p50_ms"], 2),
                "p95_latency_ms": round(stats["p95_ms"], 2),
                "peak_gpu_memory_mb": round(gpu_memory["max_allocated_bytes"] / 2**20, 1) if gpu_memory else "",
                "process_peak_ram_mb": round(peak_ram / 2**20, 1), "measurement_command": " ".join(sys.argv),
                "raw_log_relative_path": relative_log, "notes": note,
            }))
        append_log_rows(log_rows)

    meta.update({"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(), "summary_status": summary["status"]})
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"family": family, "device": device_name, "status": summary["status"], "parity": check,
                      "e2e_p50": summary["stage_summary"]["e2e"]["p50_ms"]}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_latency_v2.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v2")
    parser.add_argument("--family", choices=sorted(VARIANT_OF), required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"])
    parser.add_argument("--state", type=Path, required=True, help="State file written by --prepare, or efficientad_final.pt.")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare", action="store_true", help="Refit PatchCore or PaDiM and save the state, then stop.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.prepare:
        if args.family == "efficientad":
            raise SystemExit("EfficientAD-S uses the state saved by its reference run.")
        base = yaml.safe_load((ROOT / config["base_config"]).read_text(encoding="utf-8"))
        prepare(args.family, base, {}, args.split_dir, args.state, config["seed"])
        return
    if not args.device or not args.run_dir:
        raise SystemExit("--device and --run-dir are required when measuring.")
    measure(args.config.resolve(), args.family, args.device, args.split_dir.resolve(), args.state.resolve(), args.run_dir.resolve())


if __name__ == "__main__":
    main()
