"""Prepare locked normal-only calibration, then evaluate public tile variants C/D.

Run `prepare` before `evaluate`. `prepare` loads a frozen PatchCore checkpoint
and predicts only the 46 normal calibration images. `evaluate` later reads the
already frozen thresholds and the existing 117-image W1/W2 test artifacts.
It never fits a new model or chooses parameters from test results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import yaml

from iad_w1.normal_calibration import (
    confusion_counts,
    decisions,
    smooth_map_3x3,
    threshold_from_normal_maps,
    threshold_from_normal_scores,
)


EXPECTED_MANIFEST_SHA256 = "f8e2ef28194944912d0790bf2b8bb42c2e13ebbee75061b3d92f5f730479fa3b"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight(
    config_path: Path, baseline_dir_override: Path | None = None
) -> tuple[dict[str, Any], Path, Path, list[dict[str, str]]]:
    with config_path.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("W4 config must be a YAML mapping.")
    if config.get("model") != "Patchcore":
        raise ValueError("W4 C/D only supports the frozen PatchCore baseline.")
    if config.get("calibration_split") != "calibration" or config.get("test_split") != "test":
        raise ValueError("W4 split names must remain locked.")
    image_rule = config.get("image_decision", {})
    if image_rule != {
        "quantile": 0.95,
        "method": "nearest_rank",
        "alert_when": "score > threshold",
    }:
        raise ValueError("W4 normal-only quantile rule changed; record a new protocol first.")
    post_rule = config.get("postprocessing", {})
    if post_rule != {
        "method": "fixed_3x3_binomial",
        "kernel_1d": [1, 2, 1],
        "edge_mode": "replicate",
        "image_score": "maximum_of_smoothed_anomaly_map",
    }:
        raise ValueError("W4 fixed smoothing rule changed; record a new protocol first.")
    baseline_dir = (
        baseline_dir_override.resolve()
        if baseline_dir_override is not None
        else resolve(str(config["baseline_run"]))
    )
    manifest_path = resolve(str(config["manifest"]))
    if digest(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Locked tile manifest hash does not match W1/W2.")
    manifest_rows = read_csv(manifest_path)
    counts = {
        split: sum(row["split"] == split for row in manifest_rows)
        for split in ("model", "calibration", "test")
    }
    if counts != {"model": 184, "calibration": 46, "test": 117}:
        raise ValueError(f"Unexpected manifest split counts: {counts}")
    if any(row["label_binary"] != "0" for row in manifest_rows if row["split"] == "calibration"):
        raise ValueError("Calibration split must contain only normal images.")
    baseline = read_json(baseline_dir / "run_metadata.json")
    if baseline.get("status") != "complete" or baseline.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen baseline is incomplete or uses a different manifest.")
    model_config = baseline.get("model_config", {})
    if model_config.get("name") != "Patchcore" or baseline.get("automatic_threshold") is not False:
        raise ValueError("Frozen baseline must be raw PatchCore without automatic thresholding.")
    return config, baseline_dir, manifest_path, manifest_rows


def checkpoint_path(baseline_dir: Path) -> Path:
    # Anomalib adds a `latest` junction to the versioned folder, so the same
    # checkpoint is reached twice; count distinct physical files only.
    checkpoints = sorted({
        path.resolve()
        for path in (baseline_dir / "engine" / "Patchcore").glob("*/weights/lightning/model.ckpt")
    })
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one frozen PatchCore checkpoint, found {len(checkpoints)}.")
    return checkpoints[0]


def prepare(
    config_path: Path, output_dir: Path, baseline_dir_override: Path | None = None
) -> None:
    config, baseline_dir, manifest_path, rows = preflight(config_path, baseline_dir_override)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Calibration directory must be empty: {output_dir}")
    checkpoint = checkpoint_path(baseline_dir)
    # The large Anomalib import is deliberately delayed until all read-only
    # preflight checks have succeeded. No test artifact is read in this phase.
    import torch
    from anomalib.engine import Engine
    from torch.utils.data import DataLoader

    from iad_w1.manifest_data import ManifestImageDataset
    from run_public_baseline import (
        batch_to_records,
        create_model,
        environment_record,
        manifest_tile_root,
        read_config,
        select_device,
        set_seed,
        write_scores,
    )

    baseline_config = read_config(baseline_dir / "config.snapshot.yaml")
    if baseline_config["model"] != read_json(baseline_dir / "run_metadata.json")["model_config"]:
        raise ValueError("Baseline config snapshot disagrees with run metadata.")
    tile_root = manifest_tile_root(manifest_path)
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    device = select_device(baseline_config, None)
    set_seed(int(baseline_config["run"]["seed"]))
    dataset = ManifestImageDataset(calibration_rows, tile_root, "test", (256, 256))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        pin_memory=device == "gpu",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.snapshot.yaml")
    metadata: dict[str, Any] = {
        "status": "running",
        "started_at_utc": now_utc(),
        "command": [sys.executable, *sys.argv],
        "baseline_run": str(baseline_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest(checkpoint),
        "manifest": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "config_sha256": digest(output_dir / "config.snapshot.yaml"),
        "selected_device": device,
        "environment": environment_record(),
        "test_access_policy": "Prepare phase never reads the test split or test outputs.",
    }
    write_json(output_dir / "run_metadata.json", metadata)
    started = time.perf_counter()
    try:
        model = create_model(baseline_config)
        engine = Engine(
            default_root_dir=output_dir / "engine",
            logger=False,
            accelerator=device,
            devices=1,
            deterministic="warn",
            enable_progress_bar=True,
        )
        predictions = engine.predict(
            model=model,
            dataloaders=loader,
            ckpt_path=checkpoint,
            return_predictions=True,
        )
        if predictions is None:
            raise RuntimeError("Calibration inference did not return predictions.")
        by_path = {
            str((tile_root / row["image_relative_path"]).resolve()): row
            for row in calibration_rows
        }
        collected: list[tuple[dict[str, Any], np.ndarray]] = []
        for batch in predictions:
            records, maps, _, _ = batch_to_records(batch, by_path)
            collected.extend(zip(records, maps, strict=True))
        collected.sort(key=lambda item: item[0]["image_relative_path"])
        if len(collected) != 46 or {item[0]["record_id"] for item in collected} != {
            row["record_id"] for row in calibration_rows
        }:
            raise RuntimeError("Calibration prediction IDs do not match all 46 manifest rows.")
        score_records = [item[0] for item in collected]
        map_array = np.stack([item[1] for item in collected]).astype(np.float32)
        image_scores = np.asarray([item["raw_image_score"] for item in score_records])
        write_scores(output_dir / "calibration_scores.csv", score_records)
        np.savez_compressed(
            output_dir / "calibration_maps.npz",
            record_ids=np.asarray([item["record_id"] for item in score_records], dtype=str),
            anomaly_maps=map_array,
        )
        thresholds = {
            "status": "frozen_before_test_evaluation",
            "frozen_at_utc": now_utc(),
            "manifest_sha256": digest(manifest_path),
            "baseline_run": str(baseline_dir),
            "checkpoint_sha256": digest(checkpoint),
            "config_sha256": digest(output_dir / "config.snapshot.yaml"),
            "calibration_count": len(score_records),
            "calibration_record_ids": [item["record_id"] for item in score_records],
            "quantile": 0.95,
            "threshold_c_raw_image_score": threshold_from_normal_scores(image_scores),
            "threshold_d_smoothed_map_max": threshold_from_normal_maps(map_array),
            "calibration_score_csv_sha256": digest(output_dir / "calibration_scores.csv"),
            "calibration_maps_sha256": digest(output_dir / "calibration_maps.npz"),
        }
        write_json(output_dir / "frozen_thresholds.json", thresholds)
        metadata.update({
            "status": "complete",
            "finished_at_utc": now_utc(),
            "elapsed_seconds": time.perf_counter() - started,
            "calibration_count": len(score_records),
            "frozen_thresholds": str(output_dir / "frozen_thresholds.json"),
        })
        write_json(output_dir / "run_metadata.json", metadata)
        print(json.dumps({"calibration_dir": str(output_dir), "thresholds": thresholds}, ensure_ascii=False))
    except Exception as error:
        metadata.update({"status": "failed", "finished_at_utc": now_utc(), "error": repr(error)})
        write_json(output_dir / "run_metadata.json", metadata)
        raise


def evaluate(
    config_path: Path, output_dir: Path, baseline_dir_override: Path | None = None
) -> None:
    _, baseline_dir, manifest_path, rows = preflight(config_path, baseline_dir_override)
    thresholds = read_json(output_dir / "frozen_thresholds.json")
    metadata = read_json(output_dir / "run_metadata.json")
    if metadata.get("status") != "complete" or thresholds.get("status") != "frozen_before_test_evaluation":
        raise ValueError("Calibration must finish and freeze thresholds before evaluation.")
    if (output_dir / "w4_test_metrics.json").exists():
        raise FileExistsError("W4 test metrics already exist; this command is single-use per frozen run.")
    if thresholds["config_sha256"] != digest(output_dir / "config.snapshot.yaml"):
        raise ValueError("Frozen calibration config changed after threshold selection.")
    if thresholds["config_sha256"] != digest(config_path):
        raise ValueError("Current W4 config differs from the frozen config.")
    if thresholds["manifest_sha256"] != digest(manifest_path):
        raise ValueError("Manifest changed after calibration.")
    if thresholds["baseline_run"] != str(baseline_dir):
        raise ValueError("Calibration and test baseline runs differ.")
    if thresholds["checkpoint_sha256"] != digest(checkpoint_path(baseline_dir)):
        raise ValueError("Frozen model checkpoint changed after calibration.")
    if thresholds["calibration_score_csv_sha256"] != digest(output_dir / "calibration_scores.csv"):
        raise ValueError("Calibration scores changed after threshold selection.")
    if thresholds["calibration_maps_sha256"] != digest(output_dir / "calibration_maps.npz"):
        raise ValueError("Calibration maps changed after threshold selection.")

    test_rows = sorted((row for row in rows if row["split"] == "test"), key=lambda row: row["image_relative_path"])
    score_rows = read_csv(baseline_dir / "raw_image_scores.csv")
    by_id = {row["record_id"]: row for row in score_rows}
    if len(by_id) != len(score_rows) or set(by_id) != {row["record_id"] for row in test_rows}:
        raise ValueError("Frozen test score IDs do not match the locked manifest.")
    with np.load(baseline_dir / "raw_anomaly_maps.npz", allow_pickle=False) as archive:
        map_ids = archive["record_ids"].tolist()
        raw_maps = archive["anomaly_maps"]
        ground_truth_masks = archive["gt_masks"]
    if len(map_ids) != len(set(map_ids)) or set(map_ids) != set(by_id):
        raise ValueError("Frozen test map IDs do not match score/manifest IDs.")
    map_index = {record_id: index for index, record_id in enumerate(map_ids)}
    labels = np.asarray([int(row["label_binary"]) for row in test_rows], dtype=np.uint8)
    c_scores = np.asarray([float(by_id[row["record_id"]]["raw_image_score"]) for row in test_rows])
    d_maps = np.stack([smooth_map_3x3(raw_maps[map_index[row["record_id"]]]) for row in test_rows])
    d_scores = np.asarray([float(item.max()) for item in d_maps])
    sorted_masks = np.stack([ground_truth_masks[map_index[row["record_id"]]] for row in test_rows])
    c_threshold = float(thresholds["threshold_c_raw_image_score"])
    d_threshold = float(thresholds["threshold_d_smoothed_map_max"])
    c_decisions = decisions(c_scores, c_threshold)
    d_decisions = decisions(d_scores, d_threshold)

    # Test labels are used only here, after thresholds/config have been frozen.
    from sklearn.metrics import roc_auc_score

    c_counts = confusion_counts(labels, c_decisions)
    d_counts = confusion_counts(labels, d_decisions)
    metrics: dict[str, Any] = {
        "status": "complete",
        "evaluated_at_utc": now_utc(),
        "scope": "MVTec AD tile public benchmark only; not internal A-E results",
        "manifest_sha256": digest(manifest_path),
        "baseline_run": str(baseline_dir),
        "frozen_thresholds_sha256": digest(output_dir / "frozen_thresholds.json"),
        "test_count": len(test_rows),
        "test_normal_count": int((labels == 0).sum()),
        "test_anomalous_count": int((labels == 1).sum()),
        "C_raw_image_score": {
            "threshold": c_threshold,
            "image_auroc": float(roc_auc_score(labels, c_scores)),
            "counts": c_counts,
            "fpr": c_counts["fp"] / (c_counts["fp"] + c_counts["tn"]),
            "recall": c_counts["tp"] / (c_counts["tp"] + c_counts["fn"]),
        },
        "D_fixed_3x3_smoothed_map": {
            "threshold": d_threshold,
            "image_auroc": float(roc_auc_score(labels, d_scores)),
            "pixel_auroc": float(roc_auc_score(sorted_masks.ravel(), d_maps.ravel())),
            "counts": d_counts,
            "fpr": d_counts["fp"] / (d_counts["fp"] + d_counts["tn"]),
            "recall": d_counts["tp"] / (d_counts["tp"] + d_counts["fn"]),
        },
    }
    score_path = output_dir / "w4_test_scores.csv"
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "record_id", "image_relative_path", "label_binary", "c_raw_score", "c_decision",
            "d_smoothed_max", "d_decision",
        ])
        writer.writeheader()
        for index, row in enumerate(test_rows):
            writer.writerow({
                "record_id": row["record_id"],
                "image_relative_path": row["image_relative_path"],
                "label_binary": row["label_binary"],
                "c_raw_score": float(c_scores[index]),
                "c_decision": int(c_decisions[index]),
                "d_smoothed_max": float(d_scores[index]),
                "d_decision": int(d_decisions[index]),
            })
    metrics["test_scores_sha256"] = digest(score_path)
    write_json(output_dir / "w4_test_metrics.json", metrics)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["preflight", "prepare", "evaluate"])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "public_tile_w4_normal_only.yaml"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        help="Path to the frozen PatchCore baseline run when it is stored outside this public repository.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if args.phase == "preflight":
        config, baseline_dir, manifest_path, rows = preflight(config_path, args.baseline_dir)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "config": str(config_path),
                    "baseline_run": str(baseline_dir),
                    "manifest": str(manifest_path),
                    "manifest_sha256": digest(manifest_path),
                    "split_counts": {
                        split: sum(row["split"] == split for row in rows)
                        for split in ("model", "calibration", "test")
                    },
                    "selection_policy": config["selection_policy"],
                },
                ensure_ascii=False,
            )
        )
    elif args.phase == "prepare":
        prepare(config_path, output_dir, args.baseline_dir)
    else:
        evaluate(config_path, output_dir, args.baseline_dir)


if __name__ == "__main__":
    main()
