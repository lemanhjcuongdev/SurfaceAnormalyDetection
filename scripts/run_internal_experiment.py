"""Run the locked internal v1 main experiment (protocol D-096): variants A-E.

Gate: the split built by `build_internal_split.py` must be PASS and
experiment-ready, and its CSV hashes must match the validation record. Model
fitting uses only `model` images, thresholds use only `calibration` images, and
test labels/masks are read only after every score and threshold is fixed.
Raw outputs, tables and error cases stay local (runs/ is Git-ignored).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
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

from iad_w1.normal_calibration import confusion_counts, decisions, smooth_map_3x3, upper_empirical_quantile

ROI_MODES = {"full_frame", "roi"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


# --------------------------------------------------------------------------- metrics

def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    # scikit-learn >= 1.6 returns NaN with a warning instead of raising; the
    # bootstrap relies on ValueError to skip one-class resamples.
    if np.unique(labels).size < 2:
        raise ValueError("Both OK and NG images are required for ranking metrics.")
    result: dict[str, float] = {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_auprc": float(average_precision_score(labels, scores)),
    }
    if threshold is not None:
        counts = confusion_counts(labels, decisions(scores, threshold))
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result.update({
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "recall": recall,
            "precision": precision,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        })
    return result


def cluster_bootstrap(
    groups: np.ndarray,
    statistic,
    resamples: int,
    level: float,
    seed: int,
) -> dict[str, Any]:
    """Percentile CI from resampling whole groups (physical samples) with replacement.

    `statistic(indices)` returns a dict of floats for the image indices of one
    resample; resamples where it raises ValueError (e.g. one class only) are
    skipped and counted.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    collected: dict[str, list[float]] = {}
    skipped = 0
    for _ in range(resamples):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        indices = np.concatenate([members[group] for group in chosen])
        try:
            values = statistic(indices)
        except ValueError:
            skipped += 1
            continue
        for key, value in values.items():
            collected.setdefault(key, []).append(float(value))
    alpha = (1.0 - level) / 2.0
    return {
        "intervals": {
            key: [float(np.quantile(v, alpha)), float(np.quantile(v, 1.0 - alpha))] for key, v in collected.items()
        },
        "valid_resamples": resamples - skipped,
        "skipped_resamples": skipped,
        "groups": int(unique.size),
    }


# --------------------------------------------------------------------------- image loading

def load_view(path: Path, mode: str, roi: list[int] | None):
    from PIL import Image, ImageOps

    if mode not in ROI_MODES:
        raise ValueError(f"Unknown input mode {mode!r}")
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")  # ROI-v0.1 is defined after EXIF orientation
        if mode == "roi":
            x, y, w, h = roi
            rgb = rgb.crop((x, y, min(rgb.width, x + w), min(rgb.height, y + h)))
        return rgb.copy()


def load_mask(path: Path, mode: str, roi: list[int] | None, size: tuple[int, int]) -> np.ndarray:
    """Masks are drawn on the EXIF-transposed full frame; non-zero pixels mark the defect."""
    from PIL import Image

    with Image.open(path) as image:
        mask = image.convert("L")
        if mode == "roi":
            x, y, w, h = roi
            mask = mask.crop((x, y, min(mask.width, x + w), min(mask.height, y + h)))
        mask = mask.resize((size[1], size[0]), Image.Resampling.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def to_tensor(image, size: tuple[int, int]):
    import torch
    from PIL import Image

    from iad_w1.manifest_data import IMAGENET_MEAN, IMAGENET_STD

    resized = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


# --------------------------------------------------------------------------- models

def fit_and_score(family: str, config: dict[str, Any], seed: int, train: list, score_sets: dict[str, list], device) -> dict[str, Any]:
    import torch

    from run_public_baseline import set_seed

    set_seed(seed)
    if family == "patchcore":
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        params = config["patchcore"]
        model = PatchcoreModel(backbone=params["backbone"], layers=params["layers"], pre_trained=True,
                               num_neighbors=params["num_neighbors"]).to(device)
    else:
        from anomalib.models.image.padim.torch_model import PadimModel

        params = config["padim"]
        model = PadimModel(backbone=params["backbone"], layers=params["layers"], pre_trained=True).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    with torch.no_grad():
        for tensor in train:
            model(tensor.unsqueeze(0).to(device))
        if family == "patchcore":
            model.subsample_embedding(config["patchcore"]["coreset_sampling_ratio"])
        else:
            model.fit()
    fit_seconds = time.perf_counter() - started
    model.eval()
    outputs: dict[str, Any] = {"fit_seconds": fit_seconds}
    with torch.inference_mode():
        for name, tensors in score_sets.items():
            started = time.perf_counter()
            scores, maps = [], []
            for tensor in tensors:
                result = model(tensor.unsqueeze(0).to(device))
                scores.append(float(result.pred_score.reshape(-1)[0].item()))
                maps.append(result.anomaly_map.detach().squeeze().cpu().numpy().astype(np.float32))
            outputs[name] = {"scores": np.asarray(scores), "maps": np.stack(maps)}
            outputs[f"{name}_seconds"] = time.perf_counter() - started
    if family == "patchcore":
        outputs["memory_bank_shape"] = list(model.memory_bank.shape)
    outputs["peak_gpu_bytes"] = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


# --------------------------------------------------------------------------- reporting helpers

def jpeg_b64(image, width: int = 420) -> str:
    from PIL import Image

    height = round(image.height * width / image.width)
    buffer = io.BytesIO()
    image.resize((width, height), Image.Resampling.BILINEAR).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def overlay_image(image, score_map: np.ndarray, scale_max: float):
    import cv2
    from PIL import Image

    base = np.asarray(image.resize((420, round(image.height * 420 / image.width)), Image.Resampling.BILINEAR))
    heat = cv2.resize(score_map, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    colour = cv2.applyColorMap((np.clip(heat / scale_max, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return Image.fromarray(cv2.addWeighted(base, 0.55, cv2.cvtColor(colour, cv2.COLOR_BGR2RGB), 0.45, 0))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v1.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    parser.add_argument("--data-root", type=Path, required=True, help="Root that relative annotation_path values resolve against.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", help="Subset of the locked model_seeds (smoke runs only).")
    parser.add_argument("--bootstrap-resamples", type=int, help="Override for smoke runs only; recorded in metadata.")
    args = parser.parse_args()

    import torch

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds or config["model_seeds"]
    if not set(seeds) <= set(config["model_seeds"]) or config["primary_seed"] not in seeds:
        raise ValueError("Seeds must be a subset of the locked model_seeds and include the primary seed.")
    ci = dict(config["metrics"]["confidence_interval"])
    if args.bootstrap_resamples:
        ci["resamples"] = args.bootstrap_resamples
    official = args.seeds is None and args.bootstrap_resamples is None

    validation_path = args.split_dir / "split_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    metadata_path, assignments_path = args.split_dir / "metadata.csv", args.split_dir / "assignments.csv"
    if validation.get("integrity_status") != "PASS" or not validation.get("experiment_ready"):
        raise SystemExit("Split is not PASS/experiment_ready; fill sample_id/surface_region_id and rebuild the split.")
    if validation["metadata_sha256"] != sha256_file(metadata_path) or validation["assignments_sha256"] != sha256_file(assignments_path):
        raise SystemExit("Split CSVs changed after validation; rebuild the split.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")

    metadata = {row["image_id"]: row for row in read_csv(metadata_path)}
    split_of = {row["image_id"]: row["split"] for row in read_csv(assignments_path)}
    by_split = {name: sorted(i for i, s in split_of.items() if s == name) for name in ("model", "calibration", "test")}

    size = tuple(config["input_size"])
    roi = config["variants"]["B"]["roi"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.run_dir.mkdir(parents=True)
    run_meta: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config.get("protocol", "D-096"),
        "official_run": official,
        "command": [sys.executable, *sys.argv],
        "config_sha256": sha256_file(args.config),
        "split_validation_sha256": sha256_file(validation_path),
        "metadata_sha256": validation["metadata_sha256"],
        "assignments_sha256": validation["assignments_sha256"],
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "seeds": seeds,
        "bootstrap": ci,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    meta_path = args.run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 1. Preprocess each image once per input mode (labels are not touched here).
    tensors: dict[str, dict[str, Any]] = {mode: {} for mode in ROI_MODES}
    for image_id in sorted(metadata):
        if split_of[image_id] not in by_split:
            continue
        path = Path(metadata[image_id]["image_path"])
        for mode in ROI_MODES:
            tensors[mode][image_id] = to_tensor(load_view(path, mode, roi), size)

    # 2. Fit on model images, score calibration and test images.
    runs: dict[str, dict[str, Any]] = {}
    plan = [("patchcore", "full_frame", s) for s in seeds] + [("patchcore", "roi", s) for s in seeds]
    plan.append(("padim", "roi", config["primary_seed"]))
    timing: dict[str, Any] = {}
    for family, mode, seed in plan:
        key = f"{family}_{mode}_seed{seed}"
        outputs = fit_and_score(
            family, config, seed,
            [tensors[mode][i] for i in by_split["model"]],
            {split: [tensors[mode][i] for i in by_split[split]] for split in ("calibration", "test")},
            device,
        )
        runs[key] = outputs
        timing[key] = {k: v for k, v in outputs.items() if k.endswith("seconds") or k in ("memory_bank_shape", "peak_gpu_bytes")}
        np.savez_compressed(
            args.run_dir / f"raw_{key}.npz",
            calibration_ids=np.asarray(by_split["calibration"]), test_ids=np.asarray(by_split["test"]),
            calibration_scores=outputs["calibration"]["scores"], test_scores=outputs["test"]["scores"],
            calibration_maps=outputs["calibration"]["maps"], test_maps=outputs["test"]["maps"],
        )
        print(f"done {key}: fit {outputs['fit_seconds']:.1f}s", flush=True)
    run_meta["timing"] = timing

    # 3. Derive variant scores and freeze thresholds from calibration only.
    q = 0.95

    def variant_outputs(seed: int) -> dict[str, dict[str, Any]]:
        full, roi_run = runs[f"patchcore_full_frame_seed{seed}"], runs[f"patchcore_roi_seed{seed}"]
        smooth_cal = np.stack([smooth_map_3x3(m) for m in roi_run["calibration"]["maps"]])
        smooth_test = np.stack([smooth_map_3x3(m) for m in roi_run["test"]["maps"]])
        result = {
            "A": {"scores": full["test"]["scores"], "maps": None, "threshold": None},
            "B": {"scores": roi_run["test"]["scores"], "maps": roi_run["test"]["maps"], "threshold": None},
            "C": {"scores": roi_run["test"]["scores"], "maps": roi_run["test"]["maps"],
                  "threshold": upper_empirical_quantile(roi_run["calibration"]["scores"], q)},
            "D": {"scores": smooth_test.reshape(len(smooth_test), -1).max(axis=1), "maps": smooth_test,
                  "threshold": upper_empirical_quantile(smooth_cal.reshape(len(smooth_cal), -1).max(axis=1), q)},
        }
        if seed == config["primary_seed"]:
            padim = runs[f"padim_roi_seed{seed}"]
            result["E"] = {"scores": padim["test"]["scores"], "maps": padim["test"]["maps"],
                           "threshold": upper_empirical_quantile(padim["calibration"]["scores"], q)}
        return result

    variants = {seed: variant_outputs(seed) for seed in seeds}
    frozen = {str(seed): {k: v["threshold"] for k, v in variants[seed].items()} for seed in seeds}
    (args.run_dir / "frozen_thresholds.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")

    # 4. Only now read test labels, sample groups and masks.
    test_ids = by_split["test"]
    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids], dtype=np.uint8)
    groups = np.asarray([metadata[i]["sample_id"] for i in test_ids])
    mask_paths = [metadata[i]["annotation_path"].strip() for i in test_ids]
    ng_with_mask = config["metrics"].get("pixel_in_main_run", True) and all(mask_paths[k] for k in np.flatnonzero(labels == 1))
    masks = None
    if ng_with_mask:
        masks = np.stack([
            load_mask((args.data_root / p) if p else None, "roi", roi, size) if p else np.zeros(size, np.uint8)
            for p in mask_paths
        ])

    def pixel_auroc(maps: np.ndarray, indices: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score

        truth = masks[indices].ravel()
        if np.unique(truth).size < 2:
            raise ValueError("Pixel AUROC needs defect and normal pixels.")
        return float(roc_auc_score(truth, maps[indices].ravel()))

    primary = config["primary_seed"]
    main_rows, seed_rows, ablation_rows, localization_rows = [], [], [], []
    for seed in seeds:
        for name, item in variants[seed].items():
            metrics = image_metrics(labels, item["scores"], item["threshold"])
            seed_rows.append({"seed": seed, "variant": name, "threshold": item["threshold"], **metrics})
    for name, item in variants[primary].items():
        metrics = image_metrics(labels, item["scores"], item["threshold"])

        def stat(indices, item=item):
            return image_metrics(labels[indices], item["scores"][indices], item["threshold"])

        boot = cluster_bootstrap(groups, stat, ci["resamples"], ci["level"], ci["seed"])
        row = {"variant": name, "threshold": item["threshold"], **metrics, "ci_groups": boot["groups"],
               "ci_valid_resamples": boot["valid_resamples"]}
        for key, (low, high) in boot["intervals"].items():
            if key in ("image_auroc", "image_auprc", "fpr", "recall", "precision", "f1"):
                row[f"{key}_ci_low"], row[f"{key}_ci_high"] = low, high
        main_rows.append(row)
        if masks is not None and item["maps"] is not None and name in config["metrics"]["pixel_variants"]:
            all_idx = np.arange(len(test_ids))
            pboot = cluster_bootstrap(groups, lambda idx, m=item["maps"]: {"pixel_auroc": pixel_auroc(m, idx)},
                                      ci["resamples"], ci["level"], ci["seed"])
            low, high = pboot["intervals"].get("pixel_auroc", [None, None])
            localization_rows.append({"variant": name, "pixel_auroc": pixel_auroc(item["maps"], all_idx),
                                      "ci_low": low, "ci_high": high})
    for pair in config["metrics"]["ablation_differences"]:
        later, earlier = pair.split("-")
        a, b = variants[primary][later], variants[primary][earlier]

        def diff(indices, a=a, b=b):
            ma = image_metrics(labels[indices], a["scores"][indices], a["threshold"])
            mb = image_metrics(labels[indices], b["scores"][indices], b["threshold"])
            return {k: ma[k] - mb[k] for k in ma if k in mb and k not in ("tp", "fp", "fn", "tn")}

        point = diff(np.arange(len(test_ids)))
        boot = cluster_bootstrap(groups, diff, ci["resamples"], ci["level"], ci["seed"])
        for metric, value in point.items():
            low, high = boot["intervals"][metric]
            ablation_rows.append({"comparison": pair, "metric": metric, "difference": value, "ci_low": low, "ci_high": high})

    write_rows(args.run_dir / "table5_main_results.csv", main_rows)
    write_rows(args.run_dir / "table6_ablation.csv", ablation_rows)
    write_rows(args.run_dir / "seed_variability.csv", seed_rows)
    if localization_rows:
        write_rows(args.run_dir / "localization.csv", localization_rows)

    # 5. Error cases for the main pipeline D (primary seed); local only.
    d = variants[primary]["D"]
    predicted = decisions(d["scores"], d["threshold"])
    order_ng = [k for k in np.argsort(d["scores"]) if labels[k] == 1][:3]
    cases = []
    for k in range(len(test_ids)):
        kind = []
        if labels[k] == 1 and predicted[k] == 0:
            kind.append("false_negative")
        if labels[k] == 0 and predicted[k] == 1:
            kind.append("false_positive")
        if k in order_ng:
            kind.append("lowest_scored_ng")
        if kind:
            cases.append({"image_id": test_ids[k], "sample_id": groups[k], "label": "NG" if labels[k] else "OK",
                          "d_score": float(d["scores"][k]), "threshold": d["threshold"], "case": ";".join(kind)})
    write_rows(args.run_dir / "error_cases.csv", cases or [{"image_id": "", "case": "none"}])
    scale = float(np.quantile(d["maps"].reshape(len(d["maps"]), -1).max(axis=1), 0.9))
    cards = []
    for case in cases:
        k = test_ids.index(case["image_id"])
        view = load_view(Path(metadata[case["image_id"]]["image_path"]), "roi", roi)
        cards.append(
            f"<div class='card'><h3>{html.escape(case['image_id'])} — {case['label']} — {case['case']}</h3>"
            f"<p>điểm D {case['d_score']:.3f} · ngưỡng {case['threshold']:.3f} · vật mẫu {html.escape(case['sample_id'])}</p>"
            f"<img src='data:image/jpeg;base64,{jpeg_b64(view)}'>"
            f"<img src='data:image/jpeg;base64,{jpeg_b64(overlay_image(view, d['maps'][k], scale))}'></div>"
        )
    (args.run_dir / "error_cases.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Error cases D</title>"
        "<style>body{font-family:sans-serif;margin:16px}.card{border:1px solid #ccc;padding:8px;margin:8px 0}"
        "img{max-width:48%;margin-right:8px}</style><h1>Error cases — biến thể D, seed chính</h1>"
        + ("".join(cards) or "<p>Không có FN/FP.</p>"), encoding="utf-8")

    run_meta.update({
        "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_ok": int((labels == 0).sum()),
        "test_ng": int(labels.sum()),
        "test_sample_groups": int(np.unique(groups).size),
        "localization_computed": masks is not None,
    })
    meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(args.run_dir), "main": main_rows}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
