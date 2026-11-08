"""Paired comparisons, IoU/Dice and AU-PRO on the frozen internal maps (D-117).

Scores, maps and thresholds come from the raw outputs of run01 (B, D, E) and
efficientad01 (F); nothing is fitted, tuned or selected here. IoU/Dice use D, E, F
only: their image score equals the map maximum, so the locked threshold is on the
map scale. PatchCore's raw image score (B/C) is not.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iad_w1.normal_calibration import smooth_map_3x3  # noqa: E402
from run_internal_experiment import cluster_bootstrap, image_metrics, load_mask, read_csv, sha256_file, write_rows  # noqa: E402
from run_internal_localization import SOURCES, auroc_from_counts, binned_counts, pixel_auroc, registry_ng  # noqa: E402

IMAGE_RANKING_PAIRS = [("B", "E"), ("D", "E")]
IMAGE_DECISION_PAIRS = [("E", "D"), ("F", "D"), ("F", "E")]
PIXEL_PAIRS = [("E", "B"), ("F", "B"), ("F", "E")]
THRESHOLDED = ["D", "E", "F"]
PRO_LIMITS = [0.30, 0.05]


def pixel_counts(masks: np.ndarray, maps: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    """Per-image pixel counts; a pixel is positive when strictly above the threshold, as for images."""
    truth, pred = masks.astype(bool), maps > threshold
    axes = (1, 2)
    return {"tp": (pred & truth).sum(axes), "fp": (pred & ~truth).sum(axes),
            "fn": (~pred & truth).sum(axes), "tn": (~pred & ~truth).sum(axes)}


def iou_dice(tp: float, fp: float, fn: float) -> tuple[float, float]:
    if tp + fp + fn == 0:
        raise ValueError("IoU and Dice need at least one positive pixel.")
    return tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)


def pro_curves(masks: np.ndarray, maps: np.ndarray, bins: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-image normal-pixel histograms, summed region survival curves and region counts.

    Regions are 8-connected components of the mask. Entry b of a survival curve is the
    fraction of the region's pixels in score bins above b.
    """
    from scipy import ndimage

    edges = np.quantile(maps, np.linspace(0, 1, bins + 1)[1:-1])
    binned = np.searchsorted(edges, maps.reshape(len(maps), -1)).reshape(maps.shape)
    neg = np.zeros((len(maps), bins), np.int64)
    survival = np.zeros((len(maps), bins))
    regions = np.zeros(len(maps), np.int64)
    for k, mask in enumerate(masks.astype(bool)):
        neg[k] = np.bincount(binned[k][~mask], minlength=bins)
        labelled, count = ndimage.label(mask, structure=np.ones((3, 3)))
        regions[k] = count
        for region in range(1, count + 1):
            hist = np.bincount(binned[k][labelled == region], minlength=bins)
            survival[k] += 1.0 - np.cumsum(hist) / hist.sum()
    return neg, survival, regions


def aupro(neg: np.ndarray, survival: np.ndarray, regions: int, limit: float) -> float:
    """Area under the per-region-overlap curve up to an FPR limit, normalised to 0-1."""
    if regions == 0 or neg.sum() == 0:
        raise ValueError("AU-PRO needs defect regions and normal pixels.")
    fpr = (1.0 - np.cumsum(neg) / neg.sum())[::-1]
    pro = (survival / regions)[::-1]
    keep = fpr < limit
    x = np.concatenate([fpr[keep], [limit]])
    y = np.concatenate([pro[keep], [np.interp(limit, fpr, pro)]])
    return float(np.trapezoid(y, x) / limit)


def pair_rows(point: dict, boot: dict, base: dict) -> list[dict]:
    return [{**base, "metric": key, "difference": value, "ci_low": boot["intervals"][key][0],
             "ci_high": boot["intervals"][key][1]} for key, value in point.items()]


def check_regression(run: Path, rows: list[dict], key: str) -> None:
    """Stop if a stored result is not reproduced exactly by the same bootstrap."""
    stored = {tuple(r[k] for k in key.split(",")): r for r in csv.DictReader(open(run, encoding="utf-8"))}
    for row in rows:
        ref = stored[tuple(str(row[k]) for k in key.split(","))]
        for field in ("difference", "ci_low", "ci_high"):
            if abs(float(ref[field]) - row[field]) > 1e-9:
                raise SystemExit(f"Regression check failed for {row}: stored {ref[field]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v1.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "metadata" / "INTERNAL_IMAGE_REGISTRY.xlsx")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261101-run01")
    parser.add_argument("--efficientad-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261101-efficientad01")
    parser.add_argument("--localization-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261102-localization01")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ci = config["metrics"]["confidence_interval"]
    roi, size = config["variants"]["B"]["roi"], tuple(config["input_size"])
    boot_args = (ci["resamples"], ci["level"], ci["seed"])
    metadata = {row["image_id"]: row for row in read_csv(args.split_dir / "metadata.csv")}
    ng_registry = registry_ng(args.registry)
    runs = {"reference_run": args.reference_run, "efficientad_run": args.efficientad_run}
    thresholds = {}
    for run in runs.values():
        thresholds.update(json.loads((run / "frozen_thresholds.json").read_text(encoding="utf-8"))["42"])

    maps, scores, test_ids = {}, {}, None
    for variant, (run_key, name, smooth) in SOURCES.items():
        raw = np.load(runs[run_key] / name)
        ids = [str(i) for i in raw["test_ids"]]
        if test_ids is None:
            test_ids = ids
        elif ids != test_ids:
            raise ValueError(f"Test order differs for {variant}")
        test_maps = raw["test_maps"]
        if smooth:
            test_maps = np.stack([smooth_map_3x3(m) for m in test_maps])
        maps[variant] = test_maps
        scores[variant] = test_maps.reshape(len(test_maps), -1).max(axis=1) if smooth else raw["test_scores"]
    for variant in THRESHOLDED:
        if not np.allclose(scores[variant], maps[variant].reshape(len(test_ids), -1).max(axis=1)):
            raise SystemExit(f"Image score of {variant} is not the map maximum; the threshold is off the map scale.")

    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids], dtype=np.uint8)
    groups = np.asarray([metadata[i]["sample_id"] for i in test_ids])
    ng_idx = np.flatnonzero(labels == 1)
    masks, mask_files = np.zeros((len(test_ids), *size), np.uint8), {}
    for k in ng_idx:
        record = ng_registry[test_ids[k]]
        relative = record["annotation_path"].strip()
        if record.get("mask_status") != "da_ve" or test_ids[k][4:] not in relative:
            raise ValueError(f"Mask path missing or mismatched for {test_ids[k]}: {relative!r}")
        mask_files[test_ids[k]] = {"path": relative, "sha256": sha256_file(args.data_root / relative)}
        masks[k] = load_mask(args.data_root / relative, "roi", roi, size)

    args.run_dir.mkdir(parents=True)
    all_idx = np.arange(len(test_ids))

    # 1. Image-level paired differences on the same resamples.
    image_rows = []
    for a, b in IMAGE_RANKING_PAIRS + [("F", "B"), ("F", "E")]:
        def diff(idx, a=a, b=b):
            ma, mb = image_metrics(labels[idx], scores[a][idx], None), image_metrics(labels[idx], scores[b][idx], None)
            return {k: ma[k] - mb[k] for k in ("image_auroc", "image_auprc")}
        image_rows += pair_rows(diff(all_idx), cluster_bootstrap(groups, diff, *boot_args), {"comparison": f"{a}-{b}"})
    check_regression(args.efficientad_run / "paired_comparisons.csv",
                     [r for r in image_rows if r["comparison"] in ("F-B", "F-E")], "comparison,metric")
    for a, b in IMAGE_DECISION_PAIRS:
        def diff(idx, a=a, b=b):
            ma = image_metrics(labels[idx], scores[a][idx], thresholds[a])
            mb = image_metrics(labels[idx], scores[b][idx], thresholds[b])
            return {k: ma[k] - mb[k] for k in ("recall", "fpr", "f1")}
        image_rows += pair_rows(diff(all_idx), cluster_bootstrap(groups, diff, *boot_args), {"comparison": f"{a}-{b}"})
    write_rows(args.run_dir / "paired_image.csv", image_rows)

    # 2. Pixel-AUROC paired differences, on all test images and on NG images only.
    counts = {v: binned_counts(masks, maps[v]) for v in SOURCES}
    stored = {r["variant"]: r for r in csv.DictReader(open(args.localization_run / "localization.csv", encoding="utf-8"))}
    for variant, (pos, neg) in counts.items():
        boot = cluster_bootstrap(groups, lambda idx: {"p": auroc_from_counts(pos[idx].sum(0), neg[idx].sum(0))}, *boot_args)
        if abs(boot["intervals"]["p"][0] - float(stored[variant]["ci_low"])) > 1e-9:
            raise SystemExit(f"Regression check failed for pixel-AUROC CI of {variant}")

    def on_ng(idx):
        sub = idx[labels[idx] == 1]
        if sub.size == 0:
            raise ValueError("No NG image in this resample.")
        return sub

    pixel_rows = []
    for a, b in PIXEL_PAIRS:
        (pa, na), (pb, nb) = counts[a], counts[b]
        def diff(idx):
            return {"pixel_auroc": auroc_from_counts(pa[idx].sum(0), na[idx].sum(0)) - auroc_from_counts(pb[idx].sum(0), nb[idx].sum(0))}
        point = {"pixel_auroc": pixel_auroc(masks, maps[a], all_idx) - pixel_auroc(masks, maps[b], all_idx)}
        pixel_rows += pair_rows(point, cluster_bootstrap(groups, diff, *boot_args), {"comparison": f"{a}-{b}", "scope": "all_99"})
        def diff_ng(idx):
            return diff(on_ng(idx))
        point = {"pixel_auroc": pixel_auroc(masks, maps[a], ng_idx) - pixel_auroc(masks, maps[b], ng_idx)}
        pixel_rows += pair_rows(point, cluster_bootstrap(groups, diff_ng, *boot_args), {"comparison": f"{a}-{b}", "scope": "ng_13"})
    write_rows(args.run_dir / "paired_pixel.csv", pixel_rows)

    # 3. IoU and Dice at the locked thresholds.
    region_rows, per_image = [], []
    for variant in THRESHOLDED:
        c = pixel_counts(masks, maps[variant], thresholds[variant])
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        per_iou, per_dice = np.zeros(len(all_idx)), np.zeros(len(all_idx))
        for k in ng_idx:
            per_iou[k], per_dice[k] = iou_dice(tp[k], fp[k], fn[k])
            per_image.append({"variant": variant, "image_id": test_ids[k], "tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k]),
                              "iou": per_iou[k], "dice": per_dice[k]})

        def stat(idx):
            iou, dice = iou_dice(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())
            sub = on_ng(idx)
            return {"iou_pooled": iou, "dice_pooled": dice, "iou_mean_ng": per_iou[sub].mean(), "dice_mean_ng": per_dice[sub].mean()}
        boot = cluster_bootstrap(groups, stat, *boot_args)
        row = {"variant": variant, "threshold": thresholds[variant], "tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum()), "tn": int(tn.sum()),
               "ng_images_with_overlap": int((tp[ng_idx] > 0).sum()), "ok_images_with_positive_pixels": int((fp[labels == 0] > 0).sum()),
               "positive_pixel_rate_ok_images": float(fp[labels == 0].sum() / (fp[labels == 0].sum() + tn[labels == 0].sum())),
               "ci_valid_resamples": boot["valid_resamples"]}
        for key, value in stat(all_idx).items():
            row[key], row[f"{key}_ci_low"], row[f"{key}_ci_high"] = float(value), *boot["intervals"][key]
        region_rows.append(row)
    write_rows(args.run_dir / "iou_dice.csv", region_rows)
    write_rows(args.run_dir / "iou_dice_per_image.csv", per_image)

    # 4. AU-PRO, threshold-free.
    pro_rows = []
    for variant in SOURCES:
        neg, survival, regions = pro_curves(masks, maps[variant])
        def stat(idx, neg=neg, survival=survival, regions=regions):
            return {f"aupro_{limit:.2f}": aupro(neg[idx].sum(0), survival[idx].sum(0), int(regions[idx].sum()), limit) for limit in PRO_LIMITS}
        boot = cluster_bootstrap(groups, stat, *boot_args)
        row = {"variant": variant, "regions": int(regions.sum()), "ci_valid_resamples": boot["valid_resamples"]}
        for key, value in stat(all_idx).items():
            row[key], row[f"{key}_ci_low"], row[f"{key}_ci_high"] = value, *boot["intervals"][key]
        pro_rows.append(row)
    write_rows(args.run_dir / "aupro.csv", pro_rows)

    meta = {
        "protocol": "D-117", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256_file(args.config),
        "reference_run": str(args.reference_run), "efficientad_run": str(args.efficientad_run),
        "raw_sha256": {v: sha256_file(runs[s[0]] / s[1]) for v, s in SOURCES.items()},
        "thresholds": {v: thresholds[v] for v in THRESHOLDED}, "mask_files": mask_files, "bootstrap": ci,
        "regression_checks": "F-B and F-E image pairs, pixel-AUROC CI lower bounds reproduced",
    }
    (args.run_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paired_image": image_rows, "paired_pixel": pixel_rows, "iou_dice": region_rows, "aupro": pro_rows},
                     ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
