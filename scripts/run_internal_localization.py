"""Post-hoc localization and per-defect-type analysis for the locked internal test set (D-110).

Anomaly maps, image scores and thresholds come from the frozen raw outputs of
run01 (B, D, E) and efficientad01 (F); nothing here is fitted or tuned. Masks are
read from the registry's `annotation_path` only after those maps were frozen.
"""
from __future__ import annotations

import argparse
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
from run_internal_experiment import cluster_bootstrap, load_mask, load_view, overlay_image, read_csv, sha256_file, write_rows  # noqa: E402

# variant -> (run key, raw file, smooth maps)
SOURCES = {
    "B": ("reference_run", "raw_patchcore_roi_seed42.npz", False),
    "D": ("reference_run", "raw_patchcore_roi_seed42.npz", True),
    "E": ("reference_run", "raw_padim_roi_seed42.npz", False),
    "F": ("efficientad_run", "raw_efficientad_roi_seed42.npz", False),
}


def registry_ng(path: Path) -> dict[str, dict[str, str]]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True)
    rows = workbook["Anh_NG"].iter_rows(values_only=True)
    header = next(rows)
    out = {}
    for row in rows:
        record = {str(k): ("" if v is None else str(v)) for k, v in zip(header, row) if k}
        if record.get("image_id"):
            out[record["image_id"]] = record
    return out


def pixel_auroc(masks: np.ndarray, maps: np.ndarray, indices: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    truth = masks[indices].ravel()
    if np.unique(truth).size < 2:
        raise ValueError("Pixel AUROC needs defect and normal pixels.")
    return float(roc_auc_score(truth, maps[indices].ravel()))


def binned_counts(masks: np.ndarray, maps: np.ndarray, bins: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Per-image defect/normal pixel histograms on shared score bins, for fast bootstrap AUROC."""
    edges = np.quantile(maps, np.linspace(0, 1, bins + 1)[1:-1])
    binned = np.searchsorted(edges, maps.reshape(len(maps), -1))
    flat = masks.reshape(len(masks), -1).astype(bool)
    pos = np.stack([np.bincount(b[f], minlength=bins) for b, f in zip(binned, flat)])
    neg = np.stack([np.bincount(b[~f], minlength=bins) for b, f in zip(binned, flat)])
    return pos, neg


def auroc_from_counts(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n0 = pos.sum(), neg.sum()
    if n1 == 0 or n0 == 0:
        raise ValueError("Pixel AUROC needs defect and normal pixels.")
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (n1 * n0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v1.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "metadata" / "INTERNAL_IMAGE_REGISTRY.xlsx")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261101-run01")
    parser.add_argument("--efficientad-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261101-efficientad01")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ci = config["metrics"]["confidence_interval"]
    roi, size = config["variants"]["B"]["roi"], tuple(config["input_size"])
    metadata = {row["image_id"]: row for row in read_csv(args.split_dir / "metadata.csv")}
    ng_registry = registry_ng(args.registry)
    runs = {"reference_run": args.reference_run, "efficientad_run": args.efficientad_run}

    thresholds = {}
    for key, run in runs.items():
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

    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids], dtype=np.uint8)
    groups = np.asarray([metadata[i]["sample_id"] for i in test_ids])
    ng_idx = np.flatnonzero(labels == 1)
    mask_files, masks = {}, np.zeros((len(test_ids), *size), np.uint8)
    for k in ng_idx:
        image_id = test_ids[k]
        record = ng_registry[image_id]
        relative = record["annotation_path"].strip()
        if not relative or image_id[4:] not in relative or record.get("mask_status") != "da_ve":
            raise ValueError(f"Mask path missing or mismatched for {image_id}: {relative!r}")
        path = args.data_root / relative
        mask_files[image_id] = {"path": relative, "sha256": sha256_file(path)}
        masks[k] = load_mask(path, "roi", roi, size)
        if masks[k].sum() == 0:
            raise ValueError(f"Mask is empty inside the ROI for {image_id}")

    args.run_dir.mkdir(parents=True)
    all_idx = np.arange(len(test_ids))
    localization, per_image, per_type = [], [], []
    for variant in SOURCES:
        pos, neg = binned_counts(masks, maps[variant])
        boot = cluster_bootstrap(groups, lambda idx: {"pixel_auroc": auroc_from_counts(pos[idx].sum(0), neg[idx].sum(0))},
                                 ci["resamples"], ci["level"], ci["seed"])
        low, high = boot["intervals"]["pixel_auroc"]
        ng_only = pixel_auroc(masks, maps[variant], ng_idx)
        localization.append({"variant": variant, "pixel_auroc": pixel_auroc(masks, maps[variant], all_idx),
                             "ci_low": low, "ci_high": high, "ci_valid_resamples": boot["valid_resamples"],
                             "pixel_auroc_ng_images_only": ng_only})
        for k in ng_idx:
            image_id = test_ids[k]
            per_image.append({
                "variant": variant, "image_id": image_id, "sample_id": groups[k],
                "defect_type": ng_registry[image_id].get("defect_type", ""),
                "mask_fraction_roi": float(masks[k].mean()),
                "pixel_auroc": pixel_auroc(masks, maps[variant], np.asarray([k])),
                "score": float(scores[variant][k]), "threshold": thresholds[variant],
                "detected": None if thresholds[variant] is None else bool(scores[variant][k] >= thresholds[variant]),
            })
    for variant in SOURCES:
        rows = [r for r in per_image if r["variant"] == variant]
        for defect in sorted({r["defect_type"] for r in rows}):
            subset = [r for r in rows if r["defect_type"] == defect]
            per_type.append({"variant": variant, "defect_type": defect, "ng_images": len(subset),
                             "samples": len({r["sample_id"] for r in subset}),
                             "detected": None if subset[0]["detected"] is None else sum(r["detected"] for r in subset),
                             "median_pixel_auroc": float(np.median([r["pixel_auroc"] for r in subset]))})
    write_rows(args.run_dir / "localization.csv", localization)
    write_rows(args.run_dir / "localization_per_image.csv", per_image)
    write_rows(args.run_dir / "defect_type_breakdown.csv", per_type)

    # Figure 4 candidate: kept local (contains factory images); not viewed by the assistant.
    from PIL import Image, ImageDraw

    tiles = []
    for k in ng_idx:
        view = load_view(Path(metadata[test_ids[k]]["image_path"]), "roi", roi)
        mask_img = Image.fromarray(masks[k] * 255).resize((420, round(view.height * 420 / view.width)), Image.Resampling.NEAREST)
        row = [view.resize(mask_img.size, Image.Resampling.BILINEAR), mask_img.convert("RGB")]
        for variant in ("B", "E", "F"):
            scale = float(np.quantile(maps[variant][ng_idx], 0.999))
            row.append(overlay_image(view, maps[variant][k], scale))
        tiles.append(row)
    w, h = tiles[0][0].size
    sheet = Image.new("RGB", (w * 5, (h + 24) * len(tiles) + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for c, title in enumerate(("Ảnh (ROI)", "Mask", "B PatchCore", "E PaDiM", "F EfficientAD")):
        draw.text((c * w + 6, 4), title, fill="black")
    for r, row in enumerate(tiles):
        top = 24 + r * (h + 24)
        draw.text((6, top + 2), test_ids[ng_idx[r]], fill="black")
        for c, tile in enumerate(row):
            sheet.paste(tile.resize((w, h)), (c * w, top + 22))
    sheet.save(args.run_dir / "figure4_candidates_LOCAL_ONLY.jpg", quality=88)

    meta = {
        "protocol": "D-110", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256_file(args.config),
        "reference_run": str(args.reference_run), "efficientad_run": str(args.efficientad_run),
        "raw_sha256": {v: sha256_file(runs[s[0]] / s[1]) for v, s in SOURCES.items()},
        "thresholds": {v: thresholds[v] for v in SOURCES}, "mask_files": mask_files,
        "bootstrap": ci, "test_images": len(test_ids), "ng_images": int(ng_idx.size),
    }
    (args.run_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"localization": localization, "defect_types": per_type}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
