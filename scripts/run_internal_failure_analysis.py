"""Descriptive failure analysis of the locked internal test set and Figures 4-5 (protocol D-134).

Two NG metrics were added after a first run showed that no D peak lay inside a drawn mask:
the distance from the map peak to the mask and the highest map value inside the mask
relative to the threshold. They describe cases only.

Everything comes from the frozen v2 raw outputs and thresholds; nothing is fitted, tuned
or re-thresholded. The decision rule stays `score > threshold`. Image statistics are
computed on the 256 x 256 ROI view only to describe cases (brightness, near-saturated
pixels); they are never used to select a configuration.

`--layout-only` replaces every photograph, mask and map with synthetic placeholders so the
figure layout can be checked without displaying any captured image.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iad_w1.normal_calibration import smooth_map_3x3  # noqa: E402
from run_internal_experiment import load_mask, load_view, read_csv, sha256_file, write_rows  # noqa: E402

# variant -> (run key, raw file, smooth maps)
SOURCES = {
    "D": ("reference_run", "raw_patchcore_roi_seed42.npz", True),
    "E": ("reference_run", "raw_padim_roi_seed42.npz", False),
    "F": ("efficientad_run", "raw_efficientad_roi_seed42.npz", False),
}
BORDER_PX = 8
SATURATED = 250
DEFECT_NAMES = {"xuoc": "vết xước", "vet_ban_dom": "vết bẩn/đốm", "nhieu_loai": "nhiều loại lỗi", "viet_ve": "vết viết vẽ"}


def registry_rows(path: Path) -> dict[str, dict[str, str]]:
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for sheet in ("Anh_OK", "Anh_NG"):
        rows = workbook[sheet].iter_rows(values_only=True)
        header = next(rows)
        for row in rows:
            record = {str(k): ("" if v is None else str(v)) for k, v in zip(header, row) if k}
            if record.get("image_id"):
                out[record["image_id"]] = record
    return out


def outcome(label: int, score: float, threshold: float) -> str:
    positive = score > threshold
    return {(1, True): "TP", (1, False): "FN", (0, True): "FP", (0, False): "TN"}[(label, positive)]


def peak(score_map: np.ndarray) -> tuple[int, int, int]:
    y, x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    border = min(y, x, score_map.shape[0] - 1 - y, score_map.shape[1] - 1 - x)
    return int(y), int(x), int(border)


def image_stats(view) -> dict[str, float]:
    from PIL import Image

    small = np.asarray(view.resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32)
    grey = small.mean(axis=2)
    return {"brightness_mean": float(grey.mean()), "brightness_std": float(grey.std()),
            "near_saturated_fraction": float((small.max(axis=2) >= SATURATED).mean())}


def median_index(order: list[int], key) -> int:
    ranked = sorted(order, key=key)
    return ranked[(len(ranked) - 1) // 2]


def select_figure4(rows: list[dict]) -> list[str]:
    ng = [r for r in rows if r["label"] == "NG"]
    picks = []
    for defect in sorted({r["defect_type"] for r in ng}):
        group = [i for i, r in enumerate(ng) if r["defect_type"] == defect]
        picks.append(ng[median_index(group, lambda i: ng[i]["D_score"])]["image_id"])
    ok = [r for r in rows if r["label"] == "OK"]
    picks.append(ok[median_index(list(range(len(ok))), lambda i: ok[i]["D_score"])]["image_id"])
    return picks


def select_figure5(rows: list[dict]) -> list[tuple[str, str]]:
    ng = [r for r in rows if r["label"] == "NG"]
    missed = sorted([r for r in ng if r["D_outcome"] == "FN"], key=lambda r: r["D_margin"])
    cases = [(r["image_id"], "FN của D, biên thấp nhất") for r in missed[:2]]
    cases += [(r["image_id"], "FN của D, gần ngưỡng nhất") for r in missed[::-1][:2] if r["image_id"] not in {c[0] for c in cases}]
    cases += [(r["image_id"], "FP của E") for r in rows if r["E_outcome"] == "FP"]
    ok = sorted([r for r in rows if r["label"] == "OK"], key=lambda r: -r["D_score"])
    cases += [(r["image_id"], "OK có điểm D cao nhất") for r in ok[:2] if r["image_id"] not in {c[0] for c in cases}]
    return cases[:7]


def fmt(value: float) -> str:
    return f"{value:.1f}".replace(".", ",") if value >= 10 else f"{value:.3f}".replace(".", ",")


def draw_figures(rows_by_id, views, masks, maps, thresholds, ok_medians, fig4, fig5, out_dir: Path, suffix: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "font.size": 7})

    def tile(ax, image_id, column):
        view = views[image_id]
        height, width = view.shape[:2]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
        if column == "image":
            ax.imshow(view)
            return
        if column == "mask":
            ax.imshow(np.zeros((height, width)), cmap="gray", vmin=0, vmax=1)
            if image_id in masks:
                ax.imshow(masks[image_id], cmap="gray", vmin=0, vmax=1, extent=(0, width, height, 0))
            else:
                ax.text(width / 2, height / 2, "ảnh OK,\nkhông có lỗi", color="white", ha="center", va="center", fontsize=6)
            return
        score_map = maps[column][image_id]
        ax.imshow(view)
        ax.imshow(score_map, cmap="inferno", alpha=0.55, vmin=ok_medians[column], vmax=thresholds[column],
                  extent=(0, width, height, 0))
        if score_map.max() > thresholds[column]:
            ys, xs = np.linspace(0, height, score_map.shape[0]), np.linspace(0, width, score_map.shape[1])
            ax.contour(xs, ys, score_map, levels=[thresholds[column]], colors="#00e5ff", linewidths=0.8)

    paths = []
    names = {"image": "Ảnh (ROI)", "mask": "Mặt nạ lỗi", "D": "D: PatchCore", "E": "E: PaDiM", "F": "F: EfficientAD-S"}

    def grid(selection, columns, labels, path):
        rows_n, cols_n = len(selection), len(columns)
        aspect = views[selection[0]].shape[0] / views[selection[0]].shape[1]
        tile_w = (6.25 - 1.5) / cols_n  # fits a 6.3 in text width at 100 %
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(1.5 + tile_w * cols_n, 0.35 + rows_n * tile_w * aspect * 1.08),
                                 squeeze=False)
        for r, image_id in enumerate(selection):
            for c, column in enumerate(columns):
                tile(axes[r][c], image_id, column)
                if r == 0:
                    axes[r][c].set_title(names[column], fontsize=7, fontweight="bold")
            axes[r][0].set_ylabel(labels[r], rotation=0, ha="right", va="center", fontsize=6.2, labelpad=3)
        fig.subplots_adjust(left=1.45 / (1.5 + tile_w * cols_n), right=0.995, top=0.94, bottom=0.01, wspace=0.04, hspace=0.06)
        fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        paths.append(path)

    def label(image_id, head):
        row = rows_by_id[image_id]
        kind = "NG: " + DEFECT_NAMES.get(row["defect_type"], row["defect_type"]) if row["label"] == "NG" else "OK"
        return (f"{head}\n{row['session']} · {row['sample_id']}\n{kind}\n"
                f"D {fmt(row['D_score'])} / {fmt(thresholds['D'])}\nE {fmt(row['E_score'])} / {fmt(thresholds['E'])}")

    grid(fig4, ["image", "mask", "D", "E", "F"],
         [label(i, "Trung vị nhóm lỗi" if rows_by_id[i]["label"] == "NG" else "Trung vị ảnh OK")
          for i in fig4], out_dir / f"hinh4_dinh_vi{suffix}.png")
    grid([c[0] for c in fig5], ["image", "mask", "D", "E"], [label(i, head) for i, head in fig5],
         out_dir / f"hinh5_ca_loi{suffix}.png")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v2.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v2")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "metadata" / "INTERNAL_IMAGE_REGISTRY.xlsx")
    parser.add_argument("--reference-run", type=Path, default=ROOT / "runs" / "internal_main_v2" / "20261107-run01")
    parser.add_argument("--efficientad-run", type=Path, default=ROOT / "runs" / "internal_main_v2" / "20261107-efficientad01")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()

    if not args.layout_only and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    roi, size = config["variants"]["B"]["roi"], tuple(config["input_size"])
    metadata = {row["image_id"]: row for row in read_csv(args.split_dir / "metadata.csv")}
    registry = registry_rows(args.registry)
    runs = {"reference_run": args.reference_run, "efficientad_run": args.efficientad_run}
    thresholds = {}
    for run in runs.values():
        thresholds.update(json.loads((run / "frozen_thresholds.json").read_text(encoding="utf-8"))["42"])
    thresholds = {v: float(thresholds[v]) for v in SOURCES}

    maps, scores, calib, test_ids, calib_ids = {}, {}, {}, None, None
    score_equals_map_max = {}
    for variant, (run_key, name, smooth) in SOURCES.items():
        raw = np.load(runs[run_key] / name)
        ids = [str(i) for i in raw["test_ids"]]
        if test_ids is None:
            test_ids, calib_ids = ids, [str(i) for i in raw["calibration_ids"]]
        elif ids != test_ids or [str(i) for i in raw["calibration_ids"]] != calib_ids:
            raise ValueError(f"Image order differs for {variant}")
        test_maps, calib_maps = raw["test_maps"], raw["calibration_maps"]
        if smooth:
            test_maps = np.stack([smooth_map_3x3(m) for m in test_maps])
            calib_maps = np.stack([smooth_map_3x3(m) for m in calib_maps])
            scores[variant] = test_maps.reshape(len(test_maps), -1).max(axis=1)
            calib[variant] = calib_maps.reshape(len(calib_maps), -1).max(axis=1)
        else:
            scores[variant], calib[variant] = raw["test_scores"].astype(float), raw["calibration_scores"].astype(float)
        maps[variant] = test_maps
        score_equals_map_max[variant] = bool(np.allclose(scores[variant], test_maps.reshape(len(test_maps), -1).max(axis=1),
                                                         rtol=1e-5, atol=1e-6))

    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids])
    ok_idx = np.flatnonzero(labels == 0)
    ok_medians = {v: float(np.median(maps[v][ok_idx])) for v in SOURCES}
    rng = np.random.default_rng(0)
    views, masks, rows = {}, {}, []
    for k, image_id in enumerate(test_ids):
        meta, reg = metadata[image_id], registry[image_id]
        if args.layout_only:
            view = np.full((round(256 * roi[3] / roi[2]), 256, 3), 150, np.uint8)
            stats = {}
        else:
            pil_view = load_view(Path(meta["image_path"]), "roi", roi)
            stats = image_stats(pil_view)
            view = np.asarray(pil_view.resize((600, round(600 * pil_view.height / pil_view.width))))
        views[image_id] = view
        row = {"image_id": image_id, "label": meta["label"], "sample_id": meta["sample_id"],
               "session": meta["capture_session"].split("-")[-1], "defect_type": reg.get("defect_type", ""),
               "geometry_class": reg.get("geometry_class", ""), "color_finish": reg.get("color_finish", ""), **stats}
        mask = None
        if labels[k]:
            if args.layout_only:
                yy, xx = np.mgrid[:size[0], :size[1]]
                mask = (((yy - 128) / 30) ** 2 + ((xx - 140) / 18) ** 2 < 1).astype(np.uint8)
            else:
                mask = load_mask(args.data_root / reg["annotation_path"], "roi", roi, size)
            masks[image_id] = mask
            row["mask_fraction_roi"] = float(mask.mean())
        for v in SOURCES:
            if args.layout_only:
                maps[v][k] = rng.normal(0, 1, size).cumsum(0).cumsum(1) / 300 + ok_medians[v]
            score_map, score = maps[v][k], float(scores[v][k])
            y, x, border = peak(score_map)
            row.update({f"{v}_score": score, f"{v}_margin": score - thresholds[v],
                        f"{v}_outcome": outcome(int(labels[k]), score, thresholds[v]),
                        f"{v}_peak_y": y, f"{v}_peak_x": x, f"{v}_peak_near_border": border < BORDER_PX})
            if mask is not None:
                above = score_map > thresholds[v]
                row[f"{v}_peak_in_mask"] = bool(mask[y, x])
                row[f"{v}_peak_distance_to_mask_px"] = float(distance_transform_edt(mask == 0)[y, x])
                row[f"{v}_mask_max_over_threshold"] = float(score_map[mask > 0].max() / thresholds[v])
                row[f"{v}_above_in_mask_fraction"] = float((above & (mask > 0)).sum() / above.sum()) if above.any() else None
                row[f"{v}_ok_scoring_higher"] = int((scores[v][ok_idx] > score).sum())
        rows.append(row)

    rows_by_id = {r["image_id"]: r for r in rows}
    fig4, fig5 = select_figure4(rows), select_figure5(rows)
    map_dict = {v: {i: maps[v][k] for k, i in enumerate(test_ids)} for v in SOURCES}
    suffix = "_LAYOUT_ONLY" if args.layout_only else ""
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figures = draw_figures(rows_by_id, views, masks, map_dict, thresholds, ok_medians, fig4, fig5, args.figure_dir, suffix)
    if args.layout_only:
        print("layout figures:", [str(p) for p in figures])
        return

    calibration = []
    for sample in sorted({metadata[i]["sample_id"] for i in calib_ids}):
        idx = [k for k, i in enumerate(calib_ids) if metadata[i]["sample_id"] == sample]
        calibration.append({"sample_id": sample, "images": len(idx),
                            **{f"{v}_{s}": float(f(calib[v][idx])) for v in ("D", "E") for s, f in (("median", np.median), ("max", np.max))},
                            **{f"{v}_above_threshold": int((calib[v][idx] > thresholds[v]).sum()) for v in ("D", "E")}})
    calib_rows = [{"image_id": i, "sample_id": metadata[i]["sample_id"],
                   "geometry_class": registry[i].get("geometry_class", ""), "D_score": float(calib["D"][k]),
                   "E_score": float(calib["E"][k])} for k, i in enumerate(calib_ids)]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.run_dir / "failure_per_image.csv", rows)
    write_rows(args.run_dir / "calibration_by_sample.csv", calibration)
    write_rows(args.run_dir / "calibration_per_image.csv", calib_rows)
    write_rows(args.run_dir / "figure4_selection.csv", [{"image_id": i} for i in fig4])
    write_rows(args.run_dir / "figure5_selection.csv", [{"image_id": i, "reason": r} for i, r in fig5])

    def summary_for(v):
        ng = [r for r in rows if r["label"] == "NG"]
        ok = [r for r in rows if r["label"] == "OK"]
        fn = [r for r in ng if r[f"{v}_outcome"] == "FN"]
        return {
            "outcomes": {o: sum(r[f"{v}_outcome"] == o for r in rows) for o in ("TP", "FP", "FN", "TN")},
            "fn_by_defect_type": {d: sum(r["defect_type"] == d for r in fn) for d in sorted({r["defect_type"] for r in ng})},
            "ng_by_defect_type": {d: sum(r["defect_type"] == d for r in ng) for d in sorted({r["defect_type"] for r in ng})},
            "fn_peak_in_mask": sum(r[f"{v}_peak_in_mask"] for r in fn), "fn": len(fn),
            "ng_peak_in_mask": sum(r[f"{v}_peak_in_mask"] for r in ng),
            "ng_peak_distance_to_mask_px_median": float(np.median([r[f"{v}_peak_distance_to_mask_px"] for r in ng])),
            "ng_peak_within_8px_of_mask": sum(r[f"{v}_peak_distance_to_mask_px"] <= BORDER_PX for r in ng),
            "fn_mask_max_below_threshold": sum(r[f"{v}_mask_max_over_threshold"] <= 1 for r in fn),
            "fn_mask_max_over_threshold_range": [min(r[f"{v}_mask_max_over_threshold"] for r in fn),
                                                 max(r[f"{v}_mask_max_over_threshold"] for r in fn)] if fn else None,
            "ng_ok_scoring_higher_median": float(np.median([r[f"{v}_ok_scoring_higher"] for r in ng])),
            "ng_above_all_ok": sum(r[f"{v}_ok_scoring_higher"] == 0 for r in ng),
            "ok_score_max": max(r[f"{v}_score"] for r in ok), "ok_score_median": float(np.median([r[f"{v}_score"] for r in ok])),
            "ng_score_median": float(np.median([r[f"{v}_score"] for r in ng])),
            "calibration_score_median": float(np.median(calib[v])),
            "ok_peak_near_border": sum(r[f"{v}_peak_near_border"] for r in ok),
            "ng_peak_near_border": sum(r[f"{v}_peak_near_border"] for r in ng),
        }

    summary = {"thresholds": thresholds, "score_equals_map_max": score_equals_map_max,
               "variants": {v: summary_for(v) for v in SOURCES}, "figure4": fig4, "figure5": fig5}
    (args.run_dir / "failure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {"protocol": "D-134", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv], "config_sha256": sha256_file(args.config),
            "split_metadata_sha256": sha256_file(args.split_dir / "metadata.csv"),
            "raw_sha256": {v: sha256_file(runs[s[0]] / s[1]) for v, s in SOURCES.items()},
            "figures": {p.name: sha256_file(p) for p in figures}}
    (args.run_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
