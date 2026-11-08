"""RQ4 evaluation without new captures (protocol D-116): per-session metrics and a digital stress test.

Models are fitted on the original `model` images only and thresholds stay frozen
from the reference runs; the perturbed test images are evaluated, never used to
select anything. This simulates acquisition changes; it is not a physical capture.
"""
from __future__ import annotations

import argparse
import json
import sys
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
from run_internal_efficientad import score as score_efficientad
from run_internal_efficientad import to_unit_tensor
from run_internal_experiment import (
    cluster_bootstrap,
    fit_and_score,
    image_metrics,
    load_view,
    read_csv,
    sha256_file,
    to_tensor,
    write_rows,
)
from run_internal_localization import auroc_from_counts, pixel_auroc, registry_ng

THRESHOLDED = ("C", "D", "E", "F")
PIXEL = ("C", "D", "E", "F")


# --------------------------------------------------------------------------- perturbations

def geometry(spec: dict[str, Any], width: int, height: int) -> np.ndarray | None:
    """Affine matrix for a condition, or None when it has no geometric part."""
    import cv2

    dx, dy, degrees = spec.get("dx", 0), spec.get("dy", 0), spec.get("rotate_deg", 0)
    if not (dx or dy or degrees):
        return None
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    matrix[:, 2] += (dx, dy)
    return matrix


def perturb_image(rgb: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    import cv2

    height, width = rgb.shape[:2]
    out = rgb
    matrix = geometry(spec, width, height)
    if matrix is not None:
        out = cv2.warpAffine(out, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if "gain" in spec:
        out = np.clip(np.rint(out.astype(np.float32) * spec["gain"]), 0, 255).astype(np.uint8)
    if "gain_x" in spec:
        ramp = np.linspace(*spec["gain_x"], width, dtype=np.float32)[None, :, None]
        out = np.clip(np.rint(out.astype(np.float32) * ramp), 0, 255).astype(np.uint8)
    return out


def perturb_mask(mask: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Masks follow the geometry only; lighting does not move a defect."""
    import cv2

    height, width = mask.shape[:2]
    matrix = geometry(spec, width, height)
    if matrix is None:
        return mask
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_view(image, mode: str, roi: list[int]):
    if mode == "full_frame":
        return image
    x, y, w, h = roi
    return image.crop((x, y, min(image.width, x + w), min(image.height, y + h)))


def mask_for_condition(path: Path, spec: dict[str, Any], roi: list[int], size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    view = crop_view(Image.fromarray(perturb_mask(mask, spec)), "roi", roi)
    return (np.asarray(view.resize((size[1], size[0]), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


# --------------------------------------------------------------------------- statistics

def pixel_counts(masks: np.ndarray, maps: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = edges.size + 1
    binned = np.searchsorted(edges, maps.reshape(len(maps), -1))
    flat = masks.reshape(len(masks), -1).astype(bool)
    pos = np.stack([np.bincount(b[f], minlength=bins) for b, f in zip(binned, flat)])
    neg = np.stack([np.bincount(b[~f], minlength=bins) for b, f in zip(binned, flat)])
    return pos, neg


def relative_difference(new: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(new) - np.asarray(reference))) / max(np.max(np.abs(reference)), 1e-12))


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_robustness_v1.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "metadata" / "INTERNAL_IMAGE_REGISTRY.xlsx")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-only", action="store_true", help="Stop after the nominal reproduction gate.")
    args = parser.parse_args()

    import torch
    from PIL import Image

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / config["base_config"]).read_text(encoding="utf-8"))
    ead = yaml.safe_load((ROOT / config["efficientad_config"]).read_text(encoding="utf-8"))["efficientad"]
    reference, ead_run = ROOT / config["reference_run"], ROOT / config["efficientad_run"]
    conditions: dict[str, dict[str, Any]] = config["conditions"]
    if args.gate_only:
        conditions = {"nominal": conditions["nominal"]}
    ci, seed = base["metrics"]["confidence_interval"], config["seed"]
    size, roi = tuple(base["input_size"]), base["variants"]["B"]["roi"]

    validation_path = args.split_dir / "split_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    metadata_path, assignments_path = args.split_dir / "metadata.csv", args.split_dir / "assignments.csv"
    if validation.get("integrity_status") != "PASS" or not validation.get("experiment_ready"):
        raise SystemExit("Split is not PASS/experiment_ready.")
    if validation["metadata_sha256"] != sha256_file(metadata_path) or validation["assignments_sha256"] != sha256_file(assignments_path):
        raise SystemExit("Split CSVs changed after validation.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {row["image_id"]: row for row in read_csv(metadata_path)}
    split_of = {row["image_id"]: row["split"] for row in read_csv(assignments_path)}
    by_split = {name: sorted(i for i, s in split_of.items() if s == name) for name in ("model", "calibration", "test")}
    test_ids, cal_ids = by_split["test"], by_split["calibration"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta: dict[str, Any] = {
        "status": "running", "protocol": config.get("protocol", "D-116"), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256_file(args.config),
        "split_validation_sha256": sha256_file(validation_path), "conditions": conditions, "bootstrap": ci,
        "device": str(device), "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    meta_path = args.run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 1. Build the perturbed views of the test images (labels untouched).
    views: dict[str, dict[str, list]] = {c: {"full_frame": [], "roi": []} for c in conditions}
    for image_id in test_ids:
        frame = np.asarray(load_view(Path(metadata[image_id]["image_path"]), "full_frame", None))
        for name, spec in conditions.items():
            image = Image.fromarray(perturb_image(frame, spec))
            for mode in ("full_frame", "roi"):
                views[name][mode].append(crop_view(image, mode, roi).resize((size[1], size[0]), Image.Resampling.BILINEAR))
    print("views ready", flush=True)

    # 2. Score every condition; calibration scores only verify the frozen thresholds.
    source = {mode: {i: load_view(Path(metadata[i]["image_path"]), mode, roi if mode == "roi" else None)
                     for i in by_split["model"] + cal_ids} for mode in ("full_frame", "roi")}
    runs: dict[str, dict[str, Any]] = {}
    for key, family, mode in (("patchcore_full_frame", "patchcore", "full_frame"), ("patchcore_roi", "patchcore", "roi"),
                              ("padim_roi", "padim", "roi")):
        score_sets = {"calibration": [to_tensor(source[mode][i], size) for i in cal_ids]}
        for name in conditions:
            score_sets[name] = [to_tensor(view, size) for view in views[name][mode]]
        runs[key] = fit_and_score(family, base, seed, [to_tensor(source[mode][i], size) for i in by_split["model"]],
                                  score_sets, device)
        print(f"done {key}", flush=True)

    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize

    model = EfficientAdModel(teacher_out_channels=ead["teacher_out_channels"], model_size=EfficientAdModelSize(ead["model_size"]),
                             padding=ead["padding"], pad_maps=ead["pad_maps"]).to(device)
    model.load_state_dict(torch.load(ead_run / "efficientad_final.pt", map_location=device, weights_only=True))
    model.eval()
    efficientad: dict[str, Any] = {}
    cal_scores, _, _ = score_efficientad(model, [to_unit_tensor(source["roi"][i], size) for i in cal_ids], device)
    efficientad["calibration"] = {"scores": cal_scores}
    for name in conditions:
        scores, maps, _ = score_efficientad(model, [to_unit_tensor(view, size) for view in views[name]["roi"]], device)
        efficientad[name] = {"scores": scores, "maps": maps}
    runs["efficientad_roi"] = efficientad
    del model, source, views
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("done efficientad", flush=True)

    # 3. Assemble variants on the frozen thresholds and check they reproduce the reference runs.
    frozen = json.loads((reference / "frozen_thresholds.json").read_text(encoding="utf-8"))[str(seed)]
    frozen.update(json.loads((ead_run / "frozen_thresholds.json").read_text(encoding="utf-8"))[str(seed)])

    def variant(name: str, condition: str) -> tuple[np.ndarray, np.ndarray | None]:
        if name == "A":
            return runs["patchcore_full_frame"][condition]["scores"], None
        if name in ("B", "C"):
            item = runs["patchcore_roi"][condition]
            return item["scores"], item["maps"]
        if name == "D":
            maps = np.stack([smooth_map_3x3(m) for m in runs["patchcore_roi"][condition]["maps"]])
            return maps.reshape(len(maps), -1).max(axis=1), maps
        item = runs["padim_roi" if name == "E" else "efficientad_roi"][condition]
        return item["scores"], item["maps"]

    reference_raw = {"patchcore_full_frame": reference / "raw_patchcore_full_frame_seed42.npz",
                     "patchcore_roi": reference / "raw_patchcore_roi_seed42.npz",
                     "padim_roi": reference / "raw_padim_roi_seed42.npz",
                     "efficientad_roi": ead_run / "raw_efficientad_roi_seed42.npz"}
    gate: dict[str, Any] = {}
    for key, path in reference_raw.items():
        raw = np.load(path)
        if [str(i) for i in raw["test_ids"]] != test_ids or [str(i) for i in raw["calibration_ids"]] != cal_ids:
            raise SystemExit(f"Image order differs from {path.name}")
        gate[key] = {"test_scores": relative_difference(runs[key]["nominal"]["scores"], raw["test_scores"]),
                     "calibration_scores": relative_difference(runs[key]["calibration"]["scores"], raw["calibration_scores"])}
    derived = {"C": runs["patchcore_roi"]["calibration"]["scores"], "E": runs["padim_roi"]["calibration"]["scores"],
               "F": runs["efficientad_roi"]["calibration"]["scores"]}
    gate["thresholds"] = {k: relative_difference(np.asarray([upper_empirical_quantile(v, 0.95)]), np.asarray([frozen[k]]))
                          for k, v in derived.items()}
    limit = config["reproduction_gate"]["max_relative_difference"]
    worst = max([v for g in gate.values() for v in g.values()])
    meta["reproduction_gate"] = {"limit": limit, "worst": worst, "detail": gate, "passed": worst <= limit}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta["reproduction_gate"], indent=1), flush=True)
    if worst > limit:
        meta["status"] = "stopped_reproduction_gate"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("Reproduction gate failed; no metrics computed.")
    if args.gate_only:
        meta["status"] = "gate_only"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    for key, item in runs.items():
        np.savez_compressed(args.run_dir / f"raw_{key}.npz", test_ids=np.asarray(test_ids),
                            **{f"scores__{c}": item[c]["scores"] for c in conditions},
                            **{f"maps__{c}": item[c]["maps"].astype(np.float16) for c in conditions})

    # 4. Only now read labels, groups and masks.
    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids], dtype=np.uint8)
    groups = np.asarray([metadata[i]["sample_id"] for i in test_ids])
    sessions = np.asarray([metadata[i]["capture_session"] for i in test_ids])
    ng_registry = registry_ng(args.registry)
    masks: dict[str, np.ndarray] = {}
    for name, spec in conditions.items():
        stack = np.zeros((len(test_ids), *size), np.uint8)
        for k in np.flatnonzero(labels == 1):
            stack[k] = mask_for_condition(args.data_root / ng_registry[test_ids[k]]["annotation_path"].strip(), spec, roi, size)
        masks[name] = stack

    table5 = {row["variant"]: row for row in read_csv(reference / "table5_main_results.csv")}
    table5.update({row["variant"]: row for row in read_csv(ead_run / "table5_efficientad.csv")})
    for name in ("A", "C", "D", "E", "F"):
        scores, _ = variant(name, "nominal")
        metrics = image_metrics(labels, scores, frozen.get(name))
        if abs(metrics["image_auroc"] - float(table5[name]["image_auroc"])) > 1e-9:
            raise SystemExit(f"Nominal image-AUROC of {name} differs from Table 5.")
        if name in THRESHOLDED and any(metrics[k] != int(table5[name][k]) for k in ("tp", "fp", "fn", "tn")):
            raise SystemExit(f"Nominal confusion counts of {name} differ from Table 5.")

    # 5. Part 1: per-session metrics on the frozen thresholds (nominal images).
    session_rows = []
    for name in ("A", "C", "D", "E", "F"):
        scores, _ = variant(name, "nominal")
        for session in sorted(set(sessions)):
            idx = np.flatnonzero(sessions == session)
            threshold = frozen.get(name)
            metrics = image_metrics(labels[idx], scores[idx], threshold)
            boot = cluster_bootstrap(groups[idx], lambda i, idx=idx: image_metrics(labels[idx][i], scores[idx][i], threshold),
                                     ci["resamples"], ci["level"], ci["seed"])
            row = {"variant": name, "session": session, "ok_images": int((labels[idx] == 0).sum()), "ng_images": int(labels[idx].sum()),
                   "samples": int(np.unique(groups[idx]).size), **metrics, "ci_valid_resamples": boot["valid_resamples"]}
            for key in ("image_auroc", "recall", "fpr"):
                if key in boot["intervals"]:
                    row[f"{key}_ci_low"], row[f"{key}_ci_high"] = boot["intervals"][key]
            session_rows.append(row)
    write_rows(args.run_dir / "table7_session.csv", session_rows)

    calibration_sessions = np.asarray([metadata[i]["capture_session"] for i in cal_ids])
    drift_rows = []
    for name, key in (("C", "patchcore_roi"), ("E", "padim_roi"), ("F", "efficientad_roi")):
        for session in sorted(set(calibration_sessions)):
            values = runs[key]["calibration"]["scores"][calibration_sessions == session]
            drift_rows.append({"variant": name, "role": "calibration", "session": session, "ok_images": int(values.size),
                               "median": float(np.median(values)), "threshold": frozen[name]})
        for session in sorted(set(sessions)):
            values = runs[key]["nominal"]["scores"][(sessions == session) & (labels == 0)]
            drift_rows.append({"variant": name, "role": "test", "session": session, "ok_images": int(values.size),
                               "median": float(np.median(values)), "threshold": frozen[name]})
    write_rows(args.run_dir / "ok_score_by_session.csv", drift_rows)
    print("session tables done", flush=True)

    # 6. Part 2: stress test, each condition against nominal on the same resamples.
    deltas = ("image_auroc", "image_auprc", "recall", "fpr")
    rows, bins = [], config["pixel_bins"]
    all_idx = np.arange(len(test_ids))
    for name in ("A", "C", "D", "E", "F"):
        threshold = frozen.get(name)
        nominal_scores, nominal_maps = variant(name, "nominal")
        for condition in conditions:
            scores, maps = variant(name, condition)
            metrics = image_metrics(labels, scores, threshold)
            row: dict[str, Any] = {"condition": condition, "variant": name, "threshold": threshold, **metrics}
            base_metrics = image_metrics(labels, nominal_scores, threshold)
            for key in deltas:
                if key in metrics:
                    row[f"delta_{key}"] = metrics[key] - base_metrics[key]
            if condition != "nominal":
                def diff(indices, scores=scores):
                    a = image_metrics(labels[indices], scores[indices], threshold)
                    b = image_metrics(labels[indices], nominal_scores[indices], threshold)
                    return {k: a[k] - b[k] for k in deltas if k in a}

                boot = cluster_bootstrap(groups, diff, ci["resamples"], ci["level"], ci["seed"])
                for key, (low, high) in boot["intervals"].items():
                    row[f"delta_{key}_ci_low"], row[f"delta_{key}_ci_high"] = low, high
            if name in PIXEL:
                ng_idx = np.flatnonzero(labels == 1)
                row["pixel_auroc"] = pixel_auroc(masks[condition], maps, all_idx)
                row["pixel_auroc_ng_images_only"] = pixel_auroc(masks[condition], maps, ng_idx)
                row["delta_pixel_auroc"] = row["pixel_auroc"] - pixel_auroc(masks["nominal"], nominal_maps, all_idx)
                if condition != "nominal":
                    edges = np.quantile(np.concatenate([maps.ravel(), nominal_maps.ravel()]), np.linspace(0, 1, bins + 1)[1:-1])
                    pos_c, neg_c = pixel_counts(masks[condition], maps, edges)
                    pos_n, neg_n = pixel_counts(masks["nominal"], nominal_maps, edges)
                    pboot = cluster_bootstrap(groups, lambda i: {"d": auroc_from_counts(pos_c[i].sum(0), neg_c[i].sum(0))
                                                                 - auroc_from_counts(pos_n[i].sum(0), neg_n[i].sum(0))},
                                              ci["resamples"], ci["level"], ci["seed"])
                    row["delta_pixel_auroc_ci_low"], row["delta_pixel_auroc_ci_high"] = pboot["intervals"]["d"]
            rows.append(row)
            print(f"{name} {condition}", flush=True)
    write_rows(args.run_dir / "table7_stress.csv", rows)

    # 7. Does the ROI change the drop? (C_cond - C_nominal) - (A_cond - A_nominal) on image-AUROC.
    from sklearn.metrics import roc_auc_score

    roi_rows = []
    c_nom, a_nom = variant("C", "nominal")[0], variant("A", "nominal")[0]
    for condition in conditions:
        if condition == "nominal":
            continue
        c_cond, a_cond = variant("C", condition)[0], variant("A", condition)[0]

        def did(indices, c_cond=c_cond, a_cond=a_cond):
            y = labels[indices]
            if np.unique(y).size < 2:
                raise ValueError("one class")
            return {"did": (roc_auc_score(y, c_cond[indices]) - roc_auc_score(y, c_nom[indices]))
                    - (roc_auc_score(y, a_cond[indices]) - roc_auc_score(y, a_nom[indices]))}

        boot = cluster_bootstrap(groups, did, ci["resamples"], ci["level"], ci["seed"])
        low, high = boot["intervals"]["did"]
        roi_rows.append({"condition": condition, "roi_minus_full_frame_change": did(all_idx)["did"], "ci_low": low, "ci_high": high})
    write_rows(args.run_dir / "roi_vs_full_frame_change.csv", roi_rows)

    meta.update({"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                 "test_ok": int((labels == 0).sum()), "test_ng": int(labels.sum()),
                 "mask_sha256": {test_ids[k]: sha256_file(args.data_root / ng_registry[test_ids[k]]["annotation_path"].strip())
                                 for k in np.flatnonzero(labels == 1)}})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
