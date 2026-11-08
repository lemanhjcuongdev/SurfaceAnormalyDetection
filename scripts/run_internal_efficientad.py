"""Run EfficientAD-S on the locked internal v1 split (protocol D-109): variant F.

Uses the same split gate, ROI, threshold rule and cluster bootstrap as
`run_internal_experiment.py`. Training uses only `model` images (plus the
ImageNette penalty of the original method), map-normalisation quantiles and the
image threshold use only `calibration` images, and test labels/masks are read
only after every score and the threshold are fixed.
"""

from __future__ import annotations

import argparse
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

from iad_w1.normal_calibration import upper_empirical_quantile
from run_internal_experiment import (
    cluster_bootstrap,
    image_metrics,
    load_mask,
    load_view,
    read_csv,
    sha256_file,
    write_rows,
)


def to_unit_tensor(image, size: tuple[int, int]):
    """Resize to the model input and scale to [0, 1]; EfficientAD must not get ImageNet normalisation."""
    import torch
    from PIL import Image

    resized = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1)


def imagenette_loader(directory: Path, size: tuple[int, int], seed: int):
    """Penalty images exactly as anomalib's EfficientAd.prepare_imagenette_data, with a seeded shuffle."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.transforms.v2 import CenterCrop, Compose, RandomGrayscale, Resize, ToTensor

    transform = Compose([Resize((size[0] * 2, size[1] * 2)), RandomGrayscale(p=0.3), CenterCrop(size), ToTensor()])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ImageFolder(directory, transform=transform), batch_size=1, shuffle=True, generator=generator,
                      num_workers=2, persistent_workers=True, pin_memory=True)


def train_efficientad(params: dict[str, Any], steps: int, train: list, calibration: list, size, device, log_path: Path):
    import torch
    from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize, reduce_tensor_elems

    model = EfficientAdModel(teacher_out_channels=params["teacher_out_channels"],
                             model_size=EfficientAdModelSize(params["model_size"]),
                             padding=params["padding"], pad_maps=params["pad_maps"]).to(device)
    teacher_path = Path(params["teacher_weights_dir"]) / f"pretrained_teacher_{params['model_size']}.pth"
    model.teacher.load_state_dict(torch.load(teacher_path, map_location=device, weights_only=True))

    # Teacher channel statistics over the model split (anomalib teacher_channel_mean_std).
    with torch.no_grad():
        count, total, total_sq = 0, 0.0, 0.0
        for t in train:
            y = model.teacher(t.unsqueeze(0).to(device))
            count += y[:, 0].numel()
            total = total + y.sum(dim=(0, 2, 3))
            total_sq = total_sq + (y ** 2).sum(dim=(0, 2, 3))
        mean = total / count
        std = torch.sqrt(total_sq / count - mean ** 2)
        model.mean_std.update({"mean": mean[None, :, None, None], "std": std[None, :, None, None]})

    optimizer = torch.optim.Adam(list(model.student.parameters()) + list(model.ae.parameters()),
                                 lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(params["lr_decay_at_fraction"] * steps),
                                                gamma=params["lr_decay_gamma"])
    penalty = imagenette_loader(Path(params["imagenette_dir"]), size, params["seed"])
    penalty_iter = iter(penalty)
    order_gen = torch.Generator().manual_seed(params["seed"])
    order: list[int] = []
    log_rows, window = [], []
    model.train()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if not order:
            order = torch.randperm(len(train), generator=order_gen).tolist()
        image = train[order.pop()].unsqueeze(0).to(device, non_blocking=True)
        try:
            imagenet = next(penalty_iter)[0]
        except StopIteration:
            penalty_iter = iter(penalty)
            imagenet = next(penalty_iter)[0]
        loss_st, loss_ae, loss_stae = model(batch=image, batch_imagenet=imagenet.to(device, non_blocking=True))
        loss = loss_st + loss_ae + loss_stae
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        window.append([loss_st.item(), loss_ae.item(), loss_stae.item()])
        if step % 1000 == 0 or step == steps:
            mean_losses = np.mean(window, axis=0)
            log_rows.append({"step": step, "loss_st": mean_losses[0], "loss_ae": mean_losses[1], "loss_stae": mean_losses[2],
                             "lr": scheduler.get_last_lr()[0], "elapsed_s": time.perf_counter() - started})
            window = []
            write_rows(log_path, log_rows)
            print(f"step {step}/{steps} loss {mean_losses.sum():.4f} {time.perf_counter() - started:.0f}s", flush=True)
    train_seconds = time.perf_counter() - started
    model.eval()

    # Map-normalisation quantiles on normal-only calibration images (anomalib map_norm_quantiles).
    with torch.no_grad():
        maps_st, maps_ae = zip(*(model.get_maps(t.unsqueeze(0).to(device), normalize=False) for t in calibration))
        quantiles = {}
        for name, maps in (("st", maps_st), ("ae", maps_ae)):
            flat = reduce_tensor_elems(torch.cat(maps))
            quantiles[f"qa_{name}"] = torch.quantile(flat, q=0.9).to(device)
            quantiles[f"qb_{name}"] = torch.quantile(flat, q=0.995).to(device)
        model.quantiles.update(quantiles)
    return model, train_seconds, {k: float(v) for k, v in quantiles.items()}


def score(model, tensors: list, device) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import torch

    scores, maps, seconds = [], [], []
    with torch.inference_mode():
        for tensor in tensors:
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = model(tensor.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - started)
            scores.append(float(result.pred_score.reshape(-1)[0].item()))
            maps.append(result.anomaly_map.detach().squeeze().cpu().numpy().astype(np.float32))
    return np.asarray(scores), np.stack(maps), seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_efficientad_v1.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    parser.add_argument("--data-root", type=Path, required=True, help="Root that relative annotation_path values resolve against.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="Override for smoke runs only; recorded in metadata.")
    parser.add_argument("--bootstrap-resamples", type=int, help="Override for smoke runs only; recorded in metadata.")
    args = parser.parse_args()

    import torch

    from run_public_baseline import set_seed

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / config["base_config"]).read_text(encoding="utf-8"))
    params = config["efficientad"]
    steps = args.steps or params["steps"]
    ci = dict(base["metrics"]["confidence_interval"])
    if args.bootstrap_resamples:
        ci["resamples"] = args.bootstrap_resamples
    official = args.steps is None and args.bootstrap_resamples is None

    validation_path = args.split_dir / "split_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    metadata_path, assignments_path = args.split_dir / "metadata.csv", args.split_dir / "assignments.csv"
    if validation.get("integrity_status") != "PASS" or not validation.get("experiment_ready"):
        raise SystemExit("Split is not PASS/experiment_ready.")
    if validation["metadata_sha256"] != sha256_file(metadata_path) or validation["assignments_sha256"] != sha256_file(assignments_path):
        raise SystemExit("Split CSVs changed after validation; rebuild the split.")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")

    metadata = {row["image_id"]: row for row in read_csv(metadata_path)}
    split_of = {row["image_id"]: row["split"] for row in read_csv(assignments_path)}
    by_split = {name: sorted(i for i, s in split_of.items() if s == name) for name in ("model", "calibration", "test")}
    size = tuple(base["input_size"])
    roi = config["variant"]["F"]["roi"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.run_dir.mkdir(parents=True)
    meta: dict[str, Any] = {
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config.get("protocol", "D-109"),
        "official_run": official,
        "command": [sys.executable, *sys.argv],
        "config_sha256": sha256_file(args.config),
        "base_config_sha256": sha256_file(ROOT / config["base_config"]),
        "split_validation_sha256": sha256_file(validation_path),
        "split_counts": {k: len(v) for k, v in by_split.items()},
        "steps": steps,
        "bootstrap": ci,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    meta_path = args.run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 1. Preprocess (labels untouched).
    tensors = {i: to_unit_tensor(load_view(Path(metadata[i]["image_path"]), "roi", roi), size)
               for split in by_split.values() for i in split}

    # 2. Train on model images; quantiles and threshold from calibration only.
    set_seed(params["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, train_seconds, quantiles = train_efficientad(
        params, steps, [tensors[i] for i in by_split["model"]], [tensors[i] for i in by_split["calibration"]],
        size, device, args.run_dir / "training_log.csv")
    torch.save(model.state_dict(), args.run_dir / "efficientad_final.pt")
    cal_scores, cal_maps, _ = score(model, [tensors[i] for i in by_split["calibration"]], device)
    test_scores, test_maps, test_seconds = score(model, [tensors[i] for i in by_split["test"]], device)
    threshold = upper_empirical_quantile(cal_scores, 0.95)
    np.savez_compressed(args.run_dir / "raw_efficientad_roi_seed42.npz",
                        calibration_ids=np.asarray(by_split["calibration"]), test_ids=np.asarray(by_split["test"]),
                        calibration_scores=cal_scores, test_scores=test_scores,
                        calibration_maps=cal_maps, test_maps=test_maps)
    (args.run_dir / "frozen_thresholds.json").write_text(json.dumps({"42": {"F": threshold}}, indent=2) + "\n", encoding="utf-8")
    meta.update({
        "train_seconds": train_seconds,
        "map_quantiles": quantiles,
        "inference_ms_per_image": {"p50": float(np.percentile(test_seconds, 50) * 1000),
                                   "p95": float(np.percentile(test_seconds, 95) * 1000)},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
    })

    # 3. Only now read test labels, groups, masks and reference scores.
    test_ids = by_split["test"]
    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids], dtype=np.uint8)
    groups = np.asarray([metadata[i]["sample_id"] for i in test_ids])
    metrics = image_metrics(labels, test_scores, threshold)
    boot = cluster_bootstrap(groups, lambda idx: image_metrics(labels[idx], test_scores[idx], threshold),
                             ci["resamples"], ci["level"], ci["seed"])
    row = {"variant": "F", "threshold": threshold, **metrics, "ci_groups": boot["groups"], "ci_valid_resamples": boot["valid_resamples"]}
    for key, (low, high) in boot["intervals"].items():
        if key in ("image_auroc", "image_auprc", "fpr", "recall", "precision", "f1"):
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = low, high
    write_rows(args.run_dir / "table5_efficientad.csv", [row])

    reference_dir = ROOT / config["comparisons"]["reference_run"]
    paired_rows = []
    for pair, raw_name in config["comparisons"]["paired"].items():
        reference = np.load(reference_dir / raw_name)
        if list(reference["test_ids"]) != test_ids:
            raise SystemExit(f"Reference test order differs for {pair}.")
        other = reference["test_scores"]

        def diff(idx, other=other):
            ma, mb = image_metrics(labels[idx], test_scores[idx], None), image_metrics(labels[idx], other[idx], None)
            return {k: ma[k] - mb[k] for k in config["comparisons"]["metrics"]}

        point = diff(np.arange(len(test_ids)))
        pboot = cluster_bootstrap(groups, diff, ci["resamples"], ci["level"], ci["seed"])
        for metric, value in point.items():
            low, high = pboot["intervals"][metric]
            paired_rows.append({"comparison": pair, "metric": metric, "difference": value, "ci_low": low, "ci_high": high})
    write_rows(args.run_dir / "paired_comparisons.csv", paired_rows)

    mask_paths = [metadata[i]["annotation_path"].strip() for i in test_ids]
    if all(mask_paths[k] for k in np.flatnonzero(labels == 1)):
        from sklearn.metrics import roc_auc_score

        masks = np.stack([load_mask(args.data_root / p, "roi", roi, size) if p else np.zeros(size, np.uint8) for p in mask_paths])
        write_rows(args.run_dir / "localization.csv",
                   [{"variant": "F", "pixel_auroc": float(roc_auc_score(masks.ravel(), test_maps.ravel()))}])
        meta["localization_computed"] = True
    else:
        meta["localization_computed"] = False

    meta.update({"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                 "test_ok": int((labels == 0).sum()), "test_ng": int(labels.sum()),
                 "test_sample_groups": int(np.unique(groups).size)})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(args.run_dir), "F": row, "paired": paired_rows}, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
