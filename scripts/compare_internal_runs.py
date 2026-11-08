"""Paired differences between two internal runs on the same test images (protocol D-126).

Reads stored scores, maps and thresholds only; nothing is fitted or tuned. The later
run is compared with the baseline on the same bootstrap resamples, grouped by
`sample_id`. Used to measure the effect of a data change, never to select anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from iad_w1.normal_calibration import smooth_map_3x3  # noqa: E402
from run_internal_experiment import cluster_bootstrap, image_metrics, load_mask, read_csv, sha256_file, write_rows  # noqa: E402
from run_internal_localization import SOURCES, auroc_from_counts, binned_counts, pixel_auroc, registry_ng  # noqa: E402

RANKING_METRICS = ("image_auroc", "image_auprc")
DECISION_METRICS = ("recall", "fpr", "f1")
DECISION_VARIANTS = ("D", "E", "F")


def load_run(reference: Path, efficientad: Path) -> dict[str, Any]:
    """Variant maps, scores and thresholds of one run pair, in stored test order."""
    runs = {"reference_run": reference, "efficientad_run": efficientad}
    thresholds: dict[str, float] = {}
    for run in runs.values():
        thresholds.update(json.loads((run / "frozen_thresholds.json").read_text(encoding="utf-8"))["42"])
    maps, scores, test_ids = {}, {}, None
    for variant, (run_key, name, smooth) in SOURCES.items():
        raw = np.load(runs[run_key] / name)
        ids = [str(i) for i in raw["test_ids"]]
        if test_ids is not None and ids != test_ids:
            raise SystemExit(f"Test order differs inside one run for {variant}")
        test_ids = ids
        test_maps = raw["test_maps"]
        if smooth:
            test_maps = np.stack([smooth_map_3x3(m) for m in test_maps])
        maps[variant] = test_maps
        scores[variant] = test_maps.reshape(len(test_maps), -1).max(axis=1) if smooth else raw["test_scores"]
    return {"test_ids": test_ids, "maps": maps, "scores": scores, "thresholds": thresholds}


def difference_rows(point: dict[str, float], boot: dict[str, Any], extra: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**extra, "metric": metric, "difference": value, "ci_low": boot["intervals"][metric][0],
             "ci_high": boot["intervals"][metric][1], "ci_valid_resamples": boot["valid_resamples"]}
            for metric, value in point.items()]


def image_differences(labels: np.ndarray, groups: np.ndarray, later: dict[str, Any], earlier: dict[str, Any],
                      boot_args: tuple[int, float, int]) -> list[dict[str, Any]]:
    """Later minus earlier, per variant: ranking metrics, then decisions at each run's own threshold."""
    rows: list[dict[str, Any]] = []
    all_idx = np.arange(len(labels))
    for variant in later["scores"]:
        a, b = later["scores"][variant], earlier["scores"][variant]

        def ranking(idx, a=a, b=b):
            ma, mb = image_metrics(labels[idx], a[idx], None), image_metrics(labels[idx], b[idx], None)
            return {k: ma[k] - mb[k] for k in RANKING_METRICS}

        rows += difference_rows(ranking(all_idx), cluster_bootstrap(groups, ranking, *boot_args), {"variant": variant})
        if variant not in DECISION_VARIANTS:
            continue
        ta, tb = later["thresholds"][variant], earlier["thresholds"][variant]

        def decision(idx, a=a, b=b, ta=ta, tb=tb):
            ma, mb = image_metrics(labels[idx], a[idx], ta), image_metrics(labels[idx], b[idx], tb)
            return {k: ma[k] - mb[k] for k in DECISION_METRICS}

        rows += difference_rows(decision(all_idx), cluster_bootstrap(groups, decision, *boot_args), {"variant": variant})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "internal_main_experiment_v2.yaml")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v2")
    parser.add_argument("--baseline-split-dir", type=Path, default=ROOT / "data" / "manifests" / "internal_v1")
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "metadata" / "INTERNAL_IMAGE_REGISTRY.xlsx")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--efficientad-run", type=Path, required=True)
    parser.add_argument("--baseline-reference-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261101-run01")
    parser.add_argument("--baseline-efficientad-run", type=Path, default=ROOT / "runs" / "internal_main_v1" / "20261101-efficientad01")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be empty: {args.run_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ci = config["metrics"]["confidence_interval"]
    boot_args = (ci["resamples"], ci["level"], ci["seed"])
    roi, size = config["variants"]["B"]["roi"], tuple(config["input_size"])

    later = load_run(args.reference_run, args.efficientad_run)
    earlier = load_run(args.baseline_reference_run, args.baseline_efficientad_run)
    if later["test_ids"] != earlier["test_ids"]:
        raise SystemExit("The two runs were not scored on the same test images.")
    test_ids = later["test_ids"]
    metadata = {row["image_id"]: row for row in read_csv(args.split_dir / "metadata.csv")}
    baseline_metadata = {row["image_id"]: row for row in read_csv(args.baseline_split_dir / "metadata.csv")}
    for image_id in test_ids:
        for field in ("label", "sample_id", "image_sha256"):
            if metadata[image_id][field] != baseline_metadata[image_id][field]:
                raise SystemExit(f"{image_id}: {field} differs between the two manifests.")
    labels = np.asarray([1 if metadata[i]["label"] == "NG" else 0 for i in test_ids], dtype=np.uint8)
    groups = np.asarray([metadata[i]["sample_id"] for i in test_ids])

    image_rows = image_differences(labels, groups, later, earlier, boot_args)

    ng_registry = registry_ng(args.registry)
    masks = np.zeros((len(test_ids), *size), np.uint8)
    for k in np.flatnonzero(labels == 1):
        masks[k] = load_mask(args.data_root / ng_registry[test_ids[k]]["annotation_path"].strip(), "roi", roi, size)
    all_idx = np.arange(len(test_ids))
    pixel_rows: list[dict[str, Any]] = []
    for variant in later["maps"]:
        pa, na = binned_counts(masks, later["maps"][variant])
        pb, nb = binned_counts(masks, earlier["maps"][variant])

        def diff(idx, pa=pa, na=na, pb=pb, nb=nb):
            return {"pixel_auroc": auroc_from_counts(pa[idx].sum(0), na[idx].sum(0)) - auroc_from_counts(pb[idx].sum(0), nb[idx].sum(0))}

        point = {"pixel_auroc": pixel_auroc(masks, later["maps"][variant], all_idx) - pixel_auroc(masks, earlier["maps"][variant], all_idx)}
        pixel_rows += difference_rows(point, cluster_bootstrap(groups, diff, *boot_args), {"variant": variant})

    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.run_dir / "paired_image_vs_baseline.csv", image_rows)
    write_rows(args.run_dir / "paired_pixel_vs_baseline.csv", pixel_rows)
    meta = {
        "protocol": config.get("protocol"), "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "config_sha256": sha256_file(args.config),
        "reference_run": str(args.reference_run), "efficientad_run": str(args.efficientad_run),
        "baseline_reference_run": str(args.baseline_reference_run), "baseline_efficientad_run": str(args.baseline_efficientad_run),
        "thresholds_later": {v: later["thresholds"][v] for v in DECISION_VARIANTS},
        "thresholds_baseline": {v: earlier["thresholds"][v] for v in DECISION_VARIANTS},
        "bootstrap": ci,
    }
    (args.run_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"image": image_rows, "pixel": pixel_rows}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
