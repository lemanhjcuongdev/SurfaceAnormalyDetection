"""Split assignment, metrics and bootstrap checks for the internal v1 protocol (no model, no images)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_internal_split as split  # noqa: E402
import run_internal_experiment as experiment  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "internal_main_experiment_v1.yaml").read_text(encoding="utf-8"))


def row(image_id: str, session: str, sample: str | None, label: str = "OK") -> dict[str, object]:
    return {"image_id": image_id, "capture_session": session, "sample_id": sample, "label": label}


def registry_rows() -> list[dict[str, object]]:
    rows = []
    for number in range(10):
        rows.append(row(f"S04-{number}", "20261009-S04", f"M{number:02d}"))
        rows.append(row(f"S04-{number}b", "20261009-S04", f"M{number:02d}"))
    rows += [row(f"S06-{n}", "20261016-S06", f"M2{n}") for n in range(5)]
    rows += [row("S08-ok", "20261029-S08", "M50"), row("S08-ng", "20261029-S08", "M51", "NG"),
             row("S09-ng", "20261030-S09", "M52", "NG")]
    return rows


class SplitAssignmentTests(unittest.TestCase):
    def test_sessions_samples_and_calibration_rules(self) -> None:
        assignments, report = split.assign_splits(registry_rows(), CONFIG["split"])
        self.assertEqual({assignments[i] for i in ("S08-ok", "S08-ng", "S09-ng")}, {"test"})
        self.assertTrue(all(assignments[f"S06-{n}"] == "model" for n in range(5)))
        for number in range(10):  # both images of one sample share a split
            self.assertEqual(assignments[f"S04-{number}"], assignments[f"S04-{number}b"])
        self.assertEqual(len(report["calibration_samples"]), round(0.2 * 15))
        self.assertTrue(all(s.startswith("M0") for s in report["calibration_samples"]))

    def test_sample_seen_in_test_session_is_excluded_from_development(self) -> None:
        rows = registry_rows() + [row("S07-dup", "20261023-S07", "M51")]
        assignments, report = split.assign_splits(rows, CONFIG["split"])
        self.assertEqual(assignments["S07-dup"], "excluded")
        self.assertEqual(report["conflict_excluded_images"], ["S07-dup"])

    def test_assignment_is_deterministic(self) -> None:
        first, _ = split.assign_splits(registry_rows(), CONFIG["split"])
        second, _ = split.assign_splits(registry_rows(), CONFIG["split"])
        self.assertEqual(first, second)

    def test_session_folder(self) -> None:
        self.assertEqual(split.session_folder("20261029-S08"), "RESOURCES/29102026")


class MetricTests(unittest.TestCase):
    def test_thresholded_metrics(self) -> None:
        labels = np.asarray([0, 0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.9, 0.8, 0.95])
        result = experiment.image_metrics(labels, scores, 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (2, 1, 0, 2))
        self.assertAlmostEqual(result["fpr"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertNotIn("tp", experiment.image_metrics(labels, scores, None))

    def test_cluster_bootstrap_keeps_groups_together_and_skips_one_class(self) -> None:
        groups = np.asarray(["a", "a", "b", "c", "c"])
        labels = np.asarray([0, 0, 0, 1, 1])
        seen_sizes = set()

        def stat(indices):
            seen_sizes.add(tuple(np.bincount(np.searchsorted(["a", "b", "c"], groups[indices]), minlength=3) % 2))
            return experiment.image_metrics(labels[indices], labels[indices].astype(float), None)

        result = experiment.cluster_bootstrap(groups, stat, 200, 0.95, 0)
        self.assertEqual(result["groups"], 3)
        self.assertGreater(result["skipped_resamples"], 0)  # resamples without both classes
        low, high = result["intervals"]["image_auroc"]
        self.assertEqual((low, high), (1.0, 1.0))
        self.assertTrue(all(size[0] == 0 and size[2] == 0 for size in seen_sizes))  # pairs never split


if __name__ == "__main__":
    unittest.main()
