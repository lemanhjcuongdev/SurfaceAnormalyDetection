"""Checks for the locked latency/memory protocol without loading a model."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import measure_latency_memory as latency  # noqa: E402


CONFIG = ROOT / "configs" / "public_tile_latency_patchcore.yaml"


class LatencyProtocolTests(unittest.TestCase):
    def test_locked_config_passes(self) -> None:
        config = latency.read_protocol(CONFIG)
        self.assertEqual(config["devices"], ["cuda", "cpu"])
        self.assertFalse(config["labels_read"])

    def test_changed_protocol_is_rejected(self) -> None:
        with CONFIG.open("r", encoding="utf-8") as source:
            config = yaml.safe_load(source)
        config["warmup_image_count"] = 3
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "changed.yaml"
            changed.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                latency.read_protocol(changed)

    def test_summary_uses_nearest_rank_percentiles(self) -> None:
        summary = latency.summarize([float(value) for value in range(1, 21)])
        self.assertEqual(summary["count"], 20)
        self.assertEqual(summary["p50_ms"], 10.0)
        self.assertEqual(summary["p95_ms"], 19.0)
        self.assertAlmostEqual(summary["mean_ms"], 10.5)
        self.assertEqual((summary["min_ms"], summary["max_ms"]), (1.0, 20.0))

    def test_rows_follow_template_columns(self) -> None:
        row = latency.template_row({"run_id": "x", "p95_latency_ms": 1.0})
        self.assertEqual(list(row), latency.template_columns())
        with self.assertRaises(ValueError):
            latency.template_row({"not_a_column": 1})


if __name__ == "__main__":
    unittest.main()
