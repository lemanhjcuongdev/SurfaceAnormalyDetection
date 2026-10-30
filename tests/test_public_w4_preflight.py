from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_public_w4.py"
CONFIG = PROJECT / "configs" / "public_tile_w4_normal_only.yaml"


class PublicW4PreflightTests(unittest.TestCase):
    def test_locked_w4_protocol_passes_without_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            baseline_dir = Path(temporary_directory)
            (baseline_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "manifest_sha256": "f8e2ef28194944912d0790bf2b8bb42c2e13ebbee75061b3d92f5f730479fa3b",
                        "model_config": {"name": "Patchcore"},
                        "automatic_threshold": False,
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "preflight",
                    "--config",
                    str(CONFIG),
                    "--baseline-dir",
                    str(baseline_dir),
                    "--output-dir",
                    str(PROJECT / "runs" / "unused_preflight_output"),
                ],
                cwd=PROJECT,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(result.stdout)
        self.assertEqual(record["status"], "PASS")
        self.assertEqual(record["split_counts"], {"model": 184, "calibration": 46, "test": 117})
        self.assertIn("No test score", record["selection_policy"])
        self.assertFalse((PROJECT / "runs" / "unused_preflight_output").exists())


if __name__ == "__main__":
    unittest.main()
