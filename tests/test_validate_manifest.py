from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "validate_manifest.py"
MANIFEST = PROJECT / "data" / "manifests" / "mvtec_tile_w1_seed42.csv"


class ValidateManifestScriptTests(unittest.TestCase):
    def test_writes_machine_readable_validation_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_manifest = Path(temporary_directory) / MANIFEST.name
            shutil.copy2(MANIFEST, temporary_manifest)

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", str(temporary_manifest)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = temporary_manifest.with_suffix(".validation.json")
            self.assertTrue(report.is_file())
            self.assertEqual(list(temporary_manifest.parent.glob("*.md")), [])
            record = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "PASS")
            self.assertEqual(
                record["observed_split_counts"],
                {"model": 184, "calibration": 46, "test": 117},
            )
            self.assertEqual(record["errors"], [])


if __name__ == "__main__":
    unittest.main()
