from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "analyze_capture_stability.py"


class CaptureStabilityScriptTests(unittest.TestCase):
    def test_writes_reproducible_artifacts_for_manual_reference_burst(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            for index in range(10):
                image = Image.new("RGB", (128, 128), (160, 160, 160))
                exif = Image.Exif()
                exif[41986] = 1  # ExposureMode: manual.
                exif[41987] = 1  # WhiteBalance: manual.
                exif[34855] = 100  # ISO.
                exif[33434] = (1, 50)  # ExposureTime.
                exif[33437] = (18, 10)  # FNumber.
                exif[37386] = (54, 10)  # FocalLength.
                exif[41988] = (1, 1)  # DigitalZoomRatio.
                image.save(input_dir / f"20260930_1200{index:02d}.jpg", exif=exif)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--reference-count",
                    "10",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for filename in [
                "capture_stability.csv",
                "capture_stability.json",
                "analysis_metadata.json",
                "histogram_luminance.png",
                "median_luminance.png",
                "uniformity_grid_cv.png",
            ]:
                self.assertTrue((output_dir / filename).is_file(), filename)
            with (output_dir / "capture_stability.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["manual_exif_status"] == "manual_confirmed" for row in rows))
            self.assertTrue(all(row["illumination_qc_status"] == "reference" for row in rows))
            metadata = json.loads((output_dir / "analysis_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "PASS")
            self.assertTrue(metadata["manual_exif_all_confirmed"])
            self.assertTrue(all(Path(name).suffix != ".md" for name in metadata["output_artifacts"]))
