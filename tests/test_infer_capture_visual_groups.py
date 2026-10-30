from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "infer_capture_visual_groups.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("infer_capture_visual_groups", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Không tải được infer_capture_visual_groups.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualGroupInferenceTests(unittest.TestCase):
    def test_crop_applies_exif_orientation_before_roi(self) -> None:
        module = load_script_module()
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "portrait_orientation_6.jpg"
            image = Image.new("RGB", (400, 300), (0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 199, 149), fill=(240, 20, 20))
            draw.rectangle((200, 0, 399, 149), fill=(20, 240, 20))
            draw.rectangle((0, 150, 199, 299), fill=(20, 20, 240))
            draw.rectangle((200, 150, 399, 299), fill=(240, 240, 20))
            exif = Image.Exif()
            exif[274] = 6
            image.save(image_path, exif=exif)

            actual = module.crop_rgb(image_path, (0, 0, 300, 400))
            with Image.open(image_path) as source:
                expected_image = ImageOps.exif_transpose(source).convert("RGB")
                expected = np.asarray(
                    expected_image.crop((0, 0, 300, 400)).resize((96, 64), Image.Resampling.BILINEAR),
                    dtype=np.float64,
                ) / 255.0
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_writes_provisional_csv_and_json_without_raw_image_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            images = root / "images"
            output = root / "output"
            images.mkdir()
            Image.new("RGB", (300, 200), (150, 150, 150)).save(images / "a.jpg")
            Image.new("RGB", (300, 200), (150, 150, 150)).save(images / "b.jpg")
            patterned = Image.new("RGB", (300, 200), (30, 30, 30))
            draw = ImageDraw.Draw(patterned)
            for x in range(0, 300, 16):
                draw.line((x, 0, x, 200), fill=(230, 230, 230), width=6)
            patterned.save(images / "c.jpg")

            mapping = root / "mapping.csv"
            mapping.write_text(
                "capture_group_id,capture_session,source_folder,source_file_1\n"
                "G01,S01,local,a.jpg\n"
                "G02,S01,local,b.jpg\n"
                "G03,S01,local,c.jpg\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--mapping",
                    str(mapping),
                    "--images-dir",
                    str(images),
                    "--output-dir",
                    str(output),
                    "--roi",
                    "0,0,300,200",
                    "--family-threshold",
                    "0.94",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (output / "visual_group_inference.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["inference_status"], "provisional_visual_inference_not_for_split")
            self.assertEqual(rows[0]["inferred_visual_family"], rows[1]["inferred_visual_family"])
            self.assertNotEqual(rows[0]["inferred_visual_family"], rows[2]["inferred_visual_family"])
            summary = json.loads((output / "visual_group_inference_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["raw_images_copied"])
            self.assertEqual(summary["status"], "PROVISIONAL")
            self.assertTrue(all(path.suffix not in {".jpg", ".png"} for path in output.iterdir()))


if __name__ == "__main__":
    unittest.main()
