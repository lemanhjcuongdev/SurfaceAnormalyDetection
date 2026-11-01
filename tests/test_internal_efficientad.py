import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_internal_efficientad import to_unit_tensor  # noqa: E402


def test_to_unit_tensor_keeps_unit_range_without_imagenet_normalisation():
    image = Image.fromarray(np.full((40, 30, 3), 255, dtype=np.uint8))
    tensor = to_unit_tensor(image, (16, 12))
    assert tuple(tensor.shape) == (3, 16, 12)
    assert float(tensor.min()) == 1.0 and float(tensor.max()) == 1.0
