"""Manifest-backed Anomalib datasets for the locked W1 public benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from anomalib.data import AnomalibDataset
from anomalib.data.utils import LabelName
from torchvision.transforms.v2 import Compose, Normalize, Resize


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform(input_size: tuple[int, int]) -> Compose:
    """Return the one explicit image-and-mask transform used by W1."""
    return Compose(
        [
            Resize(input_size, antialias=True),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ManifestImageDataset(AnomalibDataset):
    """Anomalib-compatible dataset whose membership is fixed by a CSV manifest.

    The class does not scan directories or split files. It converts only the
    supplied manifest rows into an Anomalib samples table, preserving the
    original source image and mask paths.
    """

    def __init__(
        self,
        rows: Iterable[dict[str, str]],
        tile_root: Path,
        data_split: str,
        input_size: tuple[int, int],
    ) -> None:
        super().__init__(augmentations=image_transform(input_size))
        records: list[dict[str, object]] = []
        for row in rows:
            image_path = tile_root / row["image_relative_path"]
            mask_relative_path = row.get("mask_relative_path", "")
            records.append(
                {
                    "record_id": row["record_id"],
                    "image_path": str(image_path.resolve()),
                    "mask_path": (
                        str((tile_root / mask_relative_path).resolve())
                        if mask_relative_path
                        else ""
                    ),
                    "label_index": (
                        LabelName.ABNORMAL
                        if row["label_binary"] == "1"
                        else LabelName.NORMAL
                    ),
                    "split": data_split,
                    "defect_type": row["defect_type"],
                }
            )
        if not records:
            raise ValueError(f"No records supplied for manifest split {data_split}.")

        samples = pd.DataFrame.from_records(records)
        samples.attrs["task"] = "segmentation"
        self.samples = samples
        self.category = "tile"
        self._name = "MVTecADManifest"

    @property
    def name(self) -> str:
        """Use a stable dataset name in Anomalib workspace paths."""
        return self._name

