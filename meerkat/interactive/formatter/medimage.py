from typing import Any, Dict, Tuple

import numpy as np
from voxel import MedicalVolume

from meerkat.interactive.formatter.icon import IconFormatter

from ..app.src.lib.component.core.medimage import MedImage
from .base import FormatterGroup
from .image import ImageFormatter


class MedicalImageFormatter(ImageFormatter):
    component_class = MedImage
    data_prop: str = "data"

    def __init__(self, max_size: Tuple[int] = None, classes: str = ""):
        self.max_size = max_size
        self.classes = classes

    def encode(
        self, cell: MedicalVolume, *, skip_copy: bool = False, dim: int = 0
    ) -> str:
        """Encodes an image as a base64 string.

        Args:
            cell: The image to encode.
            dim: The dimension to slice along.
        """
        if isinstance(dim, str):
            dim = cell.orientation.index(dim)

        cell = cell.materialize()
        arr = np.transpose(
            cell.volume, (dim, *range(dim), *(range(dim + 1, cell.ndim)))
        )

        arr_slices = [arr[i : i + 1] for i in range(arr.shape[0])]

        return {
            "data": [super().encode(x) for x in arr_slices],
            "numSlices": arr.shape[0],
        }

    @property
    def props(self) -> Dict[str, Any]:
        return {"classes": self.classes}

    def html(self, cell: MedicalVolume) -> str:
        encoded = self.encode(cell)
        return f'<img src="{encoded[len(encoded) // 2]}">'

    def _get_state(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "classes": self.classes,
        }

    def _set_state(self, state: Dict[str, Any]):
        self.max_size = state["max_size"]
        self.classes = state["classes"]


class MedicalImageFormatterGroup(FormatterGroup):
    formatter_class: type = MedicalImageFormatter

    def __init__(self, classes: str = ""):
        super().__init__(
            icon=IconFormatter(name="Image"),
            base=self.formatter_class(classes=classes),
            tiny=self.formatter_class(max_size=[32, 32], classes=classes),
            small=self.formatter_class(max_size=[64, 64], classes=classes),
            thumbnail=self.formatter_class(max_size=[256, 256], classes=classes),
            gallery=self.formatter_class(
                max_size=[512, 512], classes="h-full w-full" + " " + classes
            ),
        )
