from typing import Any, Dict, Tuple

import numpy as np

from meerkat import env
from meerkat.interactive.formatter.icon import IconFormatter

from ..app.src.lib.component.core.medimage import MedImage
from .base import FormatterGroup
from .image import ImageFormatter

if env.package_available("voxel"):
    from voxel import MedicalVolume
else:
    MedicalVolume = None


class MedicalImageFormatter(ImageFormatter):
    component_class = MedImage
    data_prop: str = "data"

    def __init__(
        self,
        max_size: Tuple[int] = None,
        classes: str = "",
        dim: int = 2,
        scrollable: bool = False,
        show_toolbar: bool = False,
    ):
        if not env.package_available("voxel"):
            raise ImportError(
                "voxel is not installed. Install with `pip install pyvoxel`."
            )

        super().__init__(max_size=max_size, classes=classes)
        self.dim = dim
        self.scrollable = scrollable
        self.show_toolbar = show_toolbar

    def encode(self, cell: MedicalVolume, *, skip_copy: bool = False) -> str:
        """Encodes an image as a base64 string.

        Args:
            cell: The image to encode.
            dim: The dimension to slice along.
        """
        from meerkat.columns.deferred.file import FileCell

        dim = self.dim

        if isinstance(cell, FileCell):
            cell = cell()
        if isinstance(dim, str):
            dim = cell.orientation.index(dim)

        if not self.scrollable:
            # TODO: Update when np.take supported in voxel.
            length = cell.shape[dim]
            sl = slice(length // 2, length // 2 + 1)
            sls = [slice(None), slice(None), slice(None)]
            sls[dim] = sl
            cell = cell[tuple(sls)]

        cell = cell.materialize()
        cell = cell.apply_window(inplace=True)

        arr = cell.volume
        arr = np.transpose(arr, (dim, *range(dim), *(range(dim + 1, cell.ndim))))

        # Normalize the array to [0, 255] uint8.
        arr = arr - arr.min()
        arr = np.clip(arr, 0, np.percentile(arr, 98))
        arr: np.ndarray = np.round(arr / arr.max() * 255)
        arr = arr.astype(np.uint8)

        arr_slices = [arr[i] for i in range(arr.shape[0])]

        # TODO: Investigate why we need to pass MedicalImageFormatter to super.
        return [super(MedicalImageFormatter, self).encode(x) for x in arr_slices]

    @property
    def props(self) -> Dict[str, Any]:
        return {"classes": self.classes, "show_toolbar": self.show_toolbar}

    def html(self, cell: MedicalVolume) -> str:
        # TODO: Fix standard html rendering.
        encoded = self.encode(cell)
        return f'<img src="{encoded[len(encoded) // 2]}">'

    def _get_state(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "classes": self.classes,
            "dim": self.dim,
            "show_toolbar": self.show_toolbar,
        }

    def _set_state(self, state: Dict[str, Any]):
        self.max_size = state["max_size"]
        self.classes = state["classes"]
        self.dim = state["dim"]
        self.show_toolbar = state["show_toolbar"]


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
                max_size=[512, 512],
                classes="h-full w-full" + " " + classes,
            ),
            full=self.formatter_class(
                classes=classes, scrollable=True, show_toolbar=True
            ),
        )
