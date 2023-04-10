from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import zoom

from meerkat import env
from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint, endpoint
from meerkat.interactive.formatter.icon import IconFormatter

from ..app.src.lib.component.core.medimage import MedicalImage
from .base import DeferredFormatter, FormatterGroup
from .image import ImageFormatter

if env.package_available("voxel"):
    import voxel as vx
    from voxel import MedicalVolume
else:
    MedicalVolume = None


class MedicalImageFormatter(ImageFormatter):
    component_class = MedicalImage
    data_prop: str = "data"

    def __init__(
        self,
        max_size: Tuple[int] = None,
        classes: str = "",
        dim: int = 2,
        scrollable: bool = False,
        show_toolbar: bool = False,
        type: str = "image",
        segmentation_column: str = "",
    ):
        if not env.package_available("voxel"):
            raise ImportError(
                "voxel is not installed. Install with `pip install pyvoxel`."
            )
        if not self._is_valid_type(type):
            raise ValueError(
                f"Unknown type '{type}'. Expected ['image', 'segmentation']."
            )

        super().__init__(max_size=max_size, classes=classes)
        self.dim = dim
        self.scrollable = scrollable
        self.show_toolbar = show_toolbar
        self.type = type
        self.segmentation_column = segmentation_column
        self._fetch_data: Endpoint = self.fetch_data.partial(self)

    def encode(
        self,
        cell: MedicalVolume,
        *,
        skip_copy: bool = False,
        dim: int = None,
        type: str = None,
    ) -> List[str]:
        """Encodes an image as a base64 string.

        Args:
            cell: The image to encode.
            dim: The dimension to slice along.
        """
        from meerkat.columns.deferred.base import DeferredCell

        if type is None:
            type = self.type
        if not self._is_valid_type(type):
            raise ValueError(
                f"Unknown type '{type}'. Expected ['image', 'segmentation']."
            )
        if isinstance(cell, DeferredCell):
            cell = cell()
            return self.encode(cell, skip_copy=skip_copy, dim=dim, type=type)

        if dim is None:
            dim = self.dim
        if isinstance(dim, str):
            dim = cell.orientation.index(dim)

        if not self.scrollable:
            # TODO: Update when np.take supported in voxel.
            length = cell.shape[dim]
            sl = slice(length // 2, length // 2 + 1)
            sls = [slice(None), slice(None), slice(None)]
            sls[dim] = sl
            cell = cell[tuple(sls)]

        cell: MedicalVolume = cell.materialize()
        breakpoint
        if type == "image":
            return self._process_image(cell, dim)
        elif type == "segmentation":
            return self._process_segmentation(cell, dim)

    def _is_valid_type(self, type):
        return type in ["image", "segmentation"]

    def _process_image(self, cell, dim: int):
        cell = cell.apply_window(inplace=False)
        cell = self._resize(cell, slice_dim=dim)

        # Put the slice dimension first.
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

    def _process_segmentation(self, cell, dim: int):
        # We assume the segmentation is one-hot encoded.
        arr: np.ndarray = cell.volume

        # Resize the image to match resolution.
        zoom_extent = self._get_zoom_extent(cell, dim)

        # Resize the image to max size.
        if self.max_size:
            inplane_dims = [i for i in range(3) if i != dim]
            inplane_shape = np.asarray([cell.shape[i] for i in inplane_dims])
            downsample_factor = np.max(inplane_shape / np.asarray(self.max_size))
            for _dim in inplane_dims:
                zoom_extent[_dim] /= downsample_factor

        # Zoom into the array (if necessary).
        if not np.allclose(zoom_extent, 1.0):
            arr = zoom(arr, zoom_extent)
            # Threshold the zoomed image.
            arr = arr > 0.5
        arr = arr.astype(bool)

        # Colorize the image.
        arr = _colorize(arr)

        # Convert to 2D slices.
        arr = np.transpose(arr, (dim, *range(dim), *(range(dim + 1, cell.ndim))))
        arr_slices = [PILImage.fromarray(arr[i]) for i in range(arr.shape[0])]

        return [
            super(MedicalImageFormatter, self).encode(x, mode="RGBA")
            for x in arr_slices
        ]

    def _get_zoom_extent(self, mv: MedicalVolume, slice_dim: int) -> MedicalVolume:
        spacing = np.asarray(mv.pixel_spacing)
        inplane_dims = [i for i in range(3) if i != slice_dim]
        inplane_spacing = np.asarray([spacing[dim] for dim in inplane_dims])

        zoom_extent = np.ones(2)

        # Resize for pixel spacing.
        if np.allclose(inplane_spacing, inplane_spacing[0]):
            return np.ones(mv.ndim)

        resolution = np.max(inplane_spacing)
        zoom_extent *= inplane_spacing / resolution

        total_zoom_extent = np.ones(mv.ndim)
        for i, dim in enumerate(inplane_dims):
            total_zoom_extent[dim] = zoom_extent[i]
        return total_zoom_extent

    def _resize(self, mv: MedicalVolume, slice_dim: int) -> MedicalVolume:
        """Resizes a medical volume so that in-plane is the same resolution (in mm).

        Args:
            mv: The medical volume to resize.
            slice_dim: The dimension to slice along.
        """
        spacing = np.asarray(mv.pixel_spacing)
        zoom_extent = self._get_zoom_extent(mv, slice_dim=slice_dim)
        if np.allclose(zoom_extent, 1.0):
            return mv

        arr = zoom(mv.volume, zoom_extent)
        return MedicalVolume(arr, vx.to_affine(mv.orientation, spacing))

    @endpoint()
    def fetch_data(
        self,
        df: DataFrame,
        column: str,
        index: int,
        dim: int = None,
        type: str = "image",
    ) -> List[str]:
        cell = df[column][index]
        return self.encode(cell, dim=dim, type=type)

    @property
    def props(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
            "show_toolbar": self.show_toolbar,
            "on_fetch": self._fetch_data.frontend,
            "dim": self.dim,
            "segmentation_column": self.segmentation_column,
        }

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

    def __init__(self, classes: str = "", dim: int = None, type: str = None):
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

        if dim is not None:
            self.dim = dim
        if type is not None:
            self.type = type

    @property
    def dim(self):
        return self["base"].dim

    @dim.setter
    def dim(self, value):
        for v in self._dict.values():
            v = _get_wrapped_formatter(v)
            if isinstance(v, MedicalImageFormatter):
                v.dim = value

    @property
    def type(self):
        return self["base"].type

    @type.setter
    def type(self, value):
        for v in self._dict.values():
            v = _get_wrapped_formatter(v)
            if isinstance(v, MedicalImageFormatter):
                v.type = value
        if value == "segmentation":
            self._dict["tiny"] = IconFormatter(name="Image")

    @property
    def segmentation_column(self):
        return self["base"].segmentation_column

    @segmentation_column.setter
    def segmentation_column(self, value):
        for v in self._dict.values():
            v = _get_wrapped_formatter(v)
            if isinstance(v, MedicalImageFormatter):
                v.segmentation_column = value


def _get_wrapped_formatter(formatter):
    if isinstance(formatter, DeferredFormatter):
        return _get_wrapped_formatter(formatter.wrapped)
    return formatter


def _colorize(x: np.ndarray):
    """Colorize the segmentation array.

    Args:
        x: An array representing the segmentation (..., num_classes)

    Returns:
        An array of shape (..., 4) where the last dimension is RGBA.
    """
    arr = np.zeros(x.shape[:-1] + (4,), dtype=np.uint8)
    num_classes = x.shape[-1]
    for k in range(num_classes):
        arr[x[..., k]] = np.asarray(_COLORS[k] + (255,))
    return arr


_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 128),
    (255, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
]
