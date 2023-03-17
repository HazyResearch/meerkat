import base64
import os
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image as PILImage

from meerkat import env
from meerkat.columns.deferred.base import DeferredCell
from meerkat.interactive.formatter.icon import IconFormatter
from meerkat.tools.lazy_loader import LazyLoader

from ..app.src.lib.component.core.image import Image
from .base import BaseFormatter, FormatterGroup

torch = LazyLoader("torch")


class ImageFormatter(BaseFormatter):
    component_class = Image
    data_prop: str = "data"

    def __init__(self, max_size: Tuple[int] = None, classes: str = ""):
        super().__init__()
        self.max_size = max_size
        self.classes = classes

    def encode(self, cell: PILImage, skip_copy: bool = False) -> str:
        """Encodes an image as a base64 string.

        Args:
            cell: The image to encode.
            skip_copy: If True, the image may be modified in place.
                Set to ``True`` if the image is already a copy
                or is loaded dynamically (e.g. DeferredColumn).
                This may save time for large images.
        """
        if env.package_available("torch") and isinstance(cell, torch.Tensor):
            cell = cell.cpu().numpy()

        if isinstance(cell, np.ndarray):
            cell = PILImage.fromarray(cell)
            # We can skip copying if we are constructing the image from a numpy array.
            skip_copy = True

        with BytesIO() as buffer:
            if self.max_size:
                # Image.thumbnail modifies the image in place, so we need to
                # make a copy first.
                if not skip_copy:
                    cell = cell.copy()
                cell.thumbnail(self.max_size)
            cell.save(buffer, "jpeg")
            return "data:image/jpeg;base64,{im_base_64}".format(
                im_base_64=base64.b64encode(buffer.getvalue()).decode()
            )

    @property
    def props(self) -> Dict[str, Any]:
        return {"classes": self.classes}

    def html(self, cell: Image) -> str:
        encoded = self.encode(cell)
        return f'<img src="{encoded}">'

    def _get_state(self) -> Dict[str, Any]:
        return {
            "max_size": self.max_size,
            "classes": self.classes,
        }

    def _set_state(self, state: Dict[str, Any]):
        self.max_size = state["max_size"]
        self.classes = state["classes"]


class ImageFormatterGroup(FormatterGroup):
    formatter_class: type = ImageFormatter

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


class DeferredImageFormatter(ImageFormatter):
    component_class: type = Image
    data_prop: str = "data"

    def encode(self, image: DeferredCell) -> str:
        if hasattr(image, "absolute_path"):
            absolute_path = image.absolute_path
            if isinstance(absolute_path, os.PathLike):
                absolute_path = str(absolute_path)
            if isinstance(absolute_path, str) and absolute_path.startswith("http"):
                return image.absolute_path

        image = image()
        return super().encode(image, skip_copy=True)


class DeferredImageFormatterGroup(ImageFormatterGroup):
    formatter_class: type = DeferredImageFormatter
