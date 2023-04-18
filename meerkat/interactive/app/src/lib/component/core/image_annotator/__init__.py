import base64
import io
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image as PILImage

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface
from meerkat.interactive.graph.reactivity import reactive
from meerkat.interactive.graph.store import Store

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

ColorCode = Union[str, Tuple[int, int, int]]


class SelectInterface(EventInterface):
    x: float
    y: float
    click_type: Literal["single", "double", "right"]


class ColorChangeEvent(EventInterface):
    category: Hashable
    color: str  # the hex code


Segmentation = Union[np.ndarray, str]
BoundingBox = Union[np.ndarray, Tuple[float]]


class ImageAnnotator(Component):
    data: Union[np.ndarray, PILImage.Image, str]
    categories: Optional[Union[List, Dict[Hashable, ColorCode]]] = None

    segmentations: Optional[Sequence[Tuple[Segmentation, str]]] = None
    points: Optional[Sequence[Dict]] = None

    opacity: float = 0.85

    # TODO: Parameters to add
    # boxes: Bounding boxes to draw on the image.
    # polygons: Polygons to draw on the image.
    on_select: Endpoint[SelectInterface] = None

    def __init__(
        self,
        data,
        *,
        categories=None,
        segmentations=None,
        opacity: float = 0.85,
        on_select: Endpoint[SelectInterface] = None,
    ):
        """
        Args:
            data: The base image.
                Strings must be base64 encoded or a filepath to the image.
            categories: A list of categories or a dictionary mapping
            segmentations: A list of (mask, category) tuples.
            opacity: The initial opacity of the segmentation masks.
            on_select: An endpoint to call when the user clicks on the image.
        """
        super().__init__(
            data=data,
            categories=categories,
            segmentations=segmentations,
            opacity=opacity,
            on_select=on_select,
        )
        self.data = self.prepare_data(self.data)
        if self.categories is None:
            self.categories = [category for _, category in self.segmentations]

        self.categories = self.prepare_categories(self.categories)
        self.segmentations = colorize_segmentations(self.segmentations, self.categories)
        # At some point get rid of this and see if we can pass colorized segmentations.
        self.segmentations = encode_segmentations(self.segmentations)

    @reactive()
    def prepare_data(self, data):
        if isinstance(data, str):
            return str(data)

        from meerkat.interactive.formatter.image import ImageFormatter

        # TODO: Intelligently pick what the mode should be.
        return ImageFormatter().encode(data, mode="RGB")

    @reactive()
    def prepare_categories(self, categories):
        if isinstance(categories, (Tuple, List)):
            return dict(zip(categories, generate_random_colors(len(categories))))

        # Convert hex colors (if necessary).
        # This line also creates a shallow copy of the dictionary,
        # which is necessary to avoid mutating the original dictionary
        # (required for reactive functions).
        categories = {
            k: _from_hex(v) if isinstance(v, str) else v for k, v in categories.items()
        }

        # Make sure all colors are in RGBA format.
        for k in categories:
            if len(categories[k]) == 3:
                categories[k] = np.asarray(tuple(categories[k]) + (255,))

        return categories

    # @endpoint()
    # def on_color_change(self, category: Hashable, color: ColorCode):
    #     self.categories[category] = _fromcolor


@reactive()
def colorize_segmentations(segmentations, categories: Dict[Hashable, np.ndarray]):
    """Colorize the segmentation masks.

    We assume segmentations are in the form of (array, category) tuples.
    ``categories`` is a dictionary mapping categories to RGB colors.

    Returns:
        A list of RGBA numpy arrays - shape: (H, W, 4).
    """
    if segmentations is None:
        return None

    return Store(
        [
            (_colorize_mask(segmentation, categories[name]), name)
            for segmentation, name in segmentations
        ],
        backend_only=True,
    )


@reactive()
def encode_segmentations(segmentations):
    """Encode the segmentation masks as base64 strings.

    We assume segmentations are in the form of (array, category) tuples.

    Returns:
        A list of (base64 string, category) tuples.
    """
    if segmentations is None:
        return None

    return [(_encode_mask(segmentation), name) for segmentation, name in segmentations]


def _colorize_mask(mask, color):
    # TODO: Add support for torch tensors.
    color_mask = np.zeros(mask.shape + (4,), dtype=np.uint8)
    if len(color) == 3:
        color = np.asarray(tuple(color) + (255,))
    if not isinstance(color, np.ndarray):
        color = np.asarray(color)

    color_mask[mask] = color
    return color_mask


def _encode_mask(colored_mask):
    """Encode a colored mask as a base64 string."""
    ftype = "png"
    colored_mask = PILImage.fromarray(colored_mask, mode="RGBA")
    with io.BytesIO() as buffer:
        colored_mask.save(buffer, format=ftype)
        return "data:image/{ftype};base64,{im_base_64}".format(
            ftype=ftype, im_base_64=base64.b64encode(buffer.getvalue()).decode()
        )


def _from_hex(color: str):
    """Convert a hex color to an RGB tuple."""
    color = color.lstrip("#")
    if len(color) % 2 != 0:
        raise ValueError("Hex color must have an even number of digits.")
    return np.asarray(
        int(color[i * 2 : (i + 1) + 2], 16) for i in range(len(color) // 2)
    )


def generate_random_colors(n: int):
    """Generate ``n`` random colors.

    Args:
        n: The number of colors to generate.

    Returns:
        A list of ``n`` random uint8 colors in RGBA format.
    """
    out = np.random.randint(0, 255, (n, 3), dtype=np.uint8)
    out = np.concatenate((out, np.full((n, 1), 255, dtype=np.uint8)), axis=1)
    return out
