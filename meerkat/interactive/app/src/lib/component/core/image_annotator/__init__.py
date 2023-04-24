import base64
import io
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image as PILImage

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty, endpoint
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


class AddCategoryInterface(EventInterface):
    category: str


class ColorChangeEvent(EventInterface):
    category: Hashable
    color: str  # the hex code


class AddBoxInterface(EventInterface):
    box: Any


class AddPointInterface(EventInterface):
    point: Tuple[float, float]


Segmentation = Union[np.ndarray, str]
BoundingBox = Union[np.ndarray, Tuple[float]]


class ImageAnnotator(Component):
    data: Union[np.ndarray, PILImage.Image, str]
    categories: Optional[Union[List, Dict[Hashable, ColorCode]]] = None

    segmentations: Optional[Sequence[Tuple[Segmentation, str]]] = None
    points: Optional[Sequence[Dict]] = None
    boxes: Optional[Sequence[Dict]] = None

    opacity: float = 0.85

    selected_category: str = ""

    # TODO: Parameters to add
    # boxes: Bounding boxes to draw on the image.
    # polygons: Polygons to draw on the image.
    on_add_category: EndpointProperty[AddCategoryInterface] = None
    on_add_box: EndpointProperty[AddBoxInterface] = None
    on_add_point: EndpointProperty[AddPointInterface] = None

    def __init__(
        self,
        data,
        *,
        categories=None,
        segmentations=None,
        points=None,
        boxes=None,
        opacity: float = 0.85,
        selected_category: str = "",
        on_add_category: EndpointProperty[AddCategoryInterface] = None,
        on_add_box: EndpointProperty[AddBoxInterface] = None,
        on_add_point: EndpointProperty[AddPointInterface] = None,
    ):
        """
        Args:
            data: The base image.
                Strings must be base64 encoded or a filepath to the image.
            categories: The categories in the image. These categories will be used
                for all annotations. Can either be a list of category names, a
                dictionary mapping category names to colors, or a DataFrame
                with two columns ("name" and "color").
            segmentations: A list of (mask, category) tuples.
            opacity: The initial opacity of the segmentation masks.
            on_select: An endpoint to call when the user clicks on the image.
        """
        if points is None:
            points = []
        if boxes is None:
            boxes = []

        if categories is None:
            categories = [category for _, category in self.segmentations]
        if isinstance(categories, (tuple, list)):
            categories = dict(zip(categories, generate_random_colors(len(categories))))

        super().__init__(
            data=data,
            categories=categories,
            segmentations=segmentations,
            points=points,
            boxes=boxes,
            opacity=opacity,
            selected_category=selected_category,
            on_add_category=on_add_category,
            on_add_box=on_add_box,
            on_add_point=on_add_point,
        )
        self.data = self.prepare_data(self.data)

        categories = self.prepare_categories(self.categories)

        self.segmentations = colorize_segmentations(self.segmentations, categories)
        # At some point get rid of this and see if we can pass colorized segmentations.
        self.segmentations = encode_segmentations(self.segmentations)

        # Initialize endpoints
        self.on_add_category = self._add_category.partial(self)
        # self.on_clear_annotations = self.clear_annotations.partial(self)

    @reactive()
    def prepare_data(self, data):
        if isinstance(data, str):
            return str(data)

        from meerkat.interactive.formatter.image import ImageFormatter

        # TODO: Intelligently pick what the mode should be.
        return ImageFormatter().encode(data, mode="RGB")

    @reactive()
    def prepare_categories(self, categories):

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

    @endpoint()
    def _add_category(self, category):
        if category not in self.categories:
            self.categories[category] = generate_random_colors(1)[0]
            self.categories.set(self.categories)

    def clear_annotations(self, annotation_type: Optional[str] = None):
        self.points.set([])
        self.segmentations.set([])

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
