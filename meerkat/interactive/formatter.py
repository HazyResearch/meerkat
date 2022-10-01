"""Desiderata for formatters:

1.
"""
import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import PIL
import torch
from pandas.io.formats.format import format_array
from PIL.Image import Image

if TYPE_CHECKING:
    from meerkat.columns.file_column import FileCell


class Formatter(ABC):

    # one of the front end cell components implemented here:
    # meerkat/interactive/app/src/lib/components/item
    # (e.g. "image", "code")
    cell_component: str

    @abstractmethod
    def encode(self, cell: Any):
        """Encode the cell on the backend side before sending it to the
        frontend.

        The cell is lazily loaded, so when used on a LambdaColumn,
        ``cell`` will be a ``LambdaCell``. This is important for
        displays that don't actually need to apply the lambda in order
        to display the value.
        """
        pass

    @abstractmethod
    def html(self, cell: Any):
        """When not in interactive mode, objects are visualized using static
        html.

        This method should produce that static html for the cell.
        """
        pass

    @property
    def cell_props(self):
        return {}


class BasicFormatter(Formatter):
    cell_component = "basic"

    DTYPES = [
        "str",
        "int",
        "float",
        "bool",
    ]

    def __init__(self, dtype: str = "str"):
        if dtype not in self.DTYPES:
            raise ValueError(
                f"dtype {dtype} not supported. Must be one of {self.DTYPES}"
            )
        self.dtype = dtype

    def encode(self, cell: Any):
        if isinstance(cell, np.generic):
            return cell.item()
        return cell

    def html(self, cell: Any):
        cell = self.encode(cell)
        return format_array(np.array([cell]), formatter=None)[0]

    @property
    def cell_props(self):
        return {
            # backwards compatability
            "dtype": self.dtype
            if hasattr(self, "dtype")
            else "str",
        }


class ObjectFormatter(Formatter):
    cell_component = "object"

    def encode(self, cell: Any):
        return str(cell)

    def html(self, cell: Any):
        return str(cell)


class NumpyArrayFormatter(BasicFormatter):
    cell_component = "basic"

    def encode(self, cell: Any):
        if isinstance(cell, np.ndarray):
            return str(cell)
        return super().encode(cell)

    def html(self, cell: Any):
        if isinstance(cell, np.ndarray):
            return str(cell)
        return format_array(np.array([cell]), formatter=None)[0]


class IntervalFormatter(NumpyArrayFormatter):
    cell_component = "interval"

    def encode(self, cell: Any):
        if cell is not np.ndarray:
            return super().encode(cell)
        
        if cell.shape[0] != 3:
            raise ValueError(
                "Cell used with `IntervalFormatter` must be np.ndarray length 3 "
                "length 3. Got shape {}".format(cell.shape)
            )

        return [super().encode(v) for v in cell] 

    def html(self, cell: Any):
        if isinstance(cell, np.ndarray):
            return str(cell)
        return format_array(np.array([cell]), formatter=None)[0]


class TensorFormatter(BasicFormatter):
    cell_component = "basic"

    def encode(self, cell: Any):
        if isinstance(cell, torch.Tensor):
            return str(cell)
        return super().encode(cell)

    def html(self, cell: Any):
        if isinstance(cell, torch.Tensor):
            return str(cell)
        return format_array(np.array([cell]), formatter=None)[0]


class PILImageFormatter(Formatter):

    cell_component = "image"

    def encode(
        self, cell: Union["FileCell", Image, torch.Tensor], thumbnail: bool = False
    ) -> str:
        from meerkat.columns.lambda_column import LambdaCell

        if isinstance(cell, LambdaCell):
            cell = cell.get()

        if torch.is_tensor(cell) or isinstance(cell, np.ndarray):
            from torchvision.transforms.functional import to_pil_image

            try:
                cell = to_pil_image(cell)
            except ValueError:
                if isinstance(cell, np.ndarray):
                    cell = NumpyArrayFormatter.encode(self, cell)
                else:
                    cell = TensorFormatter.encode(self, cell)
                return cell

        return self._encode(cell, thumbnail=thumbnail)

    def _encode(self, image: Image, thumbnail: bool = False) -> str:
        with BytesIO() as buffer:
            if thumbnail:
                image.thumbnail((64, 64))
            image.save(buffer, "jpeg")
            return "data:image/jpeg;base64,{im_base_64}".format(
                im_base_64=base64.b64encode(buffer.getvalue()).decode()
            )

    def html(self, cell: Union["FileCell", Image]) -> str:
        encoded = self.encode(cell, thumbnail=True)
        return f'<img src="{encoded}">'


class CodeFormatter(Formatter):

    LANGUAGES = [
        "js",
        "css",
        "html",
    ]

    def __init__(self, language: str):
        if language not in self.LANGUAGES:
            raise ValueError(
                f"Language {language} not supported."
                f"Supported languages: {self.LANGUAGES}"
            )
        self.language = language

    cell_component = "code"

    def encode(self, cell: str):
        return cell

    def html(self, cell: str):
        return cell

    @property
    def cell_props(self):
        return {"language": self.language}
