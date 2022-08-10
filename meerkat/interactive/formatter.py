"""Desiderata for formatters:

1.
"""
import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Any, Union

import numpy as np
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

    def encode(self, cell: Any):
        return format_array(np.array([cell]), formatter=None)[0]

    def html(self, cell: Any):
        cell = self.encode(cell)
        return cell


class PILImageFormatter(Formatter):

    cell_component = "image"

    def encode(self, cell: Union["FileCell", Image]) -> str:
        from meerkat.columns.file_column import FileCell

        if isinstance(cell, FileCell):
            cell = cell.get()
        return self._encode(cell)

    def _encode(self, image: Image) -> str:
        with BytesIO() as buffer:
            image.save(buffer, "jpeg")
            return "data:image/jpeg;base64,{im_base_64}".format(
                im_base_64=base64.b64encode(buffer.getvalue()).decode()
            )

    def html(self, cell: Union["FileCell", Image]) -> str:
        encoded = self.encode(cell)
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
