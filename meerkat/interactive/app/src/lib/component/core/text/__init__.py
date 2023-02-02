import textwrap
from typing import Any

import numpy as np
from pandas.io.formats.format import format_array

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.formatter.base import Formatter, Variant


class Text(Component):

    data: str
    view: str = "line"
    editable: bool = False


class TextFormatter(Formatter):

    component_class: type = Text
    data_prop: str = "data"
    variants: dict = {
        "gallery": Variant(
            props={"view": "wrapped"},
            encode_kwargs={},
        ),
        "key_value": Variant(
            props={"view": "line"},
            encode_kwargs={},
        ),
        "full_screen": Variant(
            props={"view": "wrapped"},
            encode_kwargs={},
        ),
    }


    def _encode(self, cell: Any):
        return str(cell)

    def html(self, cell: Any):
        cell = self.encode(cell)
        if isinstance(cell, str):
            cell = textwrap.shorten(cell, width=100, placeholder="...")
        return format_array(np.array([cell]), formatter=None)[0]
