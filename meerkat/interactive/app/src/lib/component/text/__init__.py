from typing import Any
import textwrap

import numpy as np
from pandas.io.formats.format import format_array

from ..abstract import AutoComponent
from meerkat.interactive.formatter.base import Formatter


class Text(AutoComponent):

    data: str


class TextFormatter(Formatter):

    component_class: type = Text
    data_prop: str = "data"

    def _encode(self, cell: Any):
        return str(cell)

    def html(self, cell: Any):
        cell = self.encode(cell)
        if isinstance(cell, str):
            cell = textwrap.shorten(cell, width=100, placeholder="...")
        return format_array(np.array([cell]), formatter=None)[0]
