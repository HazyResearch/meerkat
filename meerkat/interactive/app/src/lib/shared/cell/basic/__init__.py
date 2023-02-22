import math
import textwrap
from typing import Any

import numpy as np
import pandas as pd
from pandas.io.formats.format import format_array

from meerkat.interactive.app.src.lib.component.abstract import BaseComponent
from meerkat.interactive.formatter.base import BaseFormatter


class Scalar(BaseComponent):
    data: Any
    dtype: str = None
    precision: int = 3
    percentage: bool = False


class ScalarFormatter(BaseFormatter):
    component_class: type = Scalar
    data_prop: str = "data"

    def __init__(self, dtype: str = None, precision: int = 3, percentage: bool = False):
        super().__init__(dtype=dtype, precision=precision, percentage=percentage)

    def encode(self, cell: Any):
        # check for native python nan
        if isinstance(cell, float) and math.isnan(cell):
            return "NaN"

        if isinstance(cell, np.generic):
            if pd.isna(cell):
                return "NaN"
            return cell.item()

        if hasattr(cell, "as_py"):
            return cell.as_py()
        return str(cell)

    def html(self, cell: Any):
        cell = self.encode(cell)
        if isinstance(cell, str):
            cell = textwrap.shorten(cell, width=100, placeholder="...")
        return format_array(np.array([cell]), formatter=None)[0]
