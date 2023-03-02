import math
import textwrap
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas.io.formats.format import format_array

from meerkat.interactive.app.src.lib.component.core.number import Number
from meerkat.interactive.formatter.base import BaseFormatter, FormatterGroup
from meerkat.interactive.formatter.icon import IconFormatter


class NumberFormatter(BaseFormatter):
    component_class: type = Number
    data_prop: str = "data"

    def __init__(
        self,
        dtype: str = "auto",
        precision: int = 3,
        percentage: bool = False,
        classes: str = "",
    ):
        self.dtype = dtype
        self.precision = precision
        self.percentage = percentage
        self.classes = classes

    @property
    def props(self):
        return {
            "dtype": self.dtype,
            "precision": self.precision,
            "percentage": self.percentage,
            "classes": self.classes,
        }

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

    def _get_state(self) -> Dict[str, Any]:
        return {
            "dtype": self.dtype,
            "precision": self.precision,
            "percentage": self.percentage,
            "classes": self.classes,
        }

    def _set_state(self, state: Dict[str, Any]):
        self.dtype = state["dtype"]
        self.precision = state["precision"]
        self.percentage = state["percentage"]
        self.classes = state["classes"]


class NumberFormatterGroup(FormatterGroup):
    def __init__(self, **kwargs):
        super().__init__(
            base=NumberFormatter(**kwargs), icon=IconFormatter(name="Hash")
        )
