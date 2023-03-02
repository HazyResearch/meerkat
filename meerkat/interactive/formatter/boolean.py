import numpy as np
import pandas as pd

from ..app.src.lib.component.core.checkbox import Checkbox
from .base import Formatter, FormatterGroup
from .icon import IconFormatter


class BooleanFormatter(Formatter):
    component_class: type = Checkbox
    data_prop: str = "checked"

    def encode(self, cell: bool) -> bool:
        if isinstance(cell, np.generic):
            if pd.isna(cell):
                return "NaN"
            return cell.item()

        if hasattr(cell, "as_py"):
            return cell.as_py()
        return str(cell)


class BooleanFormatterGroup(FormatterGroup):
    def __init__(self):
        super().__init__(
            base=BooleanFormatter(),
            icon=IconFormatter(name="CheckSquare"),
        )
