from typing import Any

from .base import FormatterGroup, Formatter
from .icon import IconFormatter
from meerkat.interactive.app.src.lib.component.core.tensor import Tensor


class TensorFormatter(Formatter):
    """Formatter for an embedding."""
    component_class: type = Tensor

    def encode(self, cell: Any):
        return {
            "data": cell.tolist(),
            "shape": list(cell.shape),
            "dtype": str(cell.dtype),
        }


class TensorFormatterGroup(FormatterGroup):
    def __init__(self, dtype: str = None):
        super().__init__(
            base=TensorFormatter(dtype=dtype),
            icon=IconFormatter(name="BoxFill"),
        )
