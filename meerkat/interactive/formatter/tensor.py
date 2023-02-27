from typing import Any

from meerkat.interactive.formatter.base import FormatterGroup
from meerkat.interactive.formatter.icon import IconFormatter
from meerkat.interactive.formatter.text import TextFormatter


class TensorFormatter(TextFormatter):
    """Formatter for an embedding."""

    def encode(self, cell: Any):
        return f"Tensor (shape: {cell.shape})"


class TensorFormatterGroup(FormatterGroup):
    def __init__(self):
        super().__init__(
            base=TensorFormatter(),
            icon=IconFormatter(name="BoxFill"),
        )
