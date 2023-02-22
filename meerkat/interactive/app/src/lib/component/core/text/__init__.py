import textwrap
from typing import Any, Dict

import numpy as np
from pandas.io.formats.format import format_array

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.formatter.base import BaseFormatter, FormatterGroup


class Text(Component):
    data: str
    classes: str = ""
    editable: bool = False

    def __init__(
        self,
        data: str,
        *,
        classes: str = "",
        editable: bool = False,
    ):
        """Display text.

        Args:
            data: The text to display.
            editable: Whether the text is editable.
        """
        super().__init__(
            classes=classes,
            data=data,
            editable=editable,
        )


class TextFormatter(BaseFormatter):
    """Formatter for Text component.

    Args:
        component_class: The component class to format.
        data_prop: The property name of the data to format.
        variants: The variants of the component.
    """

    component_class: type = Text
    data_prop: str = "data"

    def __init__(self, classes: str = ""):
        self.classes = classes

    def encode(self, cell: Any):
        return str(cell)

    @property
    def props(self) -> Dict[str, Any]:
        return {"classes": self.classes}

    def _get_state(self) -> Dict[str, Any]:
        return {
            "classes": self.classes,
        }

    def _set_state(self, state: Dict[str, Any]):
        self.classes = state["classes"]

    def html(self, cell: Any):
        cell = self.encode(cell)
        if isinstance(cell, str):
            cell = textwrap.shorten(cell, width=100, placeholder="...")
        return format_array(np.array([cell]), formatter=None)[0]


class TextFormatterGroup(FormatterGroup):
    def __init__(self, classes: str = ""):
        super().__init__(
            base=TextFormatter(classes=classes),
            tiny=TextFormatter(classes=classes),
            small=TextFormatter(classes=classes),
            thumbnail=TextFormatter(classes=classes),
            gallery=TextFormatter(
                classes="aspect-video h-full p-2" + classes
            ),
            tag=TextFormatter(classes="whitespace-nowrap text-ellipsis overflow-hidden text-right " + classes)
        )
# "whitespace-nowrap text-ellipsis overflow-hidden text-right "
