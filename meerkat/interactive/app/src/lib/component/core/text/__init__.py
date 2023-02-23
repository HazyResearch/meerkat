import textwrap
from typing import Any, Dict

import numpy as np
from pandas.io.formats.format import format_array

from meerkat.interactive.app.src.lib.component.abstract import Component


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



# "whitespace-nowrap text-ellipsis overflow-hidden text-right "
