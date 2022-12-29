from typing import Any

from meerkat.interactive.app.src.lib.component.abstract import Component


class Text(Component):

    data: Any
    dtype: str = None
    precision: int = 3
    percentage: bool = False
