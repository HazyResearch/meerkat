from typing import Any

from meerkat.interactive.app.src.lib.component.abstract import Component


class Number(Component):
    data: Any
    dtype: str = "auto"
    precision: int = 3
    percentage: bool = False
    classes: str = ""
