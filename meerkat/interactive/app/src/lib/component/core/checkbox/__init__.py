from typing import Optional

from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.interactive.endpoint import Endpoint

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class Checkbox(Slottable, Component):
    checked: bool = False
    disabled: bool = False
    color: Literal[
        "blue",
        "red",
        "green",
        "purple",
        "teal",
        "yellow",
        "orange",
    ] = "purple"

    classes: str = ""

    on_change: Optional[Endpoint] = None
