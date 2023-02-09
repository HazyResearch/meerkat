from typing import Literal, Optional

from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.interactive.endpoint import Endpoint


class Checkbox(Slottable, Component):
    checked: bool = False
    disabled: bool = False
    color: Literal['blue', 'red', 'green', 'purple', 'teal', 'yellow', 'orange'] = "purple"

    classes: str = "bg-violet-50 p-2 rounded-lg"

    on_change: Optional[Endpoint] = None
