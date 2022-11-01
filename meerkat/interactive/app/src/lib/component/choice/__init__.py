from typing import List, Union
from meerkat.interactive.graph import Box, Store, make_box, make_store
from meerkat.interactive.edit import EditTarget
import numpy as np
from ..abstract import Component
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from meerkat import AbstractColumn, DataPanel


class Choice(Component):
    """A choice box."""
    name: str = "Choice"
    def __init__(self, value: Union[str, int, float, Store], choices: Union[list, tuple, Store], gui_type: str = "dropdown"):
        super().__init__()
        self.value = make_store(value)
        self.choices = make_store(choices)
        self.gui_type = "dropdown"

        if gui_type not in ["dropdown"]:
            raise ValueError("gui_type must be 'dropdown'")

    @property
    def props(self):
        return {
            "value": self.value.config,
            "choices": self.choices.config,
            "gui_type": self.gui_type,
        }