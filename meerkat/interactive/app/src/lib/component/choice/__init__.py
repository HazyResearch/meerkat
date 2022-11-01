from typing import TYPE_CHECKING, Union

from meerkat.interactive.graph import Store, make_store

from ..abstract import Component

if TYPE_CHECKING:
    pass


class Choice(Component):
    """A choice box."""

    name: str = "Choice"

    def __init__(
        self,
        value: Union[str, int, float, Store],
        choices: Union[list, tuple, Store],
        gui_type: str = "dropdown",
    ):
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
