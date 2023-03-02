from typing import List, Literal, Optional

from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnChangeRadio(EventInterface):
    index: int


class Radio(Slottable, Component):
    """A single radio button.

    If you just want a basic group of radio buttons, use the
    RadioGroup component instead. Use this component only if
    you want to customize the layout of a group of radio buttons.

    For more advanced use cases, we recommend either using the
    basic HTML radio button element and styling it yourself with
    Tailwind, or using the Flowbite Radio component.

    Args:
        name (str): The name of this radio button. Assign the same
            name to multiple radio buttons to group them together.
        value (str): The value associated with this radio button.
        disabled (bool): Whether this radio button is disabled.
        color (Literal['blue', 'red', 'green', 'purple', 'teal', \
            'yellow', 'orange']): The color of this radio button.
        classes (str): The Tailwind classes to apply to the component.

        on_change: The `Endpoint` to call when this radio button is selected. \
            It must have the following signature:

            `(index: int)`

            with
                index (int): The index of the selected radio button.
    """

    name: str
    value: str = ""
    disabled: bool = False
    color: Literal[
        "blue", "red", "green", "purple", "teal", "yellow", "orange"
    ] = "purple"
    classes: str = "bg-violet-50 p-2 rounded-lg w-fit"

    on_change: Optional[Endpoint[OnChangeRadio]] = None


class OnChangeRadioGroup(EventInterface):
    index: int


class RadioGroup(Component):
    """A basic group of radio buttons.

    Args:
        values (List[str]): The values associated with each radio button. \
            The number of radio buttons will be the length of this list.
        selected (Optional[int]): The index of the selected radio button (0-indexed). \
            If None, no radio button will be preselected by default.
        disabled (bool): Whether this radio group is disabled. If True, all \
            radio buttons will be disabled and the user will not be able to \
            select any of them.
        horizontal (bool): Whether to display the radio buttons horizontally. \
            Defaults to True.
        color (Literal['blue', 'red', 'green', 'purple', 'teal', 'yellow', \
            'orange']): The color of the radio buttons.
        classes (str): The Tailwind classes to apply to the component.

        on_change: The `Endpoint` to call when the selected radio button changes. \
            It must have the following signature:

            `(index: int)`

            with
                index (int): The index of the selected radio button.
    """

    values: List[str]
    selected: Optional[int] = None
    disabled: bool = False
    horizontal: bool = True
    color: Literal[
        "blue", "red", "green", "purple", "teal", "yellow", "orange"
    ] = "purple"
    classes: str = "bg-violet-50 p-2 rounded-lg w-fit"

    on_change: Optional[Endpoint[OnChangeRadioGroup]] = None
