from typing import Optional, Union

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnBlurNumberInput(EventInterface):
    value: int


class OnKeyEnterNumberInput(EventInterface):
    value: int


class NumberInput(Component):
    value: Union[int, float]
    placeholder: str = "Enter a number..."
    debounceTimer: int = 150
    classes: str = "grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400"

    on_blur: Optional[Endpoint[OnBlurNumberInput]] = None
    on_keyenter: Optional[Endpoint[OnKeyEnterNumberInput]] = None

    def __init__(
        self,
        value: Union[int, float],
        *,
        placeholder: str = "Enter a number...",
        debounceTimer: int = 150,
        classes: str = "grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400",
        on_blur: Optional[Endpoint[OnBlurNumberInput]] = None,
        on_keyenter: Optional[Endpoint[OnKeyEnterNumberInput]] = None,
    ):
        """An input field that can be used to get a numeric input from the
        user.

        Attributes:
            value: The value in the input field.
            placeholder: The placeholder text.
            debounce_timer: The debounce timer in milliseconds.
            on_blur: The endpoint to call when the input field loses focus.
            on_enter: The endpoint to call when the user presses enter.
        """

        super().__init__(
            value=value,
            placeholder=placeholder,
            debounceTimer=debounceTimer,
            classes=classes,
            on_blur=on_blur,
            on_keyenter=on_keyenter,
        )
