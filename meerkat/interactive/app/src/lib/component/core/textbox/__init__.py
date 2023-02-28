from typing import Optional, Union

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnBlurTextbox(EventInterface):
    text: Union[str, int, float]


class OnKeyEnterTextbox(EventInterface):
    text: Union[str, int, float]


class Textbox(Component):
    text: str = ""
    placeholder: str = "Write some text..."
    debounce_timer: int = 150
    classes: str = "grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400"

    on_blur: Optional[Endpoint[OnBlurTextbox]] = None
    on_keyenter: Optional[Endpoint[OnKeyEnterTextbox]] = None

    def __init__(
        self,
        text: str = "",
        *,
        placeholder: str = "Write some text...",
        debounce_timer: int = 150,
        classes: str = "grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400",
        on_blur: Optional[Endpoint[OnBlurTextbox]] = None,
        on_keyenter: Optional[Endpoint[OnKeyEnterTextbox]] = None,
    ):
        """A textbox that can be used to get user input.

        Attributes:
            text: The text in the textbox.
            placeholder: The placeholder text.
            debounce_timer: The debounce timer in milliseconds.
            on_blur: The endpoint to call when the textbox loses focus.
            on_enter: The endpoint to call when the user presses enter.
        """

        super().__init__(
            text=text,
            placeholder=placeholder,
            debounce_timer=debounce_timer,
            classes=classes,
            on_blur=on_blur,
            on_keyenter=on_keyenter,
        )
