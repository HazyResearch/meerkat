from typing import Optional

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnBlurTextbox(EventInterface):
    text: int


class Textbox(Component):
    text: str = ""
    placeholder: str = "Write some text..."
    debounce_timer: int = 150

    on_blur: Optional[Endpoint[OnBlurTextbox]] = None

    def __init__(
        self,
        text: str = "",
        placeholder: str = "Write some text...",
        debounce_timer: int = 150,
        on_blur: Optional[Endpoint[OnBlurTextbox]] = None,
    ):
        """A textbox that can be used to get user input.

        Attributes:
            text: The text in the textbox.
            placeholder: The placeholder text.
            debounce_timer: The debounce timer in milliseconds.
            on_blur: The endpoint to call when the textbox loses focus.
        """

        super().__init__(
            text=text,
            placeholder=placeholder,
            debounce_timer=debounce_timer,
            on_blur=on_blur,
        )
