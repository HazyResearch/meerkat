from typing import Optional

from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnTextEvent(EventInterface):
    text: int


class Textbox(Component):
    text: str = ""
    placeholder: str = "Write some text..."
    debounce_timer: int = 150
    classes: str = "grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400"

    on_blur: Optional[Endpoint[OnTextEvent]] = None
    on_keyenter: Optional[Endpoint[OnTextEvent]] = None

    def __init__(
        self,
        text: str = "",
        *,
        placeholder: str = "Write some text...",
        debounce_timer: int = 150,
        classes: str = "grow h-10 px-3 rounded-md shadow-md my-1 border-gray-400",
        on_blur: Optional[Endpoint[OnTextEvent]] = None,
        on_keyenter: Optional[Endpoint[OnTextEvent]] = None,
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
