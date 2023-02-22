from meerkat.interactive.app.src.lib.component.abstract import Component


class Textbox(Component):
    text: str = ""
    placeholder: str = "Write some text..."
    debounce_timer: int = 150

    def __init__(
        self,
        text: str = "",
        placeholder: str = "Write some text...",
        debounce_timer: int = 150,
    ):
        """A textbox that can be used to get user input.

        Attributes:
            text: The text in the textbox.
            placeholder: The placeholder text.
            debounce_timer: The debounce timer in milliseconds.
        """

        super().__init__(
            text=text, placeholder=placeholder, debounce_timer=debounce_timer
        )
