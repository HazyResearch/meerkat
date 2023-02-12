from meerkat.interactive.app.src.lib.component.abstract import Component


class Textbox(Component):
    text: str = ""

    def __init__(self, text: str = ""):
        """A textbox that can be used to get user input.

        Attributes:
            text: The text in the textbox.
        """

        super().__init__(text=text)
