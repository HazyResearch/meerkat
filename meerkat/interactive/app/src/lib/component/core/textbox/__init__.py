from meerkat.interactive.app.src.lib.component.abstract import Component


class Textbox(Component):
    """
    A textbox that can be used to get user input.

    Attributes:
        text: The text in the textbox.
    """

    text: str = ""
