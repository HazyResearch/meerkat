from meerkat.interactive.graph import make_store

from ..abstract import Component


class Textbox(Component):

    name = "Textbox"

    def __init__(
        self,
        title: str = "",
    ):
        super().__init__()
        self.text = make_store("")
        self.title = title

    @property
    def props(self):
        return {
            "text": self.text.config,
            "title": self.title,
        }
