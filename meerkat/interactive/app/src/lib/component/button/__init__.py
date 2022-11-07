from meerkat.interactive.graph import make_store

from ..abstract import Component


class Button(Component):

    name = "Button"

    def __init__(self, title: str = "Button"):
        super().__init__()
        self.value = make_store(0)
        self.title = title

    @property
    def props(self):
        return {
            "value": self.value.config,
            "title": self.title,
        }
