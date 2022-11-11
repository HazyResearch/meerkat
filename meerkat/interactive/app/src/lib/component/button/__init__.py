from meerkat.interactive.graph import Endpoint, make_endpoint

from ..abstract import Component


class Button(Component):

    name = "Button"

    def __init__(
        self,
        title: str = "Button",
        on_click: Endpoint = None,
    ):
        super().__init__()
        self.title = title
        self.on_click = make_endpoint(on_click)

    @property
    def props(self):
        return {
            "title": self.title,
            "on_click": self.on_click.config,
        }
