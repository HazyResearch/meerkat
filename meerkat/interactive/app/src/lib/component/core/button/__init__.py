from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint


class Button(Component):

    title: str
    on_click: Endpoint = None
