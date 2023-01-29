from meerkat.interactive import Component
from meerkat.interactive.endpoint import Endpoint


class Button(Component):

    title: str
    on_click: Endpoint = None
