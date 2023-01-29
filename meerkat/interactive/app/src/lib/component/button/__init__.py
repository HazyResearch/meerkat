from meerkat.interactive import AutoComponent
from meerkat.interactive.endpoint import Endpoint


class Button(AutoComponent):

    title: str
    on_click: Endpoint = None
