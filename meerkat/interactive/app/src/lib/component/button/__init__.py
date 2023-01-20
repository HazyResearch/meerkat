from meerkat.interactive.endpoint import Endpoint

from ..abstract import AutoComponent


class Button(AutoComponent):

    title: str
    on_click: Endpoint = None
