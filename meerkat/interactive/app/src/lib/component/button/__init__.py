
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.graph import Store

from ..abstract import Component


class Button(Component):

    title: Store[str]
    on_click: Endpoint = None
