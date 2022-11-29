from dataclasses import dataclass
from ..abstract import Component

from meerkat.interactive.graph import Store
from meerkat.interactive.endpoint import Endpoint


class Button(Component):

    title: Store[str]
    on_click: Endpoint = None
