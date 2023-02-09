from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnClickButton(EventInterface):
    pass


class Button(Component):

    title: str
    on_click: Endpoint[EventInterface] = None
