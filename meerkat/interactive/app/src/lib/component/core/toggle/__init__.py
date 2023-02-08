from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnChangeToggle(EventInterface):
    value: bool


class Toggle(Component):
    value: bool = False

    on_change: Endpoint[OnChangeToggle] = None
