from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint
from meerkat.interactive.event import EventInterface


class OnClickButton(EventInterface):
    pass


class Button(Component):
    title: str
    classes: str = (
        "bg-slate-100 py-3 rounded-lg drop-shadow-md flex flex-col hover:bg-slate-200"
    )
    on_click: Endpoint[EventInterface] = None
