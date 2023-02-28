from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.interactive.event import EventInterface


class OnClickButton(EventInterface):
    pass


class Button(Component):
    title: str
    icon: str = None
    classes: str = "bg-slate-100 py-1 rounded-md flex flex-col hover:bg-slate-200"
    on_click: EndpointProperty[EventInterface] = None
