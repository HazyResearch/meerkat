from meerkat.interactive.app.src.lib.component.abstract import Component, Slottable
from meerkat.interactive.endpoint import Endpoint


class button(Slottable, Component):

    classes: str = ""

    on_blur: Endpoint = None
    on_click: Endpoint = None
    on_contextmenu: Endpoint = None
    on_dblclick: Endpoint = None
    on_focus: Endpoint = None
    on_mousedown: Endpoint = None
    on_mouseenter: Endpoint = None
    on_mouseleave: Endpoint = None
    on_mousemove: Endpoint = None
    on_mouseout: Endpoint = None
    on_mouseover: Endpoint = None
    on_mouseup: Endpoint = None


__all__ = [
    "button",
]
