from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.interactive.event import EventInterface


class OnRunEditor(EventInterface):
    new_code: str


class Editor(Component):
    code: str = ""
    on_run: EndpointProperty[OnRunEditor] = None
