from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty
from meerkat.interactive.event import EventInterface


class OnRunEditor(EventInterface):
    new_code: str


class Editor(Component):
    code: str = ""
    title: str = "Code Editor"
    on_run: EndpointProperty[OnRunEditor] = None
