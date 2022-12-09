from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.graph import Store


class Code(Component):

    data: Store[str]
    language: str = "python"
