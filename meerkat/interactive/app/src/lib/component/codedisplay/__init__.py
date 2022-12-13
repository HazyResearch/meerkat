from meerkat.interactive.graph import Store

from ..abstract import Component


class CodeDisplay(Component):

    data: Store[str]
    language: str = "python"
