from meerkat.interactive.graph import Store

from ..abstract import BaseComponent


class CodeDisplay(BaseComponent):

    data: Store[str]
    language: str = "python"
