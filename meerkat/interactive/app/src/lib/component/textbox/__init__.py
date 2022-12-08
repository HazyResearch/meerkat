from meerkat.interactive.graph import Store

from ..abstract import Component


class Textbox(Component):

    text: Store[str]
    title: str = ""
