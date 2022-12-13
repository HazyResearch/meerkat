from meerkat.interactive.graph import Store, store_field

from ..abstract import Component


class Textbox(Component):

    text: Store[str] = store_field("")
    title: str = ""
