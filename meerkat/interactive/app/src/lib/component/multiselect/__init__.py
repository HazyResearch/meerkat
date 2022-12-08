from meerkat.interactive.graph import Store

from ..abstract import Component


class MultiSelect(Component):

    choices: Store[list]
    selected: Store[list]
    gui_type: str = "multiselect"
    title: str = None
