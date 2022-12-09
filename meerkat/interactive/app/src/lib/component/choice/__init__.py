from meerkat.interactive.graph import Store

from ..abstract import Component


class Choice(Component):
    """A choice ref."""

    choices: Store[list]
    value: Store[str]
    gui_type: str = "dropdown"
    title: str = None
