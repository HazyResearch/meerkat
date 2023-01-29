from pydantic import Field

from meerkat.interactive.graph import Store

from ..abstract import BaseComponent


class MultiSelect(BaseComponent):

    choices: Store[list]
    selected: Store[list] = Field(default_factory=lambda: Store(list()))
    gui_type: str = "multiselect"
    title: str = None
