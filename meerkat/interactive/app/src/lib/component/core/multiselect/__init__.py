from pydantic import Field

from meerkat.interactive.app.src.lib.component.abstract import BaseComponent
from meerkat.interactive.graph import Store


class MultiSelect(BaseComponent):
    choices: Store[list]
    selected: Store[list] = Field(default_factory=lambda: Store(list()))
    gui_type: str = "multiselect"
    title: str = None
