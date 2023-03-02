from typing import Any, Dict

from ..app.src.lib.component.core.raw_html import RawHTML
from .base import BaseFormatter, FormatterGroup
from .icon import IconFormatter


class HTMLFormatter(BaseFormatter):
    component_class: type = RawHTML
    data_prop: str = "html"

    def __init__(self, view: str = "full"):
        self.view = view

    @property
    def props(self) -> dict:
        return dict(view=self.view)

    def encode(self, data: str) -> str:
        if self.view == "icon":
            # don't send data up
            return "icon"
        return data

    def html(self, cell: str) -> str:
        return "HTML"

    def _get_state(self) -> Dict[str, Any]:
        return {
            "view": self.view,
        }

    def _set_state(self, state: Dict[str, Any]):
        self.view = state["view"]


class HTMLFormatterGroup(FormatterGroup):
    def __init__(self):
        super().__init__(
            base=HTMLFormatter(view="full"),
            icon=IconFormatter(name="Globe2"),
            tag=IconFormatter(name="Globe2"),
            thumbnail=HTMLFormatter(view="thumbnail"),
            gallery=HTMLFormatter(view="thumbnail"),
        )
