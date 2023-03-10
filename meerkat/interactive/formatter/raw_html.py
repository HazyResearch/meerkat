from ..app.src.lib.component.core.raw_html import RawHTML
from .base import Formatter, FormatterGroup
from .icon import IconFormatter


class HTMLFormatter(Formatter):
    component_class: type = RawHTML
    data_prop: str = "html"

    def encode(self, data: str) -> str:
        return data

    def html(self, cell: str) -> str:
        return "HTML"


class HTMLFormatterGroup(FormatterGroup):
    def __init__(self, sanitize: bool = True, classes: str = ""):
        super().__init__(
            base=HTMLFormatter(view="full", sanitize=sanitize, classes=classes),
            icon=IconFormatter(name="Globe2"),
            tag=IconFormatter(name="Globe2"),
            thumbnail=HTMLFormatter(
                view="thumbnail", sanitize=sanitize, classes=classes
            ),
            gallery=HTMLFormatter(view="thumbnail", sanitize=sanitize, classes=classes),
        )
