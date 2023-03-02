from ..app.src.lib.component.core.code import Code
from .base import Formatter, FormatterGroup
from .icon import IconFormatter


class CodeFormatter(Formatter):
    component_class = Code
    data_prop: str = "body"


class CodeFormatterGroup(FormatterGroup):
    def __init__(self):
        super().__init__(
            base=CodeFormatter(),
            icon=IconFormatter(name="CodeSquare"),
            tiny=IconFormatter(name="CodeSquare"),
            tag=IconFormatter(name="CodeSquare"),
            thumbnail=CodeFormatter(),
            gallery=CodeFormatter(classes="h-full aspect-video"),
            full=CodeFormatter(classes="h-full w-ful rounded-lg"),
        )
