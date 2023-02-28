from ..app.src.lib.component.core.pdf import PDF
from .base import Formatter, FormatterGroup
from .icon import IconFormatter


class PDFFormatter(Formatter):
    component_class: type = PDF
    data_prop: str = "data"

    def encode(self, data: bytes) -> str:
        return data.decode("latin-1")


class PDFFormatterGroup(FormatterGroup):
    def __init__(self):
        super().__init__(
            base=PDFFormatter(),
            icon=IconFormatter(name="FileEarmarkPdf"),
            tag=IconFormatter(name="FileEarmarkPdf"),
            small=IconFormatter(name="FileEarmarkPdf"),
            thumbnail=PDFFormatter(),
            full=PDFFormatter(classes="max-w-full max-h-full"),
            gallery=PDFFormatter(classes="h-full"),
        )
