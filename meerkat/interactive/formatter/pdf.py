from typing import Any, Dict

from ..app.src.lib.component.core.pdf import PDF
from .base import Formatter, FormatterGroup
from .icon import IconFormatter


class PDFFormatter(Formatter):
    component_class: type = PDF
    data_prop: str = "data"

    def encode(self, data: bytes) -> str:
        return data


class PDFFormatterGroup(FormatterGroup):
    def __init__(self):
        super().__init__(
            base=PDFFormatter(),
            icon=IconFormatter(name="FileEarmarkPdf"),
            tag=IconFormatter(name="FileEarmarkPdf"),
            small=IconFormatter(name="FileEarmarkPdf"),
            thumbnail=PDFFormatter(),
            gallery=PDFFormatter(),
        )
