

from .base import Formatter
from ..app.src.lib.component.core.image import ImageFormatter, ImageFormatterGroup
from ..app.src.lib.component.core.text import TextFormatter, TextFormatterGroup
from ..app.src.lib.component.core.number import NumberFormatter, NumberFormatterGroup
from ..app.src.lib.component.core.raw_html import HTMLFormatter, HTMLFormatterGroup


__all__ = [
    "Formatter",
    "ImageFormatter",
    "ImageFormatterGroup",
    "TextFormatter",
    "TextFormatterGroup",
    "NumberFormatter",
    "NumberFormatterGroup",
    "HTMLFormatter",
    "HTMLFormatterGroup",
]

# # backwards compatibility
class DeprecatedFormatter:
    pass


ObjectFormatter = DeprecatedFormatter
BasicFormatter = DeprecatedFormatter
NumpyArrayFormatter = DeprecatedFormatter
TensorFormatter = DeprecatedFormatter
WebsiteFormatter = DeprecatedFormatter
CodeFormatter = DeprecatedFormatter
PILImageFormatter = DeprecatedFormatter
