from ..app.src.lib.component.core.number import NumberFormatter, NumberFormatterGroup
from ..app.src.lib.component.core.raw_html import HTMLFormatter, HTMLFormatterGroup
from ..app.src.lib.component.core.text import TextFormatter, TextFormatterGroup
from .code import CodeFormatter, CodeFormatterGroup
from .image import ImageFormatter, ImageFormatterGroup

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
    "CodeFormatter",
    "CodeFormatterGroup",
]

# # backwards compatibility
class DeprecatedFormatter:
    pass


ObjectFormatter = DeprecatedFormatter
BasicFormatter = DeprecatedFormatter
NumpyArrayFormatter = DeprecatedFormatter
TensorFormatter = DeprecatedFormatter
WebsiteFormatter = DeprecatedFormatter
# CodeFormatter = DeprecatedFormatter
PILImageFormatter = DeprecatedFormatter
