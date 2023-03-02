from .base import Formatter, deferred_formatter_group
from .boolean import BooleanFormatter, BooleanFormatterGroup
from .code import CodeFormatter, CodeFormatterGroup
from .image import ImageFormatter, ImageFormatterGroup
from .number import NumberFormatter, NumberFormatterGroup
from .pdf import PDFFormatter, PDFFormatterGroup
from .raw_html import HTMLFormatter, HTMLFormatterGroup
from .tensor import TensorFormatter, TensorFormatterGroup
from .text import TextFormatter, TextFormatterGroup

__all__ = [
    "Formatter",
    "deferred_formatter_group",
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
    "PDFFormatter",
    "PDFFormatterGroup",
    "TensorFormatter",
    "TensorFormatterGroup",
    "BooleanFormatter",
    "BooleanFormatterGroup",
]

# backwards compatibility
class DeprecatedFormatter:  # noqa: E302
    pass


ObjectFormatter = DeprecatedFormatter
BasicFormatter = DeprecatedFormatter
NumpyArrayFormatter = DeprecatedFormatter
# TensorFormatter = DeprecatedFormatter
WebsiteFormatter = DeprecatedFormatter
# CodeFormatter = DeprecatedFormatter
PILImageFormatter = DeprecatedFormatter
