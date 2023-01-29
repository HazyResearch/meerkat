from .base import Formatter
from ..app.src.lib.component.scalar import ScalarFormatter


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
