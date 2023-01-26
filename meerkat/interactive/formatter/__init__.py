

from .base import Formatter
from ..app.src.lib.component.scalar import ScalarFormatter


# # backwards compatibility
ObjectFormatter = ScalarFormatter
BasicFormatter = ScalarFormatter
NumpyArrayFormatter = ScalarFormatter
TensorFormatter = ScalarFormatter
WebsiteFormatter = ScalarFormatter
CodeFormatter = ScalarFormatter