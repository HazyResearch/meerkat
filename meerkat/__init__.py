"""Meerkat."""
# flake8: noqa

from json import JSONEncoder


def _default(self, obj):
    # https://stackoverflow.com/a/18561055
    # Monkey patch json module at import time so
    # JSONEncoder.default() checks for a "to_json()"
    # method and uses it to encode objects if it exists
    if isinstance(obj, gui.Store):
        return getattr(obj, "to_json", _default.default)()
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default


from meerkat.logging.utils import initialize_logging

initialize_logging()

import meerkat.interactive as gui
import meerkat.state as GlobalState
from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import Column, column
from meerkat.columns.deferred.audio import AudioColumn
from meerkat.columns.deferred.base import DeferredCell, DeferredColumn
from meerkat.columns.deferred.file import FileCell, FileColumn, FileLoader
from meerkat.columns.deferred.image import ImageColumn
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.scalar import ScalarColumn
from meerkat.columns.scalar.arrow import ArrowScalarColumn
from meerkat.columns.scalar.pandas import PandasScalarColumn
from meerkat.columns.tensor import TensorColumn
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.row import Row
from meerkat.datasets import get
from meerkat.ops.concat import concat
from meerkat.ops.embed import embed
from meerkat.ops.merge import merge
from meerkat.ops.sample import sample
from meerkat.ops.sort import sort
from meerkat.provenance import provenance

from .config import config

# alias for DataFrame for backwards compatibility
DataPanel = DataFrame

# aliases for columns
scalar = ScalarColumn
tensor = TensorColumn
deferred = DeferredColumn
objects = ObjectColumn
files = FileColumn
image = ImageColumn
audio = AudioColumn

# aliases for io
from_csv = DataFrame.from_csv
from_feather = DataFrame.from_feather
from_json = DataFrame.from_json
from_parquet = DataFrame.from_parquet
from_pandas = DataFrame.from_pandas
from_arrow = DataFrame.from_arrow
from_huggingface = DataFrame.from_huggingface




__all__ = [
    "GlobalState",
    "DataFrame",
    "DataPanel",
    "Row",
    "Column",
    "column",
    "ObjectColumn",
    "ScalarColumn",
    "PandasScalarColumn",
    "ArrowScalarColumn",
    "TensorColumn",
    "NumPyTensorColumn",
    "TorchTensorColumn",
    "DeferredColumn",
    "FileColumn",
    "ImageColumn",
    "AudioColumn",
    "AbstractCell",
    "DeferredCell",
    "FileCell",
    "FileLoader",
    "get",
    "concat",
    "merge",
    "embed",
    "sort",
    "sample",
    "provenance",
    "config",
    "gui",
    "from_csv",
    "from_json",
]
