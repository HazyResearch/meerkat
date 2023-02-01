"""Meerkat."""
# flake8: noqa

from json import JSONEncoder


def _default(self, obj):
    # https://stackoverflow.com/a/18561055
    # Monkey patch json module at import time so
    # JSONEncoder.default() checks for a "to_json()"
    # method and uses it to encode objects if it exists
    # Note: this may not have been for FastAPI, but another library
    if isinstance(obj, gui.Store):
        return getattr(obj, "to_json", _default.default)()
    return getattr(obj.__class__, "to_json", _default.default)(obj)


_default.default = JSONEncoder().default
JSONEncoder.default = _default

from meerkat.logging.utils import initialize_logging

initialize_logging()

import meerkat.interactive as gui

# This statement needs to be after the imports above.
import meerkat.interactive.svelte
import meerkat.state as GlobalState
from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import Column, column
from meerkat.columns.deferred.audio import AudioColumn
from meerkat.columns.deferred.base import DeferredCell, DeferredColumn
from meerkat.columns.deferred.file import FileCell, FileColumn, FileLoader
from meerkat.columns.deferred.image import ImageColumn, image
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.scalar import ScalarColumn
from meerkat.columns.scalar.arrow import ArrowScalarColumn
from meerkat.columns.scalar.pandas import PandasScalarColumn
from meerkat.columns.tensor import TensorColumn
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.datasets import get
from meerkat.ops.aggregate.aggregate import aggregate
from meerkat.ops.concat import concat
from meerkat.ops.cond import cand, cnot, cor, to_bool
from meerkat.ops.embed import embed
from meerkat.ops.map import defer, map
from meerkat.ops.merge import merge
from meerkat.ops.sample import sample
from meerkat.ops.search import search
from meerkat.ops.sliceby.clusterby import clusterby
from meerkat.ops.sliceby.explainby import explainby
from meerkat.ops.sliceby.groupby import groupby
from meerkat.ops.sort import sort
from meerkat.provenance import provenance
from meerkat.row import Row
from meerkat.tools.utils import classproperty

from .config import config

# alias for DataFrame for backwards compatibility
DataPanel = DataFrame

# aliases for columns
scalar = ScalarColumn
tensor = TensorColumn
deferred = DeferredColumn
objects = ObjectColumn
files = FileColumn
audio = AudioColumn

# aliases for io
from_csv = DataFrame.from_csv
from_feather = DataFrame.from_feather
from_json = DataFrame.from_json
from_parquet = DataFrame.from_parquet
from_pandas = DataFrame.from_pandas
from_arrow = DataFrame.from_arrow
from_huggingface = DataFrame.from_huggingface
read = DataFrame.read


__all__ = [
    "DataFrame",
    "Row",
    # <<<< Columns >>>>
    "column",
    "Column",
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
    # <<<< Operations >>>>
    "map",
    "defer",
    "concat",
    "merge",
    "embed",
    "sort",
    "sample",
    "groupby",
    "clusterby",
    "explainby",
    "aggregate",
    "cand",
    "cor",
    "cnot",
    "to_bool",
    # <<<< I/O >>>>
    "from_csv",
    "from_json",
    "from_parquet",
    "from_feather",
    "from_pandas",
    "from_arrow",
    "from_huggingface",
    "read",
    # <<<< Misc >>>>
    "provenance",
    "config",
    "gui",
    "FileLoader",
    "get",
    "GlobalState",
    # <<<< Aliases >>>>
    "DataPanel",
    "scalar",
    "tensor",
    "deferred",
    "objects",
    "files",
    "image",
    "audio",
    # <<<< Utilities >>>>
    "classproperty",
]
