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
import meerkat.interactive.formatter as format
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
from meerkat.ops.complete import complete
from meerkat.ops.concat import concat
from meerkat.ops.cond import (
    _abs,
    _all,
    _any,
    _bool,
    _complex,
    _dict,
    _float,
    _hex,
    _int,
    _len,
    _list,
    _max,
    _min,
    _oct,
    _range,
    _set,
    _slice,
    _str,
    _sum,
    _tuple,
    cand,
    cnot,
    cor,
)
from meerkat.ops.embed import embed
from meerkat.ops.map import defer, map
from meerkat.ops.merge import merge
from meerkat.ops.sample import sample
from meerkat.ops.search import search
from meerkat.ops.shuffle import shuffle
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

# aliases for meerkat magic method invokers.
all = _all
any = _any
len = _len
int = _int
float = _float
complex = _complex
hex = _hex
oct = _oct
bool = _bool
str = _str
list = _list
tuple = _tuple
sum = _sum
dict = _dict
set = _set
range = _range
abs = _abs
min = _min
max = _max
slice = _slice

# These statements needs to be after the imports above.
# Do not move them.
import meerkat.interactive.svelte
from meerkat.interactive import Store, endpoint, mark, reactive, unmarked
from meerkat.interactive.formatter.base import (
    BaseFormatter,
    FormatterGroup,
    FormatterPlaceholder,
)
from meerkat.interactive.graph.magic import magic

__all__ = [
    "DataFrame",
    "Row",
    "reactive",
    "unmarked",
    "Store",
    "mark",
    "endpoint",
    "magic",
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
    "complete",
    "merge",
    "embed",
    "sort",
    "sample",
    "shuffle",
    "groupby",
    "clusterby",
    "explainby",
    "aggregate",
    "cand",
    "cor",
    "cnot",
    "all",
    "any",
    "bool",
    "len",
    "int",
    "float",
    "complex",
    "hex",
    "oct",
    "str",
    "list",
    "tuple",
    "slice",
    "sum",
    "dict",
    "set",
    "range",
    "abs",
    # <<<< I/O >>>>
    "from_csv",
    "from_json",
    "from_parquet",
    "from_feather",
    "from_pandas",
    "from_arrow",
    "from_huggingface",
    "read",
    # <<<< Formatters >>>>
    "BaseFormatter",
    "FormatterGroup",
    "FormatterPlaceholder",
    # <<<< Misc >>>>
    "provenance",
    "config",
    "gui",
    "format",
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
