"""Meerkat."""
# flake8: noqa

from meerkat.logging.utils import initialize_logging

initialize_logging()

import meerkat.interactive as gui
import meerkat.state as GlobalState
from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import Column
from meerkat.columns.deferred.audio import AudioColumn
from meerkat.columns.deferred.file import FileCell, FileColumn, FileLoader
from meerkat.columns.deferred.image import ImageColumn
from meerkat.columns.deferred.base import DeferredCell, DeferredColumn
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.scalar import ScalarColumn
from meerkat.columns.scalar.pandas import PandasScalarColumn
from meerkat.columns.scalar.arrow import ArrowScalarColumn
from meerkat.columns.tensor import TensorColumn
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.datasets import get

from meerkat.ops.concat import concat
from meerkat.ops.embed import embed
from meerkat.ops.match import match
from meerkat.ops.merge import merge
from meerkat.ops.sample import sample
from meerkat.ops.sort import sort
from meerkat.provenance import provenance

from .config import config

# alias for DataFrame for backwards compatibility
DataPanel = DataFrame

__all__ = [
    "GlobalState",
    "DataFrame",
    "DataPanel",
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
    "VideoColumn",
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
]
