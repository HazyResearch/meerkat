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
from meerkat.cells.volume import MedicalVolumeCell
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.arrow_column import ArrowArrayColumn
from meerkat.columns.audio_column import AudioColumn
from meerkat.columns.cell_column import CellColumn
from meerkat.columns.file_column import FileCell, FileColumn, FileLoader
from meerkat.columns.image_column import ImageColumn
from meerkat.columns.lambda_column import LambdaCell, LambdaColumn
from meerkat.columns.list_column import ListColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.spacy_column import SpacyColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.columns.volume_column import MedicalVolumeColumn
from meerkat.dataframe import DataFrame
from meerkat.datasets import get

# TODO: backwards compat remove before release
from meerkat.interactive.startup import start as interactive_mode
from meerkat.ops.concat import concat
from meerkat.ops.embed import embed
from meerkat.ops.match import match
from meerkat.ops.merge import merge
from meerkat.ops.sample import sample
from meerkat.ops.sort import sort
from meerkat.provenance import provenance

from .config import config

# aliases for core column types
ArrayColumn = NumpyArrayColumn
SeriesColumn = PandasSeriesColumn
ArrowColumn = ArrowArrayColumn

# alias for DataFrame for backwards compatibility
DataPanel = DataFrame

__all__ = [
    "GlobalState",
    "DataFrame",
    "DataPanel",
    "AbstractColumn",
    "LambdaColumn",
    "CellColumn",
    "FileColumn",
    "ListColumn",
    "NumpyArrayColumn",
    "PandasSeriesColumn",
    "TensorColumn",
    "ArrowArrayColumn",
    "ArrowColumn",
    "ImageColumn",
    "AudioColumn",
    "VideoColumn",
    "SpacyColumn",
    "MedicalVolumeColumn",
    "AbstractCell",
    "LambdaCell",
    "FileCell",
    "FileLoader",
    "MedicalVolumeCell",
    "get",
    "concat",
    "merge",
    "embed",
    "sort",
    "sample",
    "provenance",
    "config",
    "interactive_mode",
    "gui",
]
