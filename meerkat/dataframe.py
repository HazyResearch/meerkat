"""DataFrame class."""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import cytoolz as tz
import dill
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._libs import lib

import meerkat
from meerkat.block.manager import BlockManager
from meerkat.columns.abstract import Column
from meerkat.columns.scalar.abstract import ScalarColumn
from meerkat.columns.scalar.arrow import ArrowScalarColumn
from meerkat.errors import ConversionError
from meerkat.interactive.graph.marking import is_unmarked_context, unmarked
from meerkat.interactive.graph.reactivity import reactive
from meerkat.interactive.modification import DataFrameModification
from meerkat.interactive.node import NodeMixin
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.mixins.deferable import DeferrableMixin
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.mixins.indexing import IndexerMixin, MaterializationMixin
from meerkat.mixins.inspect_fn import FunctionInspectorMixin
from meerkat.mixins.reactifiable import ReactifiableMixin
from meerkat.provenance import ProvenanceMixin
from meerkat.row import Row
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.tools.utils import convert_to_batch_fn, dump_yaml, load_yaml

torch = LazyLoader("torch")

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from meerkat.interactive.formatter.base import FormatterGroup


logger = logging.getLogger(__name__)

Example = Dict
Batch = Dict[str, Union[List, Column]]
BatchOrDataset = Union[Batch, "DataFrame"]


class DataFrame(
    CloneableMixin,
    FunctionInspectorMixin,
    IdentifiableMixin,
    NodeMixin,
    DeferrableMixin,
    MaterializationMixin,
    IndexerMixin,
    ProvenanceMixin,
    ReactifiableMixin,
):
    """A collection of equal length columns.

    Args:
        data (Union[dict, list]): A dictionary of columns or a list of dictionaries.
        primary_key (str, optional): The name of the primary key column.
            Defaults to ``None``.
    """

    _self_identifiable_group: str = "dataframes"

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "meerkat/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        data: Union[dict, list] = None,
        primary_key: Union[str, bool] = True,
        *args,
        **kwargs,
    ):
        super(DataFrame, self).__init__(
            *args,
            **kwargs,
        )
        self._primary_key = None if primary_key is False else primary_key
        self.data = data

        if primary_key is True:
            self._infer_primary_key(create=True)

    def _react(self):
        """Converts the object to a reactive object in-place."""
        self._self_marked = True
        for col in self.columns:
            self[col].mark()
        return self

    def _no_react(self):
        """Converts the object to a non-reactive object in-place."""
        self._self_marked = False
        for col in self.columns:
            self[col].unmark()
        return self

    @property
    def gui(self):
        from meerkat.interactive.gui import DataFrameGUI

        return DataFrameGUI(self)

    @reactive
    def format(self, formatters: Dict[str, FormatterGroup]) -> DataFrame:
        """Create a view of the DataFrame with formatted columns.

        Example

        Args:
            formatters (Dict[str, FormatterGroup]): A dictionary mapping column names to
                FormatterGroups.

        Returns:
            DataFrame: A view of the DataFrame with formatted columns.

        Examples
        ---------

        .. code-block:: python

            # assume df is a DataFrame with columns "img", "text", "id"

            gallery = mk.Gallery(
                df=df.format(
                    img={"thumbnail": ImageFormatter(max_size=(48, 48))},
                    text={"icon": TextFormatter()},
                )
            )
        """
        df = self.view()
        for k, v in formatters.items():
            if k not in self.columns:
                raise ValueError(f"Column {k} not found in DataFrame.")
            df[k] = df[k].format(v)
        return df

    @unmarked()
    def _repr_pandas_(self, max_rows: int = None):
        if max_rows is None:
            max_rows = meerkat.config.display.max_rows

        df, formatters = self.data._repr_pandas_(max_rows=max_rows)
        rename = {k: f"{k}" for k, v in self.items()}
        return (
            df[self.columns].rename(columns=rename),
            {rename[k]: v for k, v in formatters.items()},
        )

    @unmarked()
    def _ipython_display_(self):
        from IPython.display import HTML, display

        from meerkat.state import state

        if state.frontend_info is None:
            max_rows = meerkat.config.display.max_rows
            df, formatters = self._repr_pandas_(max_rows=max_rows)
            return display(
                HTML(df.to_html(formatters=formatters, max_rows=max_rows, escape=False))
            )

        return self.gui.table()._ipython_display_()

    @unmarked()
    def __repr__(self):
        return (
            f"{self.__class__.__name__}" f"(nrows: {self.nrows}, ncols: {self.ncols})"
        )

    def __len__(self):
        # __len__ is required to return an int. It cannot return a Store[int].
        # As such, it cannot be wrapped in `@reactive` decorator.
        # To get the length of a DataFrame in a reactive way, use `df.nrows`.
        self._reactive_warning("len", placeholder="df")
        with unmarked():
            _len = self.nrows
        return _len

    def __contains__(self, item):
        # __contains__ is called when using the `in` keyword.
        # `in` casts the output to a boolean (i.e. `bool(self.__contains__(item))`).
        # Store.__bool__ is not reactive because __bool__ has to return
        # a bool (not a Store). Thus, we cannot wrap __contains__ in
        # `@reactive` decorator.
        if not is_unmarked_context() and self.marked:
            warnings.warn(
                "The `in` operator is not reactive. Use `df.contains(...)`:\n"
                "\t>>> df.contains(item)"
            )
        return self.columns.__contains__(item)

    @reactive()
    def contains(self, item):
        return self.__contains__(item)

    @property
    def data(self) -> BlockManager:
        """Get the underlying data (excluding invisible rows).

        To access underlying data with invisible rows, use `_data`.
        """
        return self._data

    def _set_data(self, value: Union[BlockManager, Mapping] = None):
        if isinstance(value, BlockManager):
            self._data = value
        elif isinstance(value, Mapping):
            self._data = BlockManager.from_dict(value)
        elif isinstance(value, Sequence):
            if not isinstance(value[0], Mapping):
                raise ValueError(
                    "Cannot set DataFrame `data` to a Sequence containing object of "
                    f" type {type(value[0])}. Must be a Sequence of Mapping."
                )
            gen = (list(x.keys()) for x in value)
            columns = lib.fast_unique_multiple_list_gen(gen)
            data = lib.dicts_to_array(value, columns=columns)
            # Assert all columns are the same length
            data = {column: list(data[:, idx]) for idx, column in enumerate(columns)}
            self._data = BlockManager.from_dict(data)
        elif value is None:
            self._data = BlockManager()
        else:
            raise ValueError(
                f"Cannot set DataFrame `data` to object of type {type(value)}."
            )

        # check that the primary key is still valid after data reset
        if self._primary_key is not None:
            if (
                self._primary_key not in self
                or not self.primary_key._is_valid_primary_key()
            ):
                self.set_primary_key(None)

    @data.setter
    def data(self, value):
        self._set_data(value)

    @property
    def columns(self):
        """Column names in the DataFrame."""
        return list(self.data.keys())

    @property
    def primary_key(self) -> ScalarColumn:
        """The column acting as the primary key."""
        if self._primary_key is None:
            return None

        return self[self._primary_key]

    @property
    def primary_key_name(self) -> str:
        """The name of the column acting as the primary key."""
        return self._primary_key

    def set_primary_key(self, column: str, inplace: bool = False) -> DataFrame:
        """Set the DataFrame's primary key using an existing column. This is an
        out-of-place operation. For more information on primary keys, see the
        User Guide.

        Args:
            column (str): The name of an existing column to set as the primary key.
        """
        if column is not None:
            if column not in self.columns:
                raise ValueError(
                    "Must pass the name of an existing column to `set_primary_key`."
                    f'"{column}" not in {self.columns}'
                )

            if not self[column]._is_valid_primary_key():
                raise ValueError(
                    f'Column "{column}" cannot be used as a primary key. Ensure that '
                    "columns of this type can be used as primary keys and \
                    that the values in the column are unique."
                )

        if inplace:
            self._primary_key = column
            return self
        else:
            return self._clone(_primary_key=column)

    def create_primary_key(self, column: str):
        """Create a primary key of contiguous integers.

        Args:
            column (str): The name of the column to create.
        """
        self[column] = np.arange(self.nrows)
        self.set_primary_key(column, inplace=True)

    def _infer_primary_key(self, create: bool = False) -> str:
        """Infer the primary key from the data."""
        for column in self.columns:
            if self[column]._is_valid_primary_key():
                self.set_primary_key(column, inplace=True)
                return column
        if create:
            self.create_primary_key("pkey")
            return "pkey"
        return None

    @property
    def nrows(self):
        """Number of rows in the DataFrame."""
        if self.ncols == 0:
            return 0
        return self._data.nrows

    @property
    def ncols(self):
        """Number of rows in the DataFrame."""
        return self.data.ncols

    @property
    def shape(self):
        """Shape of the DataFrame (num_rows, num_columns)."""
        return self.nrows, self.ncols

    @reactive(nested_return=False)
    def size(self):
        """Shape of the DataFrame (num_rows, num_columns)."""
        return self.shape

    def add_column(self, name: str, data: Column.Columnable, overwrite=False) -> None:
        """Add a column to the DataFrame."""

        assert isinstance(
            name, str
        ), f"Column name must of type `str`, not `{type(name)}`."

        assert (name not in self.columns) or overwrite, (
            f"Column with name `{name}` already exists, "
            f"set `overwrite=True` to overwrite."
        )

        if not is_listlike(data):
            data = [data] * self.nrows

        if name in self.columns:
            self.remove_column(name)

        column = Column.from_data(data)

        assert len(column) == len(self) or len(self.columns) == 0, (
            f"`add_column` failed. "
            f"Values length {len(column)} != dataset length {len(self)}."
        )

        # Add the column
        self.data[name] = column

        logger.info(f"Added column `{name}` with length `{len(column)}`.")

    def remove_column(self, column: str) -> None:
        """Remove a column from the dataset."""
        assert column in self.columns, f"Column `{column}` does not exist."

        # Remove the column
        del self.data[column]

        if column == self._primary_key:
            # need to reset the primary key if we remove the column
            self.set_primary_key(None, inplace=True)

        logger.info(f"Removed column `{column}`.")

    # @capture_provenance(capture_args=["axis"])
    @reactive()
    def append(
        self,
        df: DataFrame,
        axis: Union[str, int] = "rows",
        suffixes: Tuple[str] = None,
        overwrite: bool = False,
    ) -> DataFrame:
        """Append a batch of data to the dataset.

        `example_or_batch` must have the same columns as the dataset
        (regardless of what columns are visible).
        """
        return meerkat.concat(
            [self, df], axis=axis, suffixes=suffixes, overwrite=overwrite
        )

    @reactive()
    def head(self, n: int = 5) -> DataFrame:
        """Get the first `n` examples of the DataFrame."""
        return self[:n]

    @reactive()
    def tail(self, n: int = 5) -> DataFrame:
        """Get the last `n` examples of the DataFrame."""
        return self[-n:]

    def _get_loc(self, keyidx, materialize: bool = False):
        if self.primary_key is None:
            raise ValueError(
                "Cannot use `loc` without a primary key. Set a primary key using "
                "`set_primary_key`."
            )

        if isinstance(
            keyidx, (np.ndarray, list, tuple, pd.Series, torch.Tensor, Column)
        ):
            posidxs = self.primary_key._keyidxs_to_posidxs(keyidx)
            return self._clone(
                data=self.data.apply("_get", index=posidxs, materialize=materialize)
            )

        else:
            posidx = self.primary_key._keyidx_to_posidx(keyidx)
            row = self.data.apply("_get", index=posidx, materialize=materialize)
            return {k: row[k] for k in self.columns}

    def _set_loc(self, keyidx: Union[str, int], column: str, value: any):
        if self.primary_key is None:
            raise ValueError(
                "Cannot use `loc` without a primary key. Set a primary key using "
                "`set_primary_key`."
            )

        if isinstance(
            keyidx, (np.ndarray, list, tuple, pd.Series, torch.Tensor, Column)
        ):
            raise NotImplementedError("")
        else:
            posidx = self.primary_key._keyidx_to_posidx(keyidx)
            self[column][posidx] = value
            if self.has_inode():
                mod = DataFrameModification(id=self.inode.id, scope=self.columns)
                mod.add_to_queue()

    def _get(self, posidx, materialize: bool = False):
        if isinstance(posidx, str):
            # str index => column selection (AbstractColumn)
            if posidx in self.columns:
                return self.data[posidx]
            raise KeyError(f"Column `{posidx}` does not exist.")

        elif isinstance(posidx, int):
            # int index => single row (dict)
            row = self.data.apply("_get", index=posidx, materialize=materialize)
            return Row({k: row[k] for k in self.columns})

        # cases where `index` returns a dataframe
        index_type = None
        if isinstance(posidx, slice):
            # slice index => multiple row selection (DataFrame)
            index_type = "row"

        elif (isinstance(posidx, tuple) or isinstance(posidx, list)) and len(posidx):
            # tuple or list index => multiple row selection (DataFrame)
            if isinstance(posidx[0], str):
                index_type = "column"
            else:
                index_type = "row"

        elif isinstance(posidx, np.ndarray):
            if len(posidx.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(posidx.shape))
                )
            # numpy array index => multiple row selection (DataFrame)
            index_type = "row"

        elif torch.is_tensor(posidx):
            if len(posidx.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(posidx.shape))
                )
            # torch tensor index => multiple row selection (DataFrame)
            index_type = "row"

        elif isinstance(posidx, pd.Series):
            index_type = "row"

        elif isinstance(posidx, Column):
            # column index => multiple row selection (DataFrame)
            index_type = "row"

        else:
            raise TypeError("Invalid index type: {}".format(type(posidx)))

        if index_type == "column":
            if not set(posidx).issubset(self.columns):
                missing_cols = set(posidx) - set(self.columns)
                raise KeyError(f"DataFrame does not have columns {missing_cols}")

            df = self._clone(data=self.data[posidx])
            return df
        elif index_type == "row":  # pragma: no cover
            return self._clone(
                data=self.data.apply("_get", index=posidx, materialize=materialize)
            )

    # @capture_provenance(capture_args=[])
    @reactive()
    def __getitem__(self, posidx):
        return self._get(posidx, materialize=False)

    def __setitem__(self, posidx, value):
        self.add_column(name=posidx, data=value, overwrite=True)

        # This condition will issue modifications even if we're outside an endpoint
        # but those modifications will be cleared by the endpoint before it is
        # run.
        if self.has_inode():
            # Add a modification if it's on the graph
            mod = DataFrameModification(id=self.id, scope=self.columns)
            mod.add_to_queue()

    def __call__(self):
        return self._get(slice(0, len(self), 1), materialize=True)

    def set(self, value: DataFrame):
        """Set the data of this DataFrame to the data of another DataFrame.

        This is used inside endpoints to tell Meerkat when a DataFrame
        has been modified. Calling this method outside of an endpoint
        will not have any effect on the graph.
        """
        self._set_data(value._data)
        self._set_state(value._get_state())

        if self.has_inode():
            # Add a modification if it's on the graph
            # TODO: think about what the scope should be.
            #   How does `scope` relate to the the skip_fn mechanism
            #   in Operation?
            mod = DataFrameModification(id=self.inode.id, scope=self.columns)
            mod.add_to_queue()

    def consolidate(self):
        self.data.consolidate()

    @classmethod
    # @capture_provenance()
    def from_batch(
        cls,
        batch: Batch,
    ) -> DataFrame:
        """Convert a batch to a Dataset."""
        return cls(batch)

    @classmethod
    # @capture_provenance()
    def from_batches(
        cls,
        batches: Sequence[Batch],
    ) -> DataFrame:
        """Convert a list of batches to a dataset."""

        return cls.from_batch(
            tz.merge_with(
                tz.compose(list, tz.concat),
                *batches,
            ),
        )

    @classmethod
    # @capture_provenance()
    def from_pandas(
        cls,
        df: pd.DataFrame,
        index: bool = True,
        primary_key: Optional[str] = None,
    ) -> DataFrame:
        """Create a Meerkat DataFrame from a Pandas DataFrame.

        .. warning::

            In Meerkat, column names must be strings, so non-string column
            names in the Pandas DataFrame will be converted.

        Args:
            df: The Pandas DataFrame to convert.
            index: Whether to include the index of the Pandas DataFrame as a
                column in the Meerkat DataFrame.
            primary_key: The name of the column to use as the primary key.
                If `index` is True and `primary_key` is None, the index will
                be used as the primary key. If `index` is False, then no
                primary key will be set. Optional default is None.

        Returns:
            DataFrame: The Meerkat DataFrame.
        """

        # column names must be str in meerkat
        df = df.rename(mapper=str, axis="columns")
        pandas_df = df

        df = cls.from_batch(
            pandas_df.to_dict("series"),
        )

        if index:
            df["index"] = pandas_df.index.to_series()
            if primary_key is None:
                df.set_primary_key("index", inplace=True)

        if primary_key is not None:
            df.set_primary_key(primary_key, inplace=True)

        return df

    @classmethod
    # @capture_provenance()
    def from_arrow(
        cls,
        table: pa.Table,
    ):
        """Create a Dataset from a pandas DataFrame."""
        from meerkat.block.arrow_block import ArrowBlock
        from meerkat.columns.scalar.arrow import ArrowScalarColumn

        block_views = ArrowBlock.from_block_data(table)
        return cls.from_batch(
            {view.block_index: ArrowScalarColumn(view) for view in block_views}
        )

    @classmethod
    def from_huggingface(cls, *args, **kwargs):
        """Load a Huggingface dataset as a DataFrame.

        Use this to replace `datasets.load_dataset`, so

        >>> dict_of_datasets = datasets.load_dataset('boolq')

        becomes

        >>> dict_of_dataframes = DataFrame.from_huggingface('boolq')
        """
        import datasets

        # Load the dataset
        dataset = datasets.load_dataset(*args, **kwargs)

        if isinstance(dataset, dict):
            return dict(
                map(
                    lambda t: (t[0], cls.from_arrow(t[1]._data)),
                    dataset.items(),
                )
            )
        else:
            return cls.from_arrow(dataset._data)

    @classmethod
    # @capture_provenance(capture_args=["filepath"])
    def from_csv(
        cls,
        filepath: str,
        primary_key: str = None,
        backend: str = "pandas",
        *args,
        **kwargs,
    ) -> DataFrame:
        """Create a DataFrame from a csv file. All of the columns will be
        :class:`meerkat.ScalarColumn` with backend Pandas.

        Args:
            filepath (str): The file path or buffer to load from.
                Same as :func:`pandas.read_csv`.
            *args: Argument list for :func:`pandas.read_csv`.
            **kwargs: Keyword arguments forwarded to :func:`pandas.read_csv`.

        Returns:
            DataFrame: The constructed dataframe.
        """
        if backend == "pandas":
            df = cls.from_pandas(pd.read_csv(filepath, *args, **kwargs), index=False)
        elif backend == "arrow":
            df = cls.from_arrow(pa.csv.read_csv(filepath, *args, **kwargs))
        if primary_key is not None:
            df.set_primary_key(primary_key, inplace=True)
        return df

    @classmethod
    # @capture_provenance()
    def from_feather(
        cls,
        filepath: str,
        primary_key: str = None,
        columns: Optional[Sequence[str]] = None,
        use_threads: bool = True,
        **kwargs,
    ) -> DataFrame:
        """Create a DataFrame from a feather file. All of the columns will be
        :class:`meerkat.ScalarColumn` with backend Pandas.

        Args:
            filepath (str): The file path or buffer to load from.
                Same as :func:`pandas.read_feather`.
            columns (Optional[Sequence[str]]): The columns to load.
                Same as :func:`pandas.read_feather`.
            use_threads (bool): Whether to use threads to read the file.
                Same as :func:`pandas.read_feather`.
            **kwargs: Keyword arguments forwarded to :func:`pandas.read_feather`.

        Returns:
            DataFrame: The constructed dataframe.
        """
        df = cls.from_pandas(
            pd.read_feather(
                filepath, columns=columns, use_threads=use_threads, **kwargs
            ),
            index=False,
        )
        if primary_key is not None:
            df.set_primary_key(primary_key, inplace=True)
        return df

    @classmethod
    # @capture_provenance()
    def from_parquet(
        cls,
        filepath: str,
        primary_key: str = None,
        engine: str = "auto",
        columns: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> DataFrame:
        """Create a DataFrame from a parquet file. All of the columns will be
        :class:`meerkat.ScalarColumn` with backend Pandas.

        Args:
            filepath (str): The file path or buffer to load from.
                Same as :func:`pandas.read_parquet`.
            engine (str): The parquet engine to use.
                Same as :func:`pandas.read_parquet`.
            columns (Optional[Sequence[str]]): The columns to load.
                Same as :func:`pandas.read_parquet`.
            **kwargs: Keyword arguments forwarded to :func:`pandas.read_parquet`.

        Returns:
            DataFrame: The constructed dataframe.
        """
        df = cls.from_pandas(
            pd.read_parquet(filepath, engine=engine, columns=columns, **kwargs),
            index=False,
        )
        if primary_key is not None:
            df.set_primary_key(primary_key, inplace=True)
        return df

    @classmethod
    def from_json(
        cls,
        filepath: str,
        primary_key: str = None,
        orient: str = "records",
        lines: bool = False,
        backend: str = "pandas",
        **kwargs,
    ) -> DataFrame:
        """Load a DataFrame from a json file.

        By default, data in the JSON file should be a list of dictionaries, each with
        an entry for each column. This is the ``orient="records"`` format. If the data
        is in a different format in the JSON, you can specify the ``orient`` parameter.
        See :func:`pandas.read_json` for more details.

        Args:
            filepath (str): The file path or buffer to load from.
                Same as :func:`pandas.read_json`.
            orient (str): The expected JSON string format. Options are:
                "split", "records", "index", "columns", "values".
                Same as :func:`pandas.read_json`.
            lines (bool): Whether the json file is a jsonl file.
                Same as :func:`pandas.read_json`.
            backend (str): The backend to use for the loading and reuslting columns.
            **kwargs: Keyword arguments forwarded to :func:`pandas.read_json`.

        Returns:
            DataFrame: The constructed dataframe.
        """
        from pyarrow import json

        if backend == "arrow":
            if lines is False:
                raise ValueError("Arrow backend only supports lines=True.")
            df = cls.from_arrow(json.read_json(filepath, **kwargs))
        elif backend == "pandas":
            df = cls.from_pandas(
                pd.read_json(filepath, orient=orient, lines=lines, **kwargs),
                index=False,
            )
        if primary_key is not None:
            df.set_primary_key(primary_key, inplace=True)
        return df

    def _to_base(self, base: str = "pandas", **kwargs) -> Dict[str, Any]:
        """Helper method to convert a Meerkat DataFrame to a base format."""
        data = {}
        for name, column in self.items():
            try:
                converted_column = getattr(column, f"to_{base}")(**kwargs)
            except ConversionError:
                warnings.warn(
                    f"Could not convert column {name} of type {type(column)}, it will "
                    "be dropped from the output."
                )
                continue
            data[name] = converted_column
        return data

    def to_pandas(
        self, index: bool = False, allow_objects: bool = False
    ) -> pd.DataFrame:
        """Convert a Meerkat DataFrame to a Pandas DataFrame.

        Args:
            index (bool): Use the primary key as the index of the Pandas
                DataFrame. Defaults to False.

        Returns:
            pd.DataFrame: The constructed dataframe.
        """

        df = pd.DataFrame(self._to_base(base="pandas", allow_objects=allow_objects))
        if index:
            df.set_index(self.primary_key, inplace=True)
        return df

    def to_arrow(self) -> pd.DataFrame:
        """Convert a Meerkat DataFrame to an Arrow Table.

        Returns:
            pa.Table: The constructed table.
        """
        return pa.table(self._to_base(base="arrow"))

    def _choose_engine(self):
        """Choose which library to use for export.

        If any of the columns are ArrowScalarColumn, then use Arrow.
        """
        for column in self.values():
            if isinstance(column, ArrowScalarColumn):
                return "arrow"
        return "pandas"

    def to_csv(self, filepath: str, engine: str = "auto"):
        """Save a DataFrame to a csv file.

        The engine used to write the csv to disk.

        Args:
            filepath (str): The file path to save to.
            engine (str): The library to use to write the csv. One of ["pandas",
                "arrow", "auto"]. If "auto", then the library will be chosen based on
                the column types.
        """
        if engine == "auto":
            engine = self._choose_engine()

        if engine == "arrow":
            from pyarrow.csv import write_csv

            write_csv(self.to_arrow(), filepath)
        else:
            self.to_pandas().to_csv(filepath, index=False)

    def to_feather(self, filepath: str, engine: str = "auto"):
        """Save a DataFrame to a feather file.

        The engine used to write the feather to disk.

        Args:
            filepath (str): The file path to save to.
            engine (str): The library to use to write the feather. One of ["pandas",
                "arrow", "auto"]. If "auto", then the library will be chosen based on
                the column types.
        """
        if engine == "auto":
            engine = self._choose_engine()

        if engine == "arrow":
            from pyarrow.feather import write_feather

            write_feather(self.to_arrow(), filepath)
        else:
            self.to_pandas().to_feather(filepath)

    def to_parquet(self, filepath: str, engine: str = "auto"):
        """Save a DataFrame to a parquet file.

        The engine used to write the parquet to disk.

        Args:
            filepath (str): The file path to save to.
            engine (str): The library to use to write the parquet. One of ["pandas",
                "arrow", "auto"]. If "auto", then the library will be chosen based on
                the column types.
        """
        if engine == "auto":
            engine = self._choose_engine()

        if engine == "arrow":
            from pyarrow.parquet import write_table

            write_table(self.to_arrow(), filepath)
        else:
            self.to_pandas().to_parquet(filepath)

    def to_json(
        self, filepath: str, lines: bool = False, orient: str = "records"
    ) -> None:
        """Save a Dataset to a json file.

        Args:
            filepath (str): The file path to save to.
            lines (bool): Whether to write the json file as a jsonl file.
            orient (str): The orientation of the json file.
                Same as :func:`pandas.DataFrame.to_json`.
        """
        self.to_pandas().to_json(filepath, lines=lines, orient=orient)

    def _get_collate_fns(self, columns: Iterable[str] = None):
        columns = self.data.keys() if columns is None else columns
        return {name: self.data[name].collate for name in columns}

    def _collate(self, batch: List):
        batch = tz.merge_with(list, *batch)
        column_to_collate = self._get_collate_fns(batch.keys())
        new_batch = {}
        for name, values in batch.items():
            new_batch[name] = column_to_collate[name](values)
        df = self._clone(data=new_batch)
        return df

    @staticmethod
    def _convert_to_batch_fn(
        function: Callable, with_indices: bool, materialize: bool = True, **kwargs
    ) -> callable:
        return convert_to_batch_fn(
            function=function,
            with_indices=with_indices,
            materialize=materialize,
            **kwargs,
        )

    def batch(
        self,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        materialize: bool = True,
        shuffle: bool = False,
        *args,
        **kwargs,
    ):
        """Batch the dataset.
        TODO:

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size

        Returns:
            batches of data
        """
        cell_columns, batch_columns = [], []
        from meerkat.columns.deferred.base import DeferredColumn

        for name, column in self.items():
            if isinstance(column, DeferredColumn) and materialize:
                cell_columns.append(name)
            else:
                batch_columns.append(name)

        indices = np.arange(len(self))

        if shuffle:
            indices = np.random.permutation(indices)

        if batch_columns:
            batch_indices = []
            for i in range(0, len(self), batch_size):
                if drop_last_batch and i + batch_size > len(self):
                    continue
                batch_indices.append(indices[i : i + batch_size])
            batch_dl = torch.utils.data.DataLoader(
                self[batch_columns],
                sampler=batch_indices,
                batch_size=None,
                batch_sampler=None,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )

        if cell_columns:
            df = self[cell_columns] if not shuffle else self[cell_columns][indices]
            cell_dl = torch.utils.data.DataLoader(
                df,
                batch_size=batch_size,
                collate_fn=self._collate,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )

        if batch_columns and cell_columns:
            for cell_batch, batch_batch in zip(cell_dl, batch_dl):
                out = self._clone(data={**cell_batch.data, **batch_batch.data})
                yield out() if materialize else out
        elif batch_columns:
            for batch_batch in batch_dl:
                yield batch_batch() if materialize else batch_batch
        elif cell_columns:
            for cell_batch in cell_dl:
                yield cell_batch() if materialize else cell_batch

    # @capture_provenance(capture_args=["with_indices"])
    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        remove_columns: Optional[List[str]] = None,
        num_workers: int = 0,
        output_type: Union[type, Dict[str, type]] = None,
        mmap: bool = False,
        mmap_path: str = None,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> DataFrame:
        """Update the columns of the dataset."""
        # TODO(karan): make this fn go faster
        # most of the time is spent on the merge, speed it up further

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return self

        # Get some information about the function
        df = self[input_columns] if input_columns is not None else self
        function_properties = df._inspect_function(
            function, with_indices, is_batched_fn, materialize=materialize, **kwargs
        )
        assert (
            function_properties.dict_output
        ), f"`function` {function} must return dict."

        if not is_batched_fn:
            # Convert to a batch function
            function = convert_to_batch_fn(
                function, with_indices=with_indices, materialize=materialize, **kwargs
            )
            logger.info(f"Converting `function` {function} to batched function.")

        # Update always returns a new dataset
        logger.info("Running update, a new dataset will be returned.")

        # Copy the ._data dict with a reference to the actual columns
        new_df = self.view()

        # Calculate the values for the new columns using a .map()
        output = new_df.map(
            function=function,
            with_indices=with_indices,
            is_batched_fn=True,
            batch_size=batch_size,
            num_workers=num_workers,
            output_type=output_type,
            input_columns=input_columns,
            mmap=mmap,
            mmap_path=mmap_path,
            materialize=materialize,
            pbar=pbar,
            **kwargs,
        )

        # Add new columns for the update
        for col, vals in output.data.items():
            new_df.add_column(col, vals, overwrite=True)

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_df.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")

        return new_df

    def map(
        self,
        function: Callable,
        is_batched_fn: bool = False,
        batch_size: int = 1,
        inputs: Union[Mapping[str, str], Sequence[str]] = None,
        outputs: Union[Mapping[any, str], Sequence[str]] = None,
        output_type: Union[Mapping[str, Type["Column"]], Type["Column"]] = None,
        materialize: bool = True,
        **kwargs,
    ) -> Optional[Union[Dict, List, Column]]:
        from meerkat.ops.map import map

        return map(
            data=self,
            function=function,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            inputs=inputs,
            outputs=outputs,
            output_type=output_type,
            materialize=materialize,
            **kwargs,
        )

    # @capture_provenance(capture_args=["function"])
    @reactive()
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> Optional[DataFrame]:
        """Filter operation on the DataFrame."""

        # Return if `self` has no examples
        if not len(self):
            logger.info("DataFrame empty, returning None.")
            return None

        # Get some information about the function
        df = self[input_columns] if input_columns is not None else self
        function_properties = df._inspect_function(
            function,
            with_indices,
            is_batched_fn=is_batched_fn,
            materialize=materialize,
            **kwargs,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new DataFrame will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_workers=num_workers,
            materialize=materialize,
            pbar=pbar,
            **kwargs,
        )
        indices = np.where(outputs)[0]

        # filter returns a new dataframe
        return self[indices]

    @reactive()
    def merge(
        self,
        right: meerkat.DataFrame,
        how: str = "inner",
        on: Union[str, List[str]] = None,
        left_on: Union[str, List[str]] = None,
        right_on: Union[str, List[str]] = None,
        sort: bool = False,
        suffixes: Sequence[str] = ("_x", "_y"),
        validate=None,
    ):
        from meerkat import merge

        return merge(
            self,
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            sort=sort,
            suffixes=suffixes,
            validate=validate,
        )

    @reactive()
    def sort(
        self,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        kind: str = "quicksort",
    ) -> DataFrame:
        """Sort the DataFrame by the values in the specified columns. Similar
        to ``sort_values`` in pandas.

        Args:
            by (Union[str, List[str]]): The columns to sort by.
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`.Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.

        Return:
            DataFrame: A sorted view of DataFrame.
        """
        from meerkat import sort

        return sort(data=self, by=by, ascending=ascending, kind=kind)

    @reactive()
    def sample(
        self,
        n: int = None,
        frac: float = None,
        replace: bool = False,
        weights: Union[str, np.ndarray] = None,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> DataFrame:
        """Select a random sample of rows from DataFrame. Roughly equivalent to
        ``sample`` in Pandas https://pandas.pydata.org/docs/reference/api/panda
        s.DataFrame.sample.html.

        Args:
            n (int): Number of samples to draw. If `frac` is specified, this parameter
                should not be passed. Defaults to 1 if `frac` is not passed.
            frac (float): Fraction of rows to sample. If `n` is specified, this
                parameter should not be passed.
            replace (bool): Sample with or without replacement. Defaults to False.
            weights (Union[str, np.ndarray]): Weights to use for sampling. If `None`
                (default), the rows will be sampled uniformly. If a numpy array, the
                sample will be weighted accordingly. If a string, the weights will be
                applied to the rows based on the column with the name specified. If
                weights do not sum to 1 they will be normalized to sum to 1.
            random_state (Union[int, np.random.RandomState]): Random state or seed to
                use for sampling.

        Return:
            DataFrame: A random sample of rows from the DataFrame.
        """
        from meerkat import sample

        return sample(
            data=self,
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
        )

    @reactive()
    def shuffle(self, seed: int = None) -> DataFrame:
        """Shuffle the rows of the DataFrame out-of-place.

        Args:
            seed (int): Random seed to use for shuffling.

        Returns:
            DataFrame: A shuffled view of the DataFrame.
        """
        from meerkat import shuffle

        return shuffle(data=self, seed=seed)

    @reactive()
    def rename(
        self,
        mapper: Union[Dict, Callable] = None,
        errors: Literal["ignore", "raise"] = "ignore",
    ) -> DataFrame:
        """Return a new DataFrame with the specified column labels renamed.

        Dictionary values must be unique (1-to-1). Labels not specified will be
        left unchanged. Extra labels will not throw an error.

        Args:
            mapper (Union[Dict, Callable], optional): Dict-like of function
                transformations to apply to the values of the columns. Defaults
                to None.
            errors (Literal['ignore', 'raise'], optional): If 'raise', raise a
                KeyError when the Dict contains labels that do not exist in the
                DataFrame. If 'ignore', extra keys will be ignored. Defaults to
                'ignore'.

        Raises:
            ValueError: _description_

        Returns:
            DataFrame: A new DataFrame with the specified column labels renamed.
        """
        # Copy the ._data dict with a reference to the actual columns
        new_df = self.view()

        if isinstance(mapper, Dict):
            assert len(set(mapper.values())) == len(
                mapper.values()
            ), "Dictionary values must be unique (1-to-1)."

            names = self.columns  # used to preserve order of columns
            for i, (old_name, new_name) in enumerate(mapper.items()):
                if old_name not in self.keys():
                    if errors == "raise":
                        raise KeyError(
                            f"Cannot rename nonexistent column `{old_name}`."
                        )
                    continue

                if old_name == new_name:
                    continue

                # Copy old column into new column
                new_df[new_name] = new_df[old_name]
                new_df.remove_column(old_name)
                names[names.index(old_name)] = new_name
            new_df = new_df[names]
        elif isinstance(mapper, Callable):
            for old_name in new_df.columns:
                new_name = mapper(old_name)

                if old_name == new_name:
                    continue

                # Copy old column into new column
                new_df[new_name] = new_df[old_name]
                new_df.remove_column(old_name)
        else:
            logger.info(
                f"Mapper type is not one of Dict or Callable: {type(mapper)}. Returning"
            )

        return new_df

    @reactive()
    def drop(
        self, columns: Union[str, Collection[str]], check_exists=True
    ) -> DataFrame:
        """Return a new DataFrame with the specified columns dropped.

        Args:
            columns (Union[str, Collection[str]]): The columns to drop.

        Return:
            DataFrame: A new DataFrame with the specified columns dropped.
        """
        if isinstance(columns, str):
            columns = [columns]
        for c in columns:
            if c not in self.columns and check_exists:
                raise ValueError(
                    f"Cannot drop nonexistent column '{c}' from DataFrame."
                )
        return self[[c for c in self.columns if c not in columns]]

    def items(self):
        # TODO: Add support for decorating iterators with reactive.
        for name in self.columns:
            yield name, self.data[name]

    @reactive()
    def keys(self):
        return self.columns

    def iterrows(self):
        for i in range(len(self)):
            yield self[i]

    def values(self):
        # TODO: Add support for decorating iterators with reactive.
        for name in self.columns:
            yield self.data[name]

    @classmethod
    def read(
        cls,
        path: str,
        overwrite: bool = False,
        *args,
        **kwargs,
    ) -> DataFrame:
        """Load a DataFrame stored on disk."""
        from meerkat.datasets.utils import download_df, extract_tar_file

        # URL
        if path.startswith("http://") or path.startswith("https://"):
            return download_df(path, overwrite=overwrite)

        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")

        # Compressed file.
        # TODO: Add support for other compressed formats.
        if path.endswith(".tar.gz"):
            download_dir = os.path.join(os.path.dirname(path), ".mktemp")
            cache_dir = os.path.join(
                download_dir, os.path.basename(path).split(".tar.gz")[0]
            )
            os.makedirs(download_dir, exist_ok=True)
            extract_tar_file(path, download_dir=download_dir)
            df = cls.read(cache_dir, overwrite=overwrite, *args, **kwargs)
            shutil.rmtree(download_dir)
            return df

        # Load the metadata
        metadata = load_yaml(os.path.join(path, "meta.yaml"))

        if "state" in metadata:
            state = metadata["state"]
        else:
            # backwards compatability to dill dataframes
            state = dill.load(open(os.path.join(path, "state.dill"), "rb"))
        df = cls.__new__(cls)
        df._set_id()  # TODO: consider if we want to persist this id
        df._set_inode()  # FIXME: this seems hacky
        df._set_state(state)

        # Load the the manager
        mgr_dir = os.path.join(path, "mgr")
        if os.path.exists(mgr_dir):
            data = BlockManager.read(mgr_dir, **kwargs)
        else:
            # backwards compatability to pre-manager dataframes
            data = {
                name: dtype.read(os.path.join(path, "columns", name), *args, **kwargs)
                for name, dtype in metadata["column_dtypes"].items()
            }

        df._set_data(data)

        return df

    def write(
        self,
        path: str,
    ) -> None:
        """Save a DataFrame to disk."""
        # Make all the directories to the path
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)

        # Get the DataFrame state
        state = self._get_state()

        # Get the metadata
        metadata = {
            "dtype": type(self),
            "column_dtypes": {name: type(col) for name, col in self.data.items()},
            "len": len(self),
            "state": state,
        }

        # write the block manager
        mgr_dir = os.path.join(path, "mgr")
        self.data.write(mgr_dir)

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        dump_yaml(metadata, metadata_path)

    @classmethod
    def _state_keys(cls) -> Set[str]:
        """List of attributes that describe the state of the object."""
        return {"_primary_key"}

    def _set_state(self, state: dict):
        self.__dict__.update(state)

        # backwards compatibility for old dataframes
        if "_primary_key" not in state:
            self._primary_key = None

    def _view_data(self) -> object:
        return self.data.view()

    def _copy_data(self) -> object:
        return self.data.copy()

    def __finalize__(self, *args, **kwargs):
        return self

    @reactive()
    def groupby(self, *args, **kwargs):
        from meerkat.ops.sliceby.groupby import groupby

        return groupby(self, *args, **kwargs)

    @reactive()
    def sliceby(self, *args, **kwargs):
        from meerkat.ops.sliceby.sliceby import sliceby

        return sliceby(self, *args, **kwargs)

    @reactive()
    def clusterby(self, *args, **kwargs):
        from meerkat.ops.sliceby.clusterby import clusterby

        return clusterby(self, *args, **kwargs)

    @reactive()
    def explainby(self, *args, **kwargs):
        from meerkat.ops.sliceby.explainby import explainby

        return explainby(self, *args, **kwargs)

    @reactive()
    def aggregate(
        self, function: Union[str, Callable], nuisance: str = "drop", *args, **kwargs
    ) -> Dict[str, Any]:
        from meerkat.ops.aggregate.aggregate import aggregate

        return aggregate(self, function, *args, **kwargs)

    @reactive()
    def mean(self, *args, nuisance: str = "drop", **kwargs):
        from meerkat.ops.aggregate.aggregate import aggregate

        return aggregate(self, function="mean", nuisance=nuisance, *args, **kwargs)


def is_listlike(obj) -> bool:
    """Check if the object is listlike.

    Args:
        obj (object): The object to check.

    Return:
        bool: True if the object is listlike, False otherwise.
    """
    is_column = isinstance(obj, Column)
    is_sequential = (
        hasattr(obj, "__len__")
        and hasattr(obj, "__getitem__")
        and not isinstance(obj, str)
    )
    return is_column or is_sequential


DataFrame.mark = DataFrame._react
DataFrame.unmark = DataFrame._no_react
