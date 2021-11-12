"""DataPanel class."""
from __future__ import annotations

import logging
import os
import pathlib
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cytoolz as tz
import datasets
import dill
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import ujson as json
import yaml
from jsonlines import jsonlines
from pandas._libs import lib

import meerkat
from meerkat.block.manager import BlockManager
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.cell_column import CellColumn
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.mixins.inspect_fn import FunctionInspectorMixin
from meerkat.mixins.lambdable import LambdaMixin
from meerkat.mixins.mapping import MappableMixin
from meerkat.mixins.materialize import MaterializationMixin
from meerkat.provenance import ProvenanceMixin, capture_provenance
from meerkat.tools.utils import MeerkatLoader, convert_to_batch_fn

logger = logging.getLogger(__name__)

Example = Dict
Batch = Dict[str, Union[List, AbstractColumn]]
BatchOrDataset = Union[Batch, "DataPanel"]


class DataPanel(
    CloneableMixin,
    FunctionInspectorMixin,
    LambdaMixin,
    MappableMixin,
    MaterializationMixin,
    ProvenanceMixin,
):
    """Meerkat DataPanel class."""

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "meerkat/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        data: Union[dict, list, datasets.Dataset] = None,
        *args,
        **kwargs,
    ):
        super(DataPanel, self).__init__(
            *args,
            **kwargs,
        )
        logger.debug("Creating DataPanel.")

        self.data = data

        # TODO(Sabri): fix add_index for new datset
        # Add an index to the dataset
        if not self.has_index:
            self._add_index()

    def _repr_pandas_(self, max_rows: int = None):
        if max_rows is None:
            max_rows = meerkat.config.DisplayOptions.max_rows

        df, formatters = self.data._repr_pandas_(max_rows=max_rows)
        rename = {k: f"{k} ({v.__class__.__name__})" for k, v in self.items()}
        return (
            df[self.columns].rename(columns=rename),
            {rename[k]: v for k, v in formatters.items()},
        )

    def _repr_html_(self, max_rows: int = None):
        if max_rows is None:
            max_rows = meerkat.config.DisplayOptions.max_rows

        df, formatters = self._repr_pandas_(max_rows=max_rows)

        return df.to_html(formatters=formatters, max_rows=max_rows, escape=False)

    def streamlit(self):
        return self._repr_pandas_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}" f"(nrows: {self.nrows}, ncols: {self.ncols})"
        )

    def __len__(self):
        return self.nrows

    def __contains__(self, item):
        return item in self.columns

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
                    "Cannot set DataPanel `data` to a Sequence containing object of "
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
                f"Cannot set DataPanel `data` to object of type {type(value)}."
            )

    @data.setter
    def data(self, value):
        self._set_data(value)

    @property
    def columns(self):
        """Column names in the DataPanel."""
        return list(self.data.keys())

    @property
    def nrows(self):
        """Number of rows in the DataPanel."""
        if self.ncols == 0:
            return 0
        return self.data.nrows

    @property
    def ncols(self):
        """Number of rows in the DataPanel."""
        return self.data.ncols

    @property
    def shape(self):
        """Shape of the DataPanel (num_rows, num_columns)."""
        return self.nrows, self.ncols

    def add_column(
        self, name: str, data: AbstractColumn.Columnable, overwrite=False
    ) -> None:
        """Add a column to the DataPanel."""

        assert isinstance(
            name, str
        ), f"Column name must of type `str`, not `{type(name)}`."

        assert (name not in self.columns) or overwrite, (
            f"Column with name `{name}` already exists, "
            f"set `overwrite=True` to overwrite."
        )

        if name in self.columns:
            self.remove_column(name)

        column = AbstractColumn.from_data(data)

        assert len(column) == len(self), (
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

        logger.info(f"Removed column `{column}`.")

    @capture_provenance(capture_args=["axis"])
    def append(
        self,
        dp: DataPanel,
        axis: Union[str, int] = "rows",
        suffixes: Tuple[str] = None,
        overwrite: bool = False,
    ) -> DataPanel:
        """Append a batch of data to the dataset.

        `example_or_batch` must have the same columns as the dataset
        (regardless of what columns are visible).
        """
        return meerkat.concat(
            [self, dp], axis=axis, suffixes=suffixes, overwrite=overwrite
        )

    def _add_index(self):
        """Add an index to the dataset."""
        self.add_column("index", [str(i) for i in range(len(self))])

    def head(self, n: int = 5) -> DataPanel:
        """Get the first `n` examples of the DataPanel."""
        return self.lz[:n]

    def tail(self, n: int = 5) -> DataPanel:
        """Get the last `n` examples of the DataPanel."""
        return self.lz[-n:]

    def _get(self, index, materialize: bool = False):
        if isinstance(index, str):
            # str index => column selection (AbstractColumn)
            if index in self.columns:
                return self.data[index]
            raise KeyError(f"Column `{index}` does not exist.")

        elif isinstance(index, int):
            # int index => single row (dict)
            return {
                k: self.data[k]._get(index, materialize=materialize)
                for k in self.columns
            }

        # cases where `index` returns a datapanel
        index_type = None
        if isinstance(index, slice):
            # slice index => multiple row selection (DataPanel)
            index_type = "row"

        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            # tuple or list index => multiple row selection (DataPanel)
            if isinstance(index[0], str):
                index_type = "column"
            else:
                index_type = "row"

        elif isinstance(index, np.ndarray):
            if len(index.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(index.shape))
                )
            # numpy array index => multiple row selection (DataPanel)
            index_type = "row"

        elif torch.is_tensor(index):
            if len(index.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(index.shape))
                )
            # torch tensor index => multiple row selection (DataPanel)
            index_type = "row"

        elif isinstance(index, pd.Series):
            index_type = "row"

        elif isinstance(index, AbstractColumn):
            # column index => multiple row selection (DataPanel)
            index_type = "row"

        else:
            raise TypeError("Invalid index type: {}".format(type(index)))

        if index_type == "column":
            if not set(index).issubset(self.columns):
                missing_cols = set(index) - set(self.columns)
                raise KeyError(f"DataPanel does not have columns {missing_cols}")

            dp = self._clone(data=self.data[index])
            return dp
        elif index_type == "row":  # pragma: no cover
            return self._clone(
                data=self.data.apply("_get", index=index, materialize=materialize)
            )

    # @capture_provenance(capture_args=[])
    def __getitem__(self, index):
        return self._get(index, materialize=True)

    def __setitem__(self, index, value):
        self.add_column(name=index, data=value, overwrite=True)

    @property
    def has_index(self) -> bool:
        """Check if the dataset has an index column."""
        if self.columns:
            return "index" in self.columns
        # Just return True if the dataset is empty
        return True

    def consolidate(self):
        self.data.consolidate()

    @classmethod
    def from_huggingface(cls, *args, **kwargs):
        """Load a Huggingface dataset as a DataPanel.

        Use this to replace `datasets.load_dataset`, so

        >>> dict_of_datasets = datasets.load_dataset('boolq')

        becomes

        >>> dict_of_datapanels = DataPanel.from_huggingface('boolq')
        """
        # Load the dataset
        dataset = datasets.load_dataset(*args, **kwargs)

        if isinstance(dataset, dict):
            return dict(
                map(
                    lambda t: (t[0], cls(t[1])),
                    dataset.items(),
                )
            )
        else:
            return cls(dataset)

    @classmethod
    @capture_provenance()
    def from_jsonl(
        cls,
        json_path: str,
    ) -> DataPanel:
        """Load a dataset from a .jsonl file on disk, where each line of the
        json file consists of a single example."""
        with open(json_path) as f:
            data = {k: [] for k in json.loads(f.readline())}
        # Load the .jsonl file
        with open(json_path) as f:
            for line in f:
                line = json.loads(line)
                for k in data:
                    data[k].append(line[k])

        return cls(
            data=data,
        )

    @classmethod
    @capture_provenance()
    def from_batch(
        cls,
        batch: Batch,
    ) -> DataPanel:
        """Convert a batch to a Dataset."""
        return cls(batch)

    @classmethod
    @capture_provenance()
    def from_batches(
        cls,
        batches: Sequence[Batch],
    ) -> DataPanel:
        """Convert a list of batches to a dataset."""

        return cls.from_batch(
            tz.merge_with(
                tz.compose(list, tz.concat),
                *batches,
            ),
        )

    @classmethod
    @capture_provenance()
    def from_dict(
        cls,
        d: Dict,
    ) -> DataPanel:
        """Convert a dictionary to a dataset.

        Alias for Dataset.from_batch(..).
        """
        return cls.from_batch(
            batch=d,
        )

    @classmethod
    @capture_provenance()
    def from_pandas(
        cls,
        df: pd.DataFrame,
    ):
        """Create a Dataset from a pandas DataFrame."""
        # column names must be str in meerkat
        df = df.rename(mapper=str, axis="columns")
        return cls.from_batch(
            df.to_dict("series"),
        )

    @classmethod
    @capture_provenance()
    def from_arrow(
        cls,
        table: pa.Table,
    ):
        """Create a Dataset from a pandas DataFrame."""
        from meerkat.block.arrow_block import ArrowBlock
        from meerkat.columns.arrow_column import ArrowArrayColumn

        block_views = ArrowBlock.from_block_data(table)
        return cls.from_batch(
            {view.block_index: ArrowArrayColumn(view) for view in block_views}
        )

    @classmethod
    @capture_provenance(capture_args=["filepath"])
    def from_csv(cls, filepath: str, *args, **kwargs):
        """Create a Dataset from a csv file.

        Args:
            filepath (str): The file path or buffer to load from.
                Same as :func:`pandas.read_csv`.
            *args: Argument list for :func:`pandas.read_csv`.
            **kwargs: Keyword arguments for :func:`pandas.read_csv`.

        Returns:
            DataPanel: The constructed datapanel.
        """
        return cls.from_pandas(pd.read_csv(filepath, *args, **kwargs))

    @classmethod
    @capture_provenance()
    def from_feather(
        cls,
        path: str,
    ):
        """Create a Dataset from a feather file."""
        return cls.from_batch(
            pd.read_feather(path).to_dict("list"),
        )

    @capture_provenance()
    def to_pandas(self) -> pd.DataFrame:
        """Convert a Dataset to a pandas DataFrame."""
        return pd.DataFrame(
            {
                name: column.to_pandas().reset_index(drop=True)
                for name, column in self.items()
            }
        )

    def to_jsonl(self, path: str) -> None:
        """Save a Dataset to a jsonl file."""
        with jsonlines.open(path, mode="w") as writer:
            for example in self:
                writer.write(example)

    def _get_collate_fns(self, columns: Iterable[str] = None):
        columns = self.data.keys() if columns is None else columns
        return {name: self.data[name].collate for name in columns}

    def _collate(self, batch: List):
        batch = tz.merge_with(list, *batch)
        column_to_collate = self._get_collate_fns(batch.keys())
        new_batch = {}
        for name, values in batch.items():
            new_batch[name] = column_to_collate[name](values)
        dp = self._clone(data=new_batch)
        return dp

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
        from meerkat.columns.lambda_column import LambdaColumn

        for name, column in self.items():
            if isinstance(column, (CellColumn, LambdaColumn)) and materialize:
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
                self[batch_columns] if materialize else self[batch_columns].lz,
                sampler=batch_indices,
                batch_size=None,
                batch_sampler=None,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )

        if cell_columns:
            dp = self[cell_columns] if not shuffle else self[cell_columns].lz[indices]
            cell_dl = torch.utils.data.DataLoader(
                dp if materialize else dp.lz,
                batch_size=batch_size,
                collate_fn=self._collate,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )

        if batch_columns and cell_columns:
            for cell_batch, batch_batch in zip(cell_dl, batch_dl):
                yield self._clone(data={**cell_batch.data, **batch_batch.data})
        elif batch_columns:
            for batch_batch in batch_dl:
                yield batch_batch
        elif cell_columns:
            for cell_batch in cell_dl:
                yield cell_batch

    @capture_provenance(capture_args=["with_indices"])
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
    ) -> DataPanel:
        """Update the columns of the dataset."""
        # TODO(karan): make this fn go faster
        # most of the time is spent on the merge, speed it up further

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return self

        # Get some information about the function
        dp = self[input_columns] if input_columns is not None else self
        function_properties = dp._inspect_function(
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
        new_dp = self.view()

        # Calculate the values for the new columns using a .map()
        output = new_dp.map(
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
            if col == "index":
                continue
            new_dp.add_column(col, vals, overwrite=True)

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_dp.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")

        return new_dp

    @capture_provenance()
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        output_type: Union[type, Dict[str, type]] = None,
        mmap: bool = False,
        mmap_path: str = None,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> Optional[Union[Dict, List, AbstractColumn]]:
        input_columns = self.columns if input_columns is None else input_columns
        dp = self[input_columns]
        return super(DataPanel, dp).map(
            function=function,
            with_indices=with_indices,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_workers=num_workers,
            output_type=output_type,
            mmap=mmap,
            materialize=materialize,
            pbar=pbar,
            **kwargs,
        )

    @capture_provenance(capture_args=["function"])
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
    ) -> Optional[DataPanel]:
        """Filter operation on the DataPanel."""

        # Return if `self` has no examples
        if not len(self):
            logger.info("DataPanel empty, returning None.")
            return None

        # Get some information about the function
        dp = self[input_columns] if input_columns is not None else self
        function_properties = dp._inspect_function(
            function,
            with_indices,
            is_batched_fn=is_batched_fn,
            materialize=materialize,
            **kwargs,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new DataPanel will be returned.")
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

        # filter returns a new datapanel
        return self.lz[indices]

    def merge(
        self,
        right: meerkat.DataPanel,
        how: str = "inner",
        on: Union[str, List[str]] = None,
        left_on: Union[str, List[str]] = None,
        right_on: Union[str, List[str]] = None,
        sort: bool = False,
        suffixes: Sequence[str] = ("_x", "_y"),
        validate=None,
        keep_indexes: bool = False,
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
            keep_indexes=keep_indexes,
        )

    def items(self):
        for name in self.columns:
            yield name, self.data[name]

    def keys(self):
        return self.columns

    def values(self):
        for name in self.columns:
            yield self.data[name]

    @classmethod
    def read(
        cls,
        path: str,
        *args,
        **kwargs,
    ) -> DataPanel:
        """Load a DataPanel stored on disk."""

        # Load the metadata
        metadata = dict(
            yaml.load(open(os.path.join(path, "meta.yaml")), Loader=MeerkatLoader)
        )

        state = dill.load(open(os.path.join(path, "state.dill"), "rb"))
        dp = cls.__new__(cls)
        dp._set_state(state)

        # Load the the manager
        mgr_dir = os.path.join(path, "mgr")
        if os.path.exists(mgr_dir):
            data = BlockManager.read(mgr_dir, **kwargs)
        else:
            # backwards compatability to pre-manager datapanels
            data = {
                name: dtype.read(os.path.join(path, "columns", name), *args, **kwargs)
                for name, dtype in metadata["column_dtypes"].items()
            }

        dp._set_data(data)

        return dp

    def write(
        self,
        path: str,
    ) -> None:
        """Save a DataPanel to disk."""
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the DataPanel state
        state = self._get_state()

        # Get the metadata
        metadata = {
            "dtype": type(self),
            "column_dtypes": {name: type(col) for name, col in self.data.items()},
            "len": len(self),
        }

        # write the block manager
        mgr_dir = os.path.join(path, "mgr")
        self.data.write(mgr_dir)

        # Write the state
        state_path = os.path.join(path, "state.dill")
        dill.dump(state, open(state_path, "wb"))

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {}

    def _view_data(self) -> object:
        return self.data.view()

    def _copy_data(self) -> object:
        return self.data.copy()

    def __finalize__(self, *args, **kwargs):
        return self
