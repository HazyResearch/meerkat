"""DataPanel class."""
from __future__ import annotations

import logging
import os
import pathlib
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cytoolz as tz
import datasets
import dill
import numpy as np
import pandas as pd
import torch
import ujson as json
import yaml
from datasets import DatasetInfo, NamedSplit
from datasets.arrow_dataset import DatasetInfoMixin
from jsonlines import jsonlines

import mosaic
from mosaic.columns.abstract import AbstractColumn
from mosaic.columns.cell_column import CellColumn
from mosaic.mixins.copying import DataPanelCopyMixin
from mosaic.mixins.inspect_fn import FunctionInspectorMixin
from mosaic.mixins.mapping import MappableMixin
from mosaic.mixins.materialize import MaterializationMixin
from mosaic.mixins.state import StateDictMixin
from mosaic.tools.identifier import Identifier
from mosaic.tools.utils import convert_to_batch_fn, recmerge

logger = logging.getLogger(__name__)

Example = Dict
Batch = Dict[str, Union[List, AbstractColumn]]
BatchOrDataset = Union[Batch, "DataPanel"]


class DataPanel(
    DatasetInfoMixin,
    DataPanelCopyMixin,
    FunctionInspectorMixin,
    MappableMixin,
    MaterializationMixin,
    StateDictMixin,
):
    """Mosaic DataPanel class."""

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "mosaic/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    def __init__(
        self,
        *args,
        identifier: Identifier = None,
        column_names: List[str] = None,
        info: DatasetInfo = None,
        split: Optional[NamedSplit] = None,
        **kwargs,
    ):
        # TODO(karan, sabri): copy columns when they're passed in and prevent users
        #  from setting visible_rows inside columns that belong to a datapanel

        logger.debug("Creating DataPanel.")

        # Data is a dictionary of columns
        self._data = {}

        # Single argument
        if len(args) == 1:
            assert column_names is None, "Don't pass in column_names."
            # The data is passed in
            data = args[0]

            # `data` is a dictionary
            if isinstance(data, dict) and len(data):
                data = self._create_columns(data)
                self._assert_columns_all_equal_length(data)
                self._data = data

            # `data` is a list
            elif isinstance(data, list) and len(data):
                # Transpose the list of dicts to a dict of lists i.e. a batch
                data = tz.merge_with(list, *data)
                # Assert all columns are the same length
                data = self._create_columns(data)
                self._assert_columns_all_equal_length(data)
                self._data = data

            # `data` is a datasets.Dataset
            elif isinstance(data, datasets.Dataset):
                self._data = self._create_columns(data[:])
                info, split = data.info, data.split

        # No argument
        elif len(args) == 0:

            # Use column_names to setup the data dictionary
            if column_names:
                self._check_columns_unique(column_names)
                self._data = {k: [] for k in column_names}

        # Setup the DatasetInfo
        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)

        # Create attributes for all columns and visible columns
        self.all_columns = list(self._data.keys())
        self._visible_columns = None

        # Create an identifier
        # TODO(Sabri): make _autobuild_identifier more informative
        self._identifier = Identifier(
            self._autobuild_identifier() if not identifier else identifier
        )

        # Create logging directory
        self._create_logdir()

        self._initialize_state()

        # TODO(Sabri): fix add_index for new datset
        # Add an index to the dataset
        if not self.has_index:
            self._add_index()

    @classmethod
    def _create_columns(cls, name_to_data: Dict[str, AbstractColumn.Columnable]):
        new_data = {}
        for column_name, data in name_to_data.items():
            new_data[column_name] = AbstractColumn.from_data(data=data)

        return new_data

    def _repr_pandas_(self):
        return pd.DataFrame(
            {
                f"{k} ({v.__class__.__name__})": v._repr_pandas_()
                for k, v in self.items()
            }
        )

    def _repr_html_(self):
        return self._repr_pandas_()._repr_html_()

    def __repr__(self):
        return f"{self.__class__.__name__}" f"(num_rows: {self.num_rows})"

    def __len__(self):
        # If only a subset of rows are visible
        if len(self.visible_columns) == 0:
            return 0
        return len(self[self.visible_columns[0]])

    def __contains__(self, item):
        return item in self.visible_columns

    def full_length(self):
        # If there are columns, full_length of any column, since they must be same size
        if self.column_names:
            return self._data[self.column_names[0]].full_length()
        return 0

    @property
    def column_names(self):
        """Column names in the dataset."""
        return self.all_columns

    @property
    def columns(self):
        """Column names in the dataset."""
        return self.column_names

    @property
    def num_rows(self):
        """Number of rows in the dataset."""
        return len(self)

    @property
    def shape(self):
        """Shape of the dataset (num_rows, num_columns)."""
        return self.num_rows, len(self.columns)

    @classmethod
    def _assert_columns_all_equal_length(cls, batch: Batch):
        """Check that all columns have the same length so that the data is
        tabular."""
        assert cls._columns_all_equal_length(
            batch
        ), "All columns must have equal length."

    @classmethod
    def _columns_all_equal_length(cls, batch: Batch):
        """Check that all columns have the same length so that the data is
        tabular."""
        if len(set([len(v) for k, v in batch.items()])) == 1:
            return True
        return False

    def _check_columns_exist(self, columns: List[str]):
        """Check that every column in `columns` exists."""
        for col in columns:
            assert col in self.all_columns, f"{col} is not a valid column."

    def _check_columns_unique(self, columns: List[str]):
        """Checks that all columns are unique."""
        assert len(columns) == len(set(columns))

    def _initialize_state(self):
        """Dataset state initialization."""
        # Show all columns by default
        self.visible_columns = copy(self.all_columns)

        # Set the features
        self._set_features()

    @property
    def visible_columns(self):
        return self._visible_columns

    @visible_columns.setter
    def visible_columns(self, columns: Optional[Sequence[str]] = None):
        if columns is None:
            # do nothing, keep old visible columns
            return
        for c in columns:
            if c not in self.all_columns:
                raise ValueError(f"Trying to set nonexistant column {c} to visible.")

        self._visible_columns = copy(columns)
        if "index" not in self._visible_columns and "index" in self.all_columns:
            self._visible_columns.append("index")

    @contextmanager
    def format(self, columns: List[str] = None):
        """Context where only `columns` will be visible."""
        # Get the current format
        current_format = self.get_format()

        if columns:
            # View only `columns`
            self.set_format(columns)
        else:
            # Use all columns
            self.set_format(self.column_names)
        try:
            yield
        finally:
            # Reset the format back
            self.set_format(current_format)

    def get_format(self) -> List[str]:
        """Get the dataset format."""
        return self.visible_columns

    def set_format(self, columns: List[str]):
        """Set the dataset format.

        Only `columns` are visible after set_format is invoked.
        """
        # Check that the columns exist
        self._check_columns_exist(columns)
        # Set visible columns
        self.visible_columns = columns

    def reset_format(self):
        """Reset the dataset format.

        All columns are visible.
        """
        # All columns are visible
        self.visible_columns = self.all_columns

    def _example_or_batch_to_batch(
        self, example_or_batch: Union[Example, Batch]
    ) -> Batch:

        # Check if example_or_batch is a batch
        is_batch = all(
            [isinstance(v, List) for v in example_or_batch.values()]
        ) and self._columns_all_equal_length(example_or_batch)

        # Convert to a batch if not
        if not is_batch:
            batch = {k: [v] for k, v in example_or_batch.items()}
        else:
            batch = example_or_batch

        return batch

    @classmethod
    def _merge_batch_and_output(cls, batch: Batch, output: Batch):
        """Merge an output during .map() into a batch."""
        combined = batch
        for k in output.keys():
            if k not in batch:
                combined[k] = output[k]
            else:
                if isinstance(batch[k][0], dict) and isinstance(output[k][0], dict):
                    combined[k] = [
                        recmerge(b_i, o_i) for b_i, o_i in zip(batch[k], output[k])
                    ]
                else:
                    combined[k] = output[k]
        return combined

    @classmethod
    def _mask_batch(cls, batch: Batch, boolean_mask: List[bool]):
        """Remove elements in `batch` that are masked by `boolean_mask`."""
        return {
            k: [e for i, e in enumerate(v) if boolean_mask[i]] for k, v in batch.items()
        }

    @property
    def identifier(self):
        """Identifier."""
        return self._identifier

    def _set_features(self):
        """Set the features of the dataset."""
        with self.format():
            self.info.features = None  # Features.from_arrow_schema(
            #     pa.Table.from_pydict(
            #         self[:1],
            #     ).schema
            # )

    def add_column(
        self, name: str, data: AbstractColumn.Columnable, overwrite=False
    ) -> None:
        """Add a column to the dataset."""

        assert isinstance(
            name, str
        ), f"Column name must of type `str`, not `{type(name)}`."

        assert (name not in self.all_columns) or overwrite, (
            f"Column with name `{name}` already exists, "
            f"set `overwrite=True` to overwrite."
        )

        column = AbstractColumn.from_data(data)

        assert len(column) == len(self), (
            f"`add_column` failed. "
            f"Values length {len(column)} != dataset length {len(self)}."
        )

        # Add the column
        self._data[name] = column

        if name not in self.all_columns:
            self.all_columns.append(name)
            self.visible_columns.append(name)

        # Set features
        self._set_features()

        logger.info(f"Added column `{name}` with length `{len(column)}`.")

    def remove_column(self, column: str) -> None:
        """Remove a column from the dataset."""
        assert column in self.all_columns, f"Column `{column}` does not exist."

        # Remove the column
        del self._data[column]
        self.all_columns = [col for col in self.all_columns if col != column]
        self.visible_columns = [col for col in self.visible_columns if col != column]

        # Set features
        self._set_features()

        logger.info(f"Removed column `{column}`.")

    def select_columns(self, columns: List[str]) -> Batch:
        """Select a subset of columns."""
        for col in columns:
            assert col in self._data
        return tz.keyfilter(lambda k: k in columns, self._data)

    def append(
        self,
        dp: DataPanel,
        axis: Union[str, int] = "rows",
        suffixes: Tuple[str] = None,
        overwrite: bool = False,
    ) -> None:
        """Append a batch of data to the dataset.

        `example_or_batch` must have the same columns as the dataset
        (regardless of what columns are visible).
        """
        if axis == 0 or axis == "rows":
            # append new rows
            return mosaic.concat([self, dp], axis="rows")
        elif axis == 1 or axis == "columns":
            # append new columns
            if len(dp) != len(self):
                raise ValueError(
                    "Can only append DataPanels along axis 1 (columns) if they have the"
                    f"same length. {len(self)} != {len(dp)}"
                )

            shared = set(dp.visible_columns).intersection(set(self.visible_columns))
            if not overwrite and shared:
                if suffixes is None:
                    raise ValueError()
                left_suf, right_suf = suffixes
                data = {
                    **{k + left_suf if k in shared else k: v for k, v in self.items()},
                    **{k + right_suf if k in shared else k: v for k, v in dp.items()},
                }
            else:
                data = {**dict(self.items()), **dict(dp.items())}

            return DataPanel.from_batch(data)
        else:
            raise ValueError("DataPanel `axis` must be either 0 or 1.")

    def _add_index(self):
        """Add an index to the dataset."""
        self.add_column("index", [str(i) for i in range(len(self))])

    def head(self, n: int = 5) -> DataPanel:
        """Get the first `n` examples of the DataPanel."""
        return self.lz[:n]

    def tail(self, n: int = 5) -> DataPanel:
        """Get the last `n` examples of the DataPanel."""
        return self.lz[-n:]

    def _create_logdir(self):
        """Create and assign a directory for logging this dataset's files."""
        if self.identifier.name == "RGDataset":
            # TODO(karan): handle temporarily constructed datasets differently
            self.logdir /= str(self.identifier)
            self.logdir.mkdir(parents=True, exist_ok=True)
        else:
            self.logdir /= str(self.identifier)
            self.logdir.mkdir(parents=True, exist_ok=True)

    def _autobuild_identifier(self) -> Identifier:
        """Automatically build an identifier for the dataset using available
        information."""
        # Look for a name, otherwise assign a default
        _name = self.info.builder_name if self.info.builder_name else "RGDataset"

        # Check for split, version information
        split = str(self.split) if self.split else None
        version = str(self.version) if self.version else None

        # Add all available information to kwargs dict
        kwargs = {}
        if split:
            kwargs["split"] = split
        if version:
            kwargs["version"] = version

        # Create identifier
        return Identifier(_name=_name, **kwargs)

    def _get(self, index, materialize: bool = False):
        if isinstance(index, int):
            # int index => single row (dict)
            return {
                k: self._data[k]._get(index, materialize=materialize)
                for k in self.visible_columns
            }

        elif isinstance(index, str):
            # str index => column selection (AbstractColumn)
            if index in self.column_names:
                return self._data[index]
            raise AttributeError(f"Column {index} does not exist.")

        # cases where `index` returns a datapanel
        elif isinstance(index, slice):
            # slice index => multiple row selection (DataPanel)
            return DataPanel.from_batch(
                {
                    k: self._data[k]._get(index, materialize=materialize)
                    for k in self.visible_columns
                }
            )

        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            # tuple or list index => multiple row selection (DataPanel)
            if isinstance(index[0], str):
                if not set(index).issubset(self.visible_columns):
                    missing_cols = set(self.visible_columns) - set(index)
                    raise ValueError(f"DataPanel does not have columns {missing_cols}")
                dp = self.view()
                dp.visible_columns = index
                return dp

            return DataPanel.from_batch(
                {
                    k: self._data[k]._get(index, materialize=materialize)
                    for k in self.visible_columns
                }
            )
        elif isinstance(index, np.ndarray):
            if len(index.shape) != 1:
                raise ValueError(
                    "Index must have 1 axis, not {}".format(len(index.shape))
                )
            # numpy array index => multiple row selection (DataPanel)
            return DataPanel.from_batch(
                {
                    k: self._data[k]._get(index, materialize=materialize)
                    for k in self.visible_columns
                }
            )
        elif isinstance(index, AbstractColumn):
            # column index => multiple row selection (DataPanel)
            return DataPanel.from_batch(
                {
                    k: self._data[k]._get(index, materialize=materialize)
                    for k in self.visible_columns
                }
            )
        else:
            raise TypeError("Invalid index type: {}".format(type(index)))

    def __getitem__(self, index):
        return self._get(index, materialize=True)

    def get(self, column, value=None):
        if column in self:
            return self[column]
        return value

    def __setitem__(self, index, value):
        self.add_column(name=index, data=value, overwrite=True)

    @property
    def has_index(self) -> bool:
        """Check if the dataset has an index column."""
        if self.column_names:
            return "index" in self.column_names
        # Just return True if the dataset is empty
        return True

    @classmethod
    def uncached_batch(cls, batch: Batch, copy=True) -> Batch:
        """Return batch with the "cache" and "slices" columns removed."""
        return tz.keyfilter(
            lambda k: k not in ["cache", "slices"], deepcopy(batch) if copy else batch
        )

    @classmethod
    def uncached_example(cls, example: Dict, copy=True) -> Dict:
        """Return example with the "cache" and "slices" columns removed."""
        return tz.keyfilter(
            lambda k: k not in ["cache", "slices"],
            deepcopy(example) if copy else example,
        )

    @classmethod
    def load_huggingface(cls, *args, **kwargs):
        """Load a Huggingface dataset as a DataPanel.

        Use this to replace `datasets.load_dataset`, so

        >>> dict_of_datasets = datasets.load_dataset('boolq')

        becomes

        >>> dict_of_datapanels = DataPanel.load_huggingface('boolq')
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
    def from_columns(
        cls,
        columns: Dict[str, AbstractColumn],
        identifier: Identifier = None,
    ) -> DataPanel:
        """Create a Dataset from a dict of columns."""
        return cls(
            columns,
            identifier=identifier,
        )

    @classmethod
    def from_jsonl(
        cls,
        json_path: str,
        identifier: Identifier = None,
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
            data,
            identifier=identifier
            if identifier
            else Identifier("Jsonl", jsonl=json_path),
        )

    @classmethod
    def from_batch(
        cls,
        batch: Batch,
        identifier: Identifier = None,
    ) -> DataPanel:
        """Convert a batch to a Dataset."""
        return cls(batch, identifier=identifier)

    @classmethod
    def from_batches(
        cls,
        batches: Sequence[Batch],
        identifier: Identifier = None,
    ) -> DataPanel:
        """Convert a list of batches to a dataset."""

        return cls.from_batch(
            tz.merge_with(
                tz.compose(list, tz.concat),
                *batches,
            ),
            identifier=identifier,
        )

    @classmethod
    def from_dict(
        cls,
        d: Dict,
        identifier: Identifier = None,
    ) -> DataPanel:
        """Convert a dictionary to a dataset.

        Alias for Dataset.from_batch(..).
        """
        return cls.from_batch(
            batch=d,
            identifier=identifier,
        )

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        identifier: Identifier = None,
    ):
        """Create a Dataset from a pandas DataFrame."""
        return cls.from_batch(
            df.to_dict("series"),
            identifier=identifier,
        )

    @classmethod
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
    def from_feather(
        cls,
        path: str,
        identifier: Identifier = None,
    ):
        """Create a Dataset from a feather file."""
        return cls.from_batch(
            pd.read_feather(path).to_dict("list"),
            identifier=Identifier("Feather", path=path)
            if not identifier
            else identifier,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Convert a Dataset to a pandas DataFrame."""
        return pd.DataFrame({name: column.to_pandas() for name, column in self.items()})

    def to_jsonl(self, path: str) -> None:
        """Save a Dataset to a jsonl file."""
        with jsonlines.open(path, mode="w") as writer:
            for example in self:
                writer.write(example)

    def _get_collate_fns(self, columns: Iterable[str] = None):
        columns = self._data.keys() if columns is None else columns
        return {name: self._data[name].collate for name in columns}

    def _collate(self, batch: List):
        batch = tz.merge_with(list, *batch)
        column_to_collate = self._get_collate_fns(batch.keys())
        new_batch = {}
        for name, values in batch.items():
            new_batch[name] = column_to_collate[name](values)
        dp = DataPanel.from_batch(new_batch)
        return dp

    @staticmethod
    def _convert_to_batch_fn(
        function: Callable, with_indices: bool, materialize: bool = True
    ) -> callable:
        return convert_to_batch_fn(
            function=function, with_indices=with_indices, materialize=materialize
        )

    def batch(
        self,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        materialize: bool = True,
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
        for name, column in self.items():
            if isinstance(column, CellColumn):
                cell_columns.append(name)
            else:
                batch_columns.append(name)

        if batch_columns:
            batch_indices = []
            indices = np.arange(len(self))
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
            cell_dl = torch.utils.data.DataLoader(
                self[cell_columns] if materialize else self[cell_columns].lz,
                batch_size=batch_size,
                collate_fn=self._collate,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )

        if batch_columns and cell_columns:
            for cell_batch, batch_batch in zip(cell_dl, batch_dl):
                yield DataPanel.from_batch({**cell_batch._data, **batch_batch._data})
        elif batch_columns:
            for batch_batch in batch_dl:
                yield batch_batch
        elif cell_columns:
            for cell_batch in cell_dl:
                yield cell_batch

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        remove_columns: Optional[List[str]] = None,
        num_workers: int = 0,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> DataPanel:
        """Update the columns of the dataset."""
        # TODO(karan): make this fn go faster
        # most of the time is spent on the merge, speed it up further

        # Return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return self

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return self

        # Get some information about the function
        with self.format(input_columns):
            function_properties = self._inspect_function(
                function, with_indices, is_batched_fn, materialize=materialize
            )
            assert (
                function_properties.dict_output
            ), f"`function` {function} must return dict."

        if not is_batched_fn:
            # Convert to a batch function
            function = convert_to_batch_fn(
                function, with_indices=with_indices, materialize=materialize
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
            input_columns=input_columns,
            materialize=materialize,
            pbar=pbar,
        )

        # Add new columns for the update
        for col, vals in output._data.items():
            if col == "index":
                continue
            new_dp.add_column(col, vals, overwrite=True)

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_dp.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")

        return new_dp

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: int = 0,
        output_type: type = None,
        mmap: bool = False,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ) -> Optional[Union[Dict, List, AbstractColumn]]:
        input_columns = self.visible_columns if input_columns is None else input_columns
        with self.format(input_columns):
            return super().map(
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

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("DataPanel empty, returning None.")
            return None

        # Get some information about the function
        with self.format(input_columns):
            function_properties = self._inspect_function(
                function,
                with_indices,
                is_batched_fn=is_batched_fn,
                materialize=materialize,
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
        )
        indices = np.where(outputs)[0]

        # filter returns a new datapanel
        new_datapanel = self.view()
        for column in new_datapanel._data.values():
            column.visible_rows = indices

        return new_datapanel

    def merge(
        self,
        right: mosaic.DataPanel,
        how: str = "inner",
        on: Union[str, List[str]] = None,
        left_on: Union[str, List[str]] = None,
        right_on: Union[str, List[str]] = None,
        sort: bool = False,
        suffixes: Sequence[str] = ("_x", "_y"),
        validate=None,
        keep_indexes: bool = False,
    ):
        from mosaic import merge

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
        for name in self.visible_columns:
            yield name, self._data[name]

    def keys(self):
        return self.visible_columns

    def values(self):
        for name in self.visible_columns:
            yield self._data[name]

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
            yaml.load(open(os.path.join(path, "meta.yaml")), Loader=yaml.FullLoader)
        )

        state = dill.load(open(os.path.join(path, "state.dill"), "rb"))

        # Load the columns
        if not metadata["write_together"]:
            data = {
                name: dtype.read(os.path.join(path, "columns", name), *args, **kwargs)
                for name, dtype in metadata["column_dtypes"].items()
            }
            state["_data"] = data

        # Create a DataPanel from the loaded state
        datapanel = cls.from_state(state)

        return datapanel

    def write(
        self,
        path: str,
        write_together: bool = False,
    ) -> None:
        """Save a DataPanel to disk."""
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the DataPanel state
        state = self.get_state()

        # Get the metadata
        metadata = {
            "dtype": type(self),
            "column_dtypes": {name: type(col) for name, col in self._data.items()},
            "len": len(self),
            "write_together": write_together,
        }

        if not write_together:
            if "_data" not in state:
                raise ValueError(
                    "DataPanel's state must include `_data` when using "
                    "`write_together=False`."
                )
            del state["_data"]

            # Create a directory for the columns at `path`
            columns_path = os.path.join(path, "columns")
            os.makedirs(columns_path, exist_ok=True)

            # Save each column in the DataPanel separately
            for name, column in self._data.items():
                column.write(os.path.join(columns_path, name))

        # Write the state
        state_path = os.path.join(path, "state.dill")
        dill.dump(state, open(state_path, "wb"))

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def from_state(cls, state: Dict, *args, **kwargs) -> DataPanel:
        datapanel = super(DataPanel, cls).from_state(state, *args, **kwargs)
        datapanel._create_logdir()
        datapanel._set_features()
        return datapanel

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "_identifier",
            "_data",
            "all_columns",
            "_visible_columns",
            "_info",
            "_split",
        }
