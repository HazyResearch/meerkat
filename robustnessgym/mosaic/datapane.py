"""DataPane class."""
from __future__ import annotations

import json
import logging
import os
import pathlib
from collections import defaultdict
from contextlib import contextmanager
from copy import copy, deepcopy
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import cytoolz as tz
import datasets
import dill
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import DatasetInfo, NamedSplit
from datasets.arrow_dataset import DatasetInfoMixin
from jsonlines import jsonlines
from tqdm.auto import tqdm

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import convert_to_batch_fn, recmerge
from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.columns.abstract import AbstractColumn
from robustnessgym.mosaic.columns.cell_column import CellColumn
from robustnessgym.mosaic.columns.list_column import ListColumn
from robustnessgym.mosaic.columns.numpy_column import NumpyArrayColumn
from robustnessgym.mosaic.mixins.copying import CopyMixin
from robustnessgym.mosaic.mixins.inspect_fn import FunctionInspectorMixin
from robustnessgym.mosaic.mixins.state import StateDictMixin
from robustnessgym.mosaic.writers.numpy_writer import NumpyMemmapWriter

logger = logging.getLogger(__name__)

Example = Dict
Batch = Dict[str, Union[List, AbstractColumn]]
BatchOrDataset = Union[Batch, "DataPane"]
# TODO(sabri): change the name of this!
Columnable = Union[AbstractColumn, List, np.ndarray, pd.Series]


class DataPane(
    DatasetInfoMixin,
    CopyMixin,
    FunctionInspectorMixin,
    StateDictMixin,
):
    """Mosaic DataPane class."""

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "robustnessgym/mosaic/"

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

        logger.debug("Creating DataPane.")

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
                self._data = {k: [] for k in column_names}

        # Setup the DatasetInfo
        info = info.copy() if info is not None else DatasetInfo()
        DatasetInfoMixin.__init__(self, info=info, split=split)

        # Create attributes for all columns and visible columns
        self.all_columns = list(self._data.keys())
        self.visible_columns = None

        # Create attributes for visible rows
        self.visible_rows = None

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
    def _create_column(cls, data: Columnable):
        # TODO (sabri): put in a registry
        if isinstance(data, AbstractColumn):
            return data
        elif isinstance(data, np.ndarray):
            return NumpyArrayColumn(data)
        elif isinstance(data, pd.Series):
            return NumpyArrayColumn(data.values)
        elif len(data) != 0 and isinstance(data[0], AbstractCell):
            return CellColumn(data)
        else:
            return ListColumn(data)

    @classmethod
    def _create_columns(cls, name_to_data: Dict[str, Columnable]):
        new_data = {}
        for column_name, data in name_to_data.items():
            new_data[column_name] = cls._create_column(data=data)

        return new_data

    def __repr__(self):
        return f"{self.__class__.__name__}" f"(num_rows: {self.num_rows})"

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        # If there are columns, len of any column
        if self.column_names:
            return len(self._data[self.column_names[0]])
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

    def _initialize_state(self):
        """Dataset state initialization."""
        # Show all columns by default
        self.visible_columns = copy(self.all_columns)

        # Show all rows by default
        self.visible_rows = None

        # Set the features
        self._set_features()

    def set_visible_rows(self, indices: Optional[Sequence]):
        """Set the visible rows in the dataset."""
        if indices is None:
            self.visible_rows = None
        else:
            if len(indices):
                assert min(indices) >= 0 and max(indices) < len(self), (
                    f"Ensure min index {min(indices)} >= 0 and "
                    f"max index {max(indices)} < {len(self)}."
                )
            if self.visible_rows is not None:
                self.visible_rows = self.visible_rows[np.array(indices, dtype=int)]
            else:
                self.visible_rows = np.array(indices, dtype=int)

    def reset_visible_rows(self):
        """Reset to make all rows visible."""
        self.visible_rows = None

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

    def add_column(self, name: str, data: Columnable, overwrite=False) -> None:
        """Add a column to the dataset."""

        assert (name not in self.all_columns) or overwrite, (
            f"Column with name `{name}` already exists, "
            f"set `overwrite=True` to overwrite."
        )

        column = self._create_column(data)

        assert len(column) == len(self), (
            f"`add_column` failed. "
            f"Values length {len(column)} != dataset length {len(self)}."
        )

        # Add the column
        self._data[name] = column
        self.all_columns.append(name)
        self.visible_columns.append(name)

        # Set features
        self._set_features()

        logger.info(f"Added column `{name}` with length `{len(column)}`.")

    def remove_column(self, column: str) -> None:
        """Remove a column from the dataset."""
        assert column in self.all_columns, f"Column `{column}` does not exist."

        if self.visible_rows is not None:
            # Materialize the data
            self._materialize()

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
        example_or_batch: Union[Example, Batch],
    ) -> None:
        """Append a batch of data to the dataset.

        `example_or_batch` must have the same columns as the dataset
        (regardless of what columns are visible).
        """
        self._dataset.append(example_or_batch)

    def _add_index(self):
        """Add an index to the dataset."""
        self.add_column("index", [str(i) for i in range(len(self))])

    def head(self, n: int, columns: List[str] = None):
        """View the first `n` examples of the dataset."""
        with self.format(columns):
            return pd.DataFrame(self[:n])

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

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        if isinstance(index, int) or isinstance(index, np.int):
            # int index => single row (dict)
            return {k: self._data[k][index] for k in self.visible_columns}

        elif isinstance(index, str):
            # str index => column selection (AbstractColumn)
            if index in self.column_names:
                if self.visible_rows is not None:
                    return [self._data[index][i] for i in self.visible_rows]
                return self._data[index]
            raise AttributeError(f"Column {index} does not exist.")

        # TODO(sabri): discuss with others returning a DataPane here
        elif isinstance(index, slice):
            # slice index => multiple row selection (DataPane)
            return {k: self._data[k][index] for k in self.visible_columns}
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            # tuple or list index => multiple row selection (DataPane)
            if isinstance(index[0], str):
                return DataPane.from_batch(
                    {k: self._data[k] for k in index if k in self.visible_columns}
                )
            return {k: self._data[k][index] for k in self.visible_columns}
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            # numpy array index => multiple row selection (DataPane)
            return {k: self._data[k][index] for k in self.visible_columns}
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def _remap_index(self, index):
        if isinstance(index, int):
            return self.visible_rows[index].item()
        elif isinstance(index, slice):
            return self.visible_rows[index].tolist()
        elif isinstance(index, str):
            return index
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            return self.visible_rows[index].tolist()
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            return self.visible_rows[index].tolist()
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

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
        """
        Load a Huggingface dataset as a DataPane.

        Use this to replace `datasets.load_dataset`, so

        >>> dict_of_datasets = datasets.load_dataset('boolq')

        becomes

        >>> dict_of_datapanes = DataPane.load_huggingface('boolq')
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
    ) -> DataPane:
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
    ) -> DataPane:
        """Load a dataset from a .jsonl file on disk, where each line of the
        json file consists of a single example."""

        # Load the .jsonl file
        with open(json_path) as f:
            data = [json.loads(line) for line in f]

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
    ) -> DataPane:
        """Convert a batch to a Dataset."""

        return cls(batch, identifier=identifier)

    @classmethod
    def from_batches(
        cls,
        batches: Sequence[Batch],
        identifier: Identifier = None,
    ) -> DataPane:
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
    ) -> DataPane:
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
        return pd.DataFrame(self[:])

    def to_jsonl(self, path: str) -> None:
        """Save a Dataset to a jsonl file."""
        with jsonlines.open(path, mode="w") as writer:
            for example in self:
                writer.write(example)

    def _get_collate_fns(self):
        return {name: column.collate for name, column in self._data.items()}

    def _collate(self, batch: List, column_to_collate):
        batch = tz.merge_with(list, *batch)
        new_batch = {}
        for name, values in batch.items():
            new_batch[name] = column_to_collate[name](values)

        return new_batch

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        num_workers: int = 4,
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
        column_to_collate = self._get_collate_fns()

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=partial(self._collate, column_to_collate=column_to_collate),
            drop_last=drop_last_batch,
            num_workers=num_workers,
            *args,
            **kwargs,
        )

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        # input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        num_workers: int = 4,
        **kwargs,
    ) -> DataPane:
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
        function_properties = self._inspect_function(function, with_indices, batched)
        assert (
            function_properties.dict_output
        ), f"`function` {function} must return dict."

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to batched function.")

        # Update always returns a new dataset
        logger.info("Running update, a new dataset will be returned.")
        if self.visible_rows is not None:
            # Run .map() to get updated batches and pass them into a new dataset
            new_dataset = DataPane(
                self.map(
                    (
                        lambda batch, indices: self._merge_batch_and_output(
                            batch, function(batch, indices)
                        )
                    )
                    if with_indices
                    else (
                        lambda batch: self._merge_batch_and_output(
                            batch, function(batch)
                        )
                    ),
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                )
            )
        else:
            if function_properties.updates_existing_column:
                # Copy the ._data dict with a reference to the actual columns
                new_dataset = self.copy()

                # Calculate the values for the updated columns using a .map()
                output = self.map(
                    (
                        lambda batch, indices:
                        # Only merge columns that get updated
                        self._merge_batch_and_output(
                            {
                                k: v
                                for k, v in batch.items()
                                if k in function_properties.existing_columns_updated
                            },
                            function(batch, indices),
                        )
                    )
                    if with_indices
                    else (
                        lambda batch:
                        # Only merge columns that get updated
                        self._merge_batch_and_output(
                            {
                                k: v
                                for k, v in batch.items()
                                if k in function_properties.existing_columns_updated
                            },
                            function(batch),
                        )
                    ),
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                )

                # Add new columns / overwrite existing columns for the update
                for col, vals in output.items():
                    new_dataset.add_column(col, vals, overwrite=True)
            else:
                # Copy the ._data dict with a reference to the actual columns
                new_dataset = self.copy()

                # Calculate the values for the new columns using a .map()
                output = new_dataset.map(
                    function=function,
                    with_indices=with_indices,
                    batched=True,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                # Add new columns for the update
                for col, vals in output.items():
                    new_dataset.add_column(col, vals)

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_dataset.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")

        return new_dataset

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 32,
        drop_last_batch: bool = False,
        num_workers: int = 4,
        output_type: type = None,
        mmap: bool = False,
        **kwargs,
    ) -> Optional[Union[Dict, List, AbstractColumn]]:
        """Apply a map over the dataset."""

        # Just return if `function` is None or `self` has no examples
        if function is None or not len(self):
            return None

        # Convert `input_columns` to a list
        if isinstance(input_columns, str):
            input_columns = [input_columns]

        # Convert to a batched function
        if not batched:
            function = convert_to_batch_fn(function, with_indices=with_indices)
            batched = True

        with self.format(input_columns):

            # Variable to store the outputs
            outputs = None

            # Create the batches
            batches = self.batch(batch_size, drop_last_batch, num_workers=num_workers)
            num_batches = (len(self) // batch_size) + (1 - int(drop_last_batch))

            # Run the map
            for i, batch in tqdm(enumerate(batches), total=num_batches):

                # Calculate the start and end indexes for the batch
                start_index = i * batch_size
                end_index = min(len(self), (i + 1) * batch_size)

                # Use the first batch for setup
                if i == 0:

                    # Get some information about the function
                    function_properties = self._inspect_function(
                        function,
                        with_indices,
                        batched,
                        batch,
                        range(start_index, end_index),
                    )

                    # Pull out information
                    output = function_properties.output
                    dtype = function_properties.output_dtype
                    shape = (len(self), *function_properties.output_shape)

                    # Setup for writing to a certain output column
                    if output_type == NumpyArrayColumn and mmap:
                        # Create the writer
                        writer = NumpyMemmapWriter()

                        # Construct the mmap file path
                        # TODO: how/where to store the files
                        path = self.logdir / f"{hash(function)}"

                        # Open the output writer
                        writer.open(str(path), dtype, shape=shape)

                    else:
                        # Create an empty dict or list for the outputs
                        outputs = (
                            defaultdict(list) if isinstance(output, Mapping) else []
                        )

                else:
                    # Run `function` on the batch
                    output = (
                        function(batch, range(start_index, end_index))
                        if with_indices
                        else function(batch)
                    )

                # Append the output
                if output is not None:
                    if isinstance(output, Mapping):
                        for k in output.keys():
                            outputs[k].append(output[k])
                    else:
                        # Store output
                        if output_type == NumpyArrayColumn and mmap:
                            # Store the output in the mmap file and flush
                            writer.write(output)
                        else:
                            # Extend outputs by default
                            outputs.append(output)

        # Check if we are returning a special output type
        if output_type == NumpyArrayColumn:
            if mmap:
                writer.flush()
                # TODO: Writers should have correspondence to their own columns
                return NumpyArrayColumn.read(
                    str(path),
                    mmap=mmap,
                    dtype=dtype,
                    shape=shape,
                )
            else:
                return NumpyArrayColumn(outputs)

        if outputs is None or not len(outputs):
            return None

        if isinstance(outputs, dict):
            return {k: self._concat_batches(v) for k, v in outputs.items()}
        return self._concat_batches(outputs)

    @staticmethod
    def _concat_batches(batches):
        first_batch = batches[0]
        if isinstance(first_batch, np.ndarray):
            return np.concatenate(batches, axis=0)
        elif isinstance(first_batch, torch.tensor):
            return torch.cat(batches, axis=0)
        elif isinstance(first_batch, list) or isinstance(first_batch, tuple):
            return tz.concat(batches)
        else:
            output = []
            for batch in batches:
                output.extend(batch)
            return output

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_workers: int = 4,
        **kwargs,
    ) -> Optional[DataPane]:
        """Filter operation on the DataPane."""

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("DataPane empty, returning None.")
            return None

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new DataPane will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_workers=num_workers,
        )
        indices = np.where(outputs)[0]

        # Reset the format to set visible columns for the filter
        with self.format():
            # Filter returns a new dataset
            new_datapane = self.copy()
            new_datapane.set_visible_rows(indices)

        return new_datapane

    @classmethod
    def read(
        cls,
        path: str,
        *args,
        **kwargs,
    ) -> DataPane:
        """Load a DataPane stored on disk."""

        # Load the metadata
        metadata = dict(
            yaml.load(open(os.path.join(path, "meta.yaml")), Loader=yaml.FullLoader)
        )

        # Load the columns
        if metadata["write_together"]:
            data = dill.load(open(os.path.join(path, "data.dill"), "rb"))
            data = {
                name: metadata["column_dtypes"][name].from_state(state, *args, **kwargs)
                for name, state in data.items()
            }
        else:
            data = {
                name: dtype.read(os.path.join(path, "columns", name), *args, **kwargs)
                for name, dtype in metadata["column_dtypes"].items()
            }

        # Create the state dict
        state = metadata["state"]
        state["_data"] = data

        # Create a DataPane from the loaded state
        datapane = cls.from_state(state)
        datapane._create_logdir()
        datapane._initialize_state()
        datapane.visible_rows = state["visible_rows"]

        return datapane

    def write(
        self,
        path: str,
        write_together: bool = False,
    ) -> None:
        """Save a DataPane to disk."""
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the DataPane state
        state = self.get_state()
        del state["_data"]

        # Get the metadata
        metadata = {
            "dtype": type(self),
            "column_dtypes": {name: type(col) for name, col in self._data.items()},
            "len": len(self),
            "write_together": write_together,
            "state": state,
        }

        if write_together:
            # Write the entire DataPane together
            data_path = os.path.join(path, "data.dill")

            # Save all the columns in a single dill file
            dill.dump(
                {name: col.get_state() for name, col in self._data.items()},
                open(data_path, "wb"),
            )

        else:
            # Create a directory for the columns at `path`
            columns_path = os.path.join(path, "columns")
            os.makedirs(columns_path, exist_ok=True)

            # Save each column in the DataPane separately
            for name, column in self._data.items():
                column.write(os.path.join(columns_path, name))

        # Save the metadata as a yaml file
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "_identifier",
            "_data",
            "all_columns",
            "visible_rows",
            "_info",
            "_split",
        }
