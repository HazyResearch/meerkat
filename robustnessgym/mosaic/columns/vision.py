from __future__ import annotations

import copy
import gzip
import logging
import os
import pickle
import tempfile
import uuid
from collections import defaultdict
from types import SimpleNamespace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import cytoolz as tz
import datasets
import numpy as np

# import pyarrow as pa
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

# from datasets import DatasetInfo, Features
# from joblib import Parallel, delayed
from robustnessgym.core.cells.abstract import AbstractCell
from robustnessgym.core.columns.abstract import AbstractColumn
from robustnessgym.core.dataformats.vision import TorchDataset, VisionDataset

# from robustnessgym.core.dataformats.abstract import AbstractDataset
from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import convert_to_batch_fn

logger = logging.getLogger(__name__)

Example = Dict
Batch = Dict[str, List]

"""
Map
-----
Materialize.
Cache.
Batch.
--> 8 combinations

NumpyColumn: np.ndarray --> no real notion of a Cell (sequence of cells)
TensorColumn: torch.tensor (len_column, .....) -->

ImageColumn --> list of cells
PathColumn -> TensorColumn (unmaterialized images)

tensor_column[2:4] --> tensor
path_column[2:4] --> list of paths
image_column[2:4] --> list of Image objects

# Storing objects as cells
DosmaColumn -> [MedicalVolume_1, ..., MedicalVolume_n]
SpacyColumn --> [Doc_1, Doc_2, ..., Doc_n]
spacy_column.serialize() --> [Doc_1.serialize(), ..., Doc_n.serialize()]

# Conclusions
1. we need cells: for complex objects
2. cell columns can have batching functions (dataloader)

# Workflow (in-memory image workflow)
How Stored:
- stored with np.save --> [n, ...]
    - load it with np.load --> NumpyColumn
    - chunks?

Apache Arrow:
- Record, RecordBatch, Column, Table

Candidates:
- NumpyColumn
- TensorColumn

# API
dataset = Dataset.from_batch({
    'text': ['a', 'b'],
    'metadata': [{'c': {'d': 2}}, {'c': {'d': 5}}],
    'arrays': [np.array([1,2,3]), np.array([3,4,5])],
    'arr': np.array([1, 2]),
    'paths': [Path('../'), Path('/')], # specify the paths to where the thing is
    'images': [Image('abc'), Image('b')],
    'spacy': [nlp('a'), nlp('b')],
    'tensor': torch.tensor([1, 2]),
    'tensor_gpu': torch.tensor([1, 2]).cuda(),
})
# Functions that mutate datasets
dataset --> dataset
# NLI / Dialog
dataset of conversations where each example is a conversation
-->
dataset of turns where each example is a turn in a conversation

dataset[2:4] -- natural + the user has to do fewer steps

# 1 Path --> Multiple examples
# k Paths --> 1 example

dataset.map(lambda example: activation(example['images'])) # grab activations
# Procedure
# Require that the len(column) are equal
# For each column, determine the appropriate Column to use
    # Column(data)? or Column.from_sequence()?


# Materialized image column is a tensor column
image_column
image_column.materialize() -> TensorColumn [in-memory]

# Do I want to materialize the whole column? (Memory)
vs.
# Materialize by batch

# Loader

ImageCell
------------
| filepath  |
| loader    |
| transform |
------------ materialize
| data      |
|           |
------------



# Types
# Lazy vs. eager

"""


class Image(AbstractCell):
    """This class acts as an interface to allow the user to manipulate the
    images without actually loading them into memory."""

    def __init__(self, filepath: str, transform: callable = None):
        super(Image, self).__init__()

        # Images contain a filepath and filename
        self.filepath = filepath
        self.name = os.path.split(filepath)[-1]

        # Cache the transforms applied on the image when VisionDataset.update
        # gets called
        self.transform = transform
        self.data = None

    def display(self):
        pass

    @property
    def is_materialized(self):
        return self.data is not None

    def materialize(self):
        if self.data is not None:
            return self.data

        import torchvision.datasets.folder as folder

        image = torch.from_numpy(np.array(folder.default_loader(self.filepath)))

        if self.transform is not None:
            image = self.transform(image)

        self.data = image

        return image

    def __getitem__(self, idx):
        image = self.load()
        return image[idx]

    def __str__(self):
        return self.filepath

    def __repr__(self):
        return "Image(%s)" % self.name

    def __eq__(self, other):
        filepath_eq = self.filepath == other.filepath
        transform_eq = self.transform == other.transform
        return filepath_eq and transform_eq


class VisionColumn(AbstractColumn):
    """Class for vision datasets that are to be stored in memory."""

    def __init__(
        self,
        data: Union[List, Dict, datasets.Dataset],
        transform: Callable = None,
        collate_fn: Callable = None,
        identifier: Identifier = None,
    ):

        # Data is a dictionary of lists
        self._data = {}

        self.transform = (lambda x: x) if transform is None else transform

        self.collate_fn = default_collate if collate_fn is None else collate_fn

        # Internal state variables for the update function
        # TODO(sabri): look into what's good with these
        self._updating_images = False
        self._adding_images = False
        self._callstack = []

        if data is not None:
            # `data` is a dictionary
            if isinstance(data, list) and len(data):
                # Assert all columns are the same length
                data = self._paths_to_images(data)
                self._data = data

            # `data` is a datasets.Dataset
            elif isinstance(data, VisionColumn):
                # TODO(sabri): once we implement indexing, make sure this checks out
                self._data = data[:]

        else:
            # Create an empty Column
            self._data = []

        # Call the superclass
        AbstractColumn.__init__(self, num_rows=len(self._data), identifier=identifier)

        # Create attributes for visible rows
        self.visible_rows = None

        # Initialization
        self._initialize_state()

        # flag indicating whether the images themselves are actually stored in memory
        self._materialized = False

    def _initialize_state(self):

        # Show all rows by default
        self.visible_rows = None

        self._set_features()

    @staticmethod
    def _paths_to_images(paths: List[str]):
        """Convert a list of paths to images data[key] into a list of Image
        instances."""
        if isinstance(paths[0], Image):
            return paths  # Can happen when we're copying a dataset
        return [Image(path) for path in paths]

    def _set_features(self):
        """Set the features of the dataset."""
        # TODO: discuss whether this needs to exist
        self.dtype = "Image"

    def _materialize(self):
        # Materialize data, instead of using a reference to an ancestor Dataset
        self._data = {k: self[k] for k in self._data}

        # Reset visible_rows
        self.set_visible_rows(None)

    def add_column(self, column: str, values: List, overwrite=False) -> None:
        """Add a column to the dataset."""

        assert (
            column not in self.all_columns
        ) or overwrite, (
            f"Column `{column}` already exists, set `overwrite=True` to overwrite."
        )
        assert len(values) == len(self), (
            f"`add_column` failed. "
            f"Values length {len(values)} != dataset length {len(self)}."
        )

        if self.visible_rows is not None:
            # Materialize the data
            self._materialize()

        # Add the column
        self._data[column] = list(values)
        self.all_columns.append(column)
        self.visible_columns.append(column)

        # Set features
        self._set_features()

        logger.info(f"Added column `{column}` with length `{len(values)}`.")

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

    def _append_to_empty_dataset(self, example_or_batch: Union[Example, Batch]) -> None:
        """Append a batch of data to the dataset when it's empty."""
        # Convert to batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # TODO(karan): what other data properties need to be in sync here
        self.all_columns = list(batch.keys())
        self.visible_columns = list(batch.keys())

        # Dataset is empty: create the columns and append the batch
        self._data = {k: [] for k in self.column_names}
        for k in self.column_names:
            self._data[k].extend(batch[k])

    def append(
        self,
        example_or_batch: Union[Example, Batch],
    ) -> None:
        """Append a batch of data to the dataset.

        `batch` must have the same columns as the dataset (regardless of
        what columns are visible).
        """
        if not self.column_names:
            return self._append_to_empty_dataset(example_or_batch)

        # Check that example_or_batch has the same format as the dataset
        # TODO(karan): require matching on nested features?
        columns = list(example_or_batch.keys())
        assert set(columns) == set(
            self.column_names
        ), f"Mismatched columns\nbatch: {columns}\ndataset: {self.column_names}"

        # Convert to a batch
        batch = self._example_or_batch_to_batch(example_or_batch)

        # Append to the dataset
        for k in self.column_names:
            if k in self._img_columns:
                batch[k] = list(map(Image, batch[k]))
            self._data[k].extend(batch[k])

    def _inspect_update_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:
        """Load the images before calling _inspect_function, and check if new
        image columns are being added."""

        properties = self._inspect_function(function, with_indices, batched)

        # Check if new columns are added
        if batched:
            if with_indices:
                output = function(self[:2], range(2))
            else:
                output = function(self[:2])

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])
        new_columns = set(output.keys()).difference(set(self.all_columns))

        # Check if any of those new columns is an image column
        new_img_columns = []
        for key in new_columns:
            val = output[key]
            if isinstance(val, torch.Tensor) and len(val.shape) >= 2:
                new_img_columns.append(key)

        properties.new_image_columns = new_img_columns

        return properties

    def update(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        # input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        cache_dir: str = None,
        **kwargs,
    ) -> Optional[VisionDataset]:
        """Update the columns of the dataset."""
        # TODO(karan): make this fn go faster
        # most of the time is spent on the merge, speed it up further

        # Sanity check when updating the images
        self._callstack.append("update")

        # Return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return self

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return self

        # Get some information about the function
        function_properties = self._inspect_update_function(
            function, with_indices, batched
        )
        assert (
            function_properties.dict_output
        ), f"`function` {function} must return dict."

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to batched function.")

        updated_columns = function_properties.existing_columns_updated
        changed_images = [key in self._img_columns for key in updated_columns]
        new_image_columns = function_properties.new_image_columns

        # Set the internal state for the map function
        self._updating_images = any(changed_images)
        self._adding_images = any(new_image_columns)
        if self._updating_images or self._adding_images:
            # Set the cache directory where the modified images will be stored
            if not cache_dir:
                cache_dir = tempfile.gettempdir()
                logger.warning(
                    "Modifying the images without setting a cache directory.\n"
                    "Consider setting it if your dataset is very large.\n"
                    "The default image cache location is: {}".format(cache_dir)
                )

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            cache_dir = os.path.join(cache_dir, uuid.uuid4().hex)
            os.mkdir(cache_dir)

        # Update always returns a new dataset
        logger.info("Running update, a new dataset will be returned.")
        if self.visible_rows is not None:
            # Run .map() to get updated batches and pass them into a new dataset
            new_dataset = VisionDataset(
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
                    cache_dir=cache_dir,
                ),
                img_columns=self._img_columns,
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
                    cache_dir=cache_dir,
                    new_image_columns=new_image_columns,
                )

                # If new image columns were added, update that information
                if self._adding_images:
                    new_dataset._img_columns.extend(new_image_columns)

                # Add new columns / overwrite existing columns for the update
                for col, vals in output.items():
                    if isinstance(vals[0], torch.Tensor) and vals[
                        0
                    ].shape == torch.Size([]):
                        # Scalar tensor. Convert to Python.
                        new_vals = []
                        for val in vals:
                            new_vals.append(val.item())
                        vals = new_vals
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
                    cache_dir=cache_dir,
                    new_image_columns=new_image_columns,
                )

                # If new image columns were added, update that information
                if self._adding_images:
                    new_dataset._img_columns.extend(new_image_columns)

                # Add new columns for the update
                for col, vals in output.items():
                    if isinstance(vals[0], torch.Tensor) and vals[
                        0
                    ].shape == torch.Size([]):
                        # Scalar tensor. Convert to Python.
                        new_vals = []
                        for val in vals:
                            new_vals.append(val.item())
                        vals = new_vals
                    new_dataset.add_column(col, vals)

        # Remove columns
        if remove_columns:
            for col in remove_columns:
                new_dataset.remove_column(col)
            logger.info(f"Removed columns {remove_columns}.")
        # Reset the format
        # if input_columns:
        #     self.set_format(previous_format)

        # Remember to reset the internal state
        self._updating_images = False
        self._adding_images = False
        # And remove this call from the callstack
        self._callstack.pop()

        # If the new dataset is a copy we also need to reset it
        new_dataset._updating_images = False
        new_dataset._adding_images = False
        new_dataset._callstack.pop()

        return new_dataset

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        num_proc: int = 0,
    ):
        """Batch the dataset.

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size

        Returns:
            batches of data
        """
        # Load the images
        return self.to_dataloader(
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last_batch,
            num_workers=num_proc,
            collate_fn=self.collate_fn,
        )

    def _inspect_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:

        # Initialize variables to track
        no_output = dict_output = bool_output = list_output = False

        # If dict_output = True and `function` is used for updating the dataset
        # useful to know if any existing column is modified
        updates_existing_column = True
        existing_columns_updated = []

        # Run the function to test it
        if batched:
            if with_indices:
                output = function(self[:2].to_batch(), range(2))
            else:
                output = function(self[:2].to_batch())

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])

        if isinstance(output, Mapping):
            # `function` returns a dict output
            dict_output = True

            # Set of columns that are updated
            existing_columns_updated = set(self.all_columns).intersection(
                set(output.keys())
            )

            # Check if `function` updates an existing column
            if len(existing_columns_updated) == 0:
                updates_existing_column = False

        elif output is None:
            # `function` returns None
            no_output = True
        elif isinstance(output, bool):
            # `function` returns a bool
            bool_output = True
        elif isinstance(output, list):
            # `function` returns a list
            list_output = True
            if batched and isinstance(output[0], bool):
                # `function` returns a bool per example
                bool_output = True

        return SimpleNamespace(
            dict_output=dict_output,
            no_output=no_output,
            bool_output=bool_output,
            list_output=list_output,
            updates_existing_column=updates_existing_column,
            existing_columns_updated=existing_columns_updated,
        )

    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = None,
        materialize: bool = None,
        **kwargs,
    ) -> Optional[Union[Dict, List]]:
        """Apply a map over the dataset."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False
        materialize = self._materialize if materialize is None else materialize

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_proc is not None
        if num_proc is None:
            num_proc = 0

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if not batched:
            # Convert to a batch function
            function = convert_to_batch_fn(function, with_indices=with_indices)
            logger.info(f"Converting `function` {function} to a batched function.")

        # # Get some information about the function
        # function_properties = self._inspect_function(
        #     function,
        #     with_indices,
        #     batched=batched,
        # )

        # If we are updating, prepare image savers and perform sanity checks
        if self._updating_images or self._adding_images:
            assert "update" in self._callstack, (
                "_updating_images and _adding_images can only be set by "
                "VisionDataset.update"
            )
            assert "cache_dir" in kwargs, "No cache directory specified"
            # cache_dir = kwargs["cache_dir"]
        if self._adding_images:
            assert "new_image_columns" in kwargs, "New image column names not specified"
            # new_image_columns = kwargs["new_image_columns"]

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = None
        for i, batch in tqdm(
            enumerate(self.batch(batch_size, drop_last_batch)),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
        ):
            # Run `function` on the batch
            output = (
                function(
                    batch,
                    range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                )
                if with_indices
                else function(batch)
            )

            # Save the modified images
            # if self._updating_images:
            #     for key in image_loaders:
            #         images = output[key]
            #
            #         # Save the images in parallel
            #         Images = Parallel(n_jobs=num_proc)(
            #             delayed(save_image)(
            #                 images[idx],
            #                 os.path.join(
            #                     cache_dir,
            #                     "{0}{1}.png".format(key, i * batch_size + idx),
            #                 ),
            #             )
            #             for idx in range(len(images))
            #         )
            #         output[key] = Images

            # if self._adding_images:
            #     for key in new_image_columns:
            #         images = output[key]
            #
            #         # Save the images in parallel
            #         Images = Parallel(n_jobs=num_proc)(
            #             delayed(save_image)(
            #                 images[idx],
            #                 os.path.join(
            #                     cache_dir,
            #                     "{0}{1}.png".format(key, i * batch_size + idx),
            #                 ),
            #             )
            #             for idx in range(len(images))
            #         )
            #
            #         output[key] = Images

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, Mapping) else []

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        # Reset the format
        # if input_columns:
        #     self.set_format(previous_format)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        num_proc: Optional[int] = 64,
        **kwargs,
    ) -> Optional[VisionDataset]:
        """Apply a filter over the dataset."""
        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_proc=num_proc,
        )
        indices = np.where(outputs)[0]

        # Reset the format to set visible columns for the filter
        with self.format():
            # Filter returns a new dataset
            new_dataset = self.copy()
            new_dataset.set_visible_rows(indices)

        return new_dataset

    def to_dataloader(
        self,
        columns: Sequence[str],
        column_to_transform: Optional[Mapping[str, Callable]] = None,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """Get a PyTorch dataloader that iterates over a subset of the columns
        (specified by `keys`) in the dataset. This is handy when using the dataset with
        training or evaluation loops outside of robustnessgym.  For example:
        ```
        dataset = Dataset(...)
        for img, target in dataset.to_dataloader(
            keys=["img_path", "label"],
            batch_size=16,
            num_workers=12
        ):
            out = model(img)
            loss = loss(out, target)
            ...
        ```

        Args:
            columns (Sequence[str]): A subset of the columns in the vision dataset.
                Specifies the columns to load. The dataloader will return values in same
                 order as `columns` here.
            column_to_transform (Optional[Mapping[str, Callable]], optional): A mapping
                from zero or more `keys` to callable transforms to be applied by the
                dataloader. Defaults to None, in which case no transforms are applied.
                Example: `column_to_transform={"img_path": transforms.Resize((16,16))}`.
                Note: these transforms will be applied after the transforms specified
                via the `transform` argument to `VisionDataset`.

        Returns:
            torch.utils.data.DataLoader: dataloader that iterates over dataset
        """
        img_folder = TorchDataset(
            self, columns=columns, column_to_transform=column_to_transform
        )
        return torch.utils.data.DataLoader(img_folder, **kwargs)

    def copy(self, deepcopy=False):
        """Return a copy of the dataset."""
        if deepcopy:
            return copy.deepcopy(self)
        else:
            dataset = VisionDataset()
            dataset.__dict__ = {k: copy.copy(v) for k, v in self.__dict__.items()}
            return dataset

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {
            "_data",
            "all_columns",
            "visible_rows",
            "_info",
            "_split",
            "_img_columns",
            "_updating_images",
            "_adding_images",
            "_callstack",
        }

    @classmethod
    def _assert_state_keys(cls, state: Dict) -> None:
        """Assert that a state contains all required keys."""
        assert (
            set(state.keys()) == cls._state_keys()
        ), f"State must contain all state keys: {cls._state_keys()}."

    def __getstate__(self) -> Dict:
        """Get the internal state of the dataset."""
        state = {key: getattr(self, key) for key in self._state_keys()}
        self._assert_state_keys(state)
        return state

    def __setstate__(self, state: Dict) -> None:
        """Set the internal state of the dataset."""
        if not isinstance(state, dict):
            raise ValueError(
                f"`state` must be a dictionary containing " f"{self._state_keys()}."
            )

        self._assert_state_keys(state)

        for key in self._state_keys():
            setattr(self, key, state[key])

        # Do some initialization
        self._initialize_state()

    @classmethod
    def load_from_disk(cls, path: str) -> VisionDataset:
        """Load the in-memory dataset from disk."""

        with gzip.open(os.path.join(path, "data.gz")) as f:
            dataset = pickle.load(f)
        # # Empty state dict
        # state = {}
        #
        # # Load the data
        # with gzip.open(os.path.join(path, "data.gz")) as f:
        #     state['_data'] = pickle.load(f)
        #
        # # Load the metadata
        # metadata = json.load(
        #     open(os.path.join(path, "metadata.json"))
        # )
        #
        # # Merge the metadata into the state
        # state = {**state, **metadata}

        # Create an empty `VisionDataset` and set its state
        # dataset = cls()
        # dataset.__setstate__(state)

        return dataset

    def save_to_disk(self, path: str):
        """Save the in-memory dataset to disk."""
        # Create all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Store the data in a compressed format
        with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
            pickle.dump(self, f)

        # # Get the dataset state
        # state = self.__getstate__()
        #
        # # Store the data in a compressed format
        # with gzip.open(os.path.join(path, "data.gz"), "wb") as f:
        #     pickle.dump(state['_data'], f)
        #
        # # Store the metadata
        # json.dump(
        #     {k: v for k, v in state.items() if k != '_data'},
        #     open(os.path.join(path, "metadata.json"), 'w'),
        # )
