from __future__ import annotations

import abc
import logging
import reprlib
from collections import defaultdict
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from tqdm.auto import tqdm

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import convert_to_batch_column_fn
from robustnessgym.mosaic.mixins.collate import CollateMixin
from robustnessgym.mosaic.mixins.copying import CopyMixin
from robustnessgym.mosaic.mixins.identifier import IdentifierMixin
from robustnessgym.mosaic.mixins.index import IndexableMixin
from robustnessgym.mosaic.mixins.inspect_fn import FunctionInspectorMixin
from robustnessgym.mosaic.mixins.materialize import MaterializationMixin
from robustnessgym.mosaic.mixins.state import StateDictMixin
from robustnessgym.mosaic.mixins.storage import ColumnStorageMixin
from robustnessgym.mosaic.mixins.visibility import VisibilityMixin

logger = logging.getLogger(__name__)


class AbstractColumn(
    CollateMixin,
    ColumnStorageMixin,
    CopyMixin,
    FunctionInspectorMixin,
    IdentifierMixin,
    IndexableMixin,
    MaterializationMixin,
    StateDictMixin,
    VisibilityMixin,
    abc.ABC,
):
    """An abstract class for Mosaic columns."""

    _data: Sequence = None

    def __init__(
        self,
        data: Sequence = None,
        identifier: Identifier = None,
        materialize: bool = True,
        collate_fn: Callable = None,
        *args,
        **kwargs,
    ):
        # Assign to data
        self._data = data

        super(AbstractColumn, self).__init__(
            n=len(data) if data is not None else 0,
            identifier=identifier,
            materialize=materialize,
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )

        # Log creation
        logger.info(f"Created `{self.__class__.__name__}` with {len(self)} rows.")

    def __repr__(self):
        if self.visible_rows is not None:
            return (
                f"{self.__class__.__name__}View"
                f"({reprlib.repr([self.data[i] for i in self.visible_rows[:8]])})"
            )
        return f"{self.__class__.__name__}({reprlib.repr(self.data)})"

    def __str__(self):
        if self.visible_rows is not None:
            return (
                f"{self.__class__.__name__}View"
                f"({reprlib.repr([self.data[i] for i in self.visible_rows[:8]])})"
            )
        return f"{self.__class__.__name__}({reprlib.repr(self.data)})"

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return {}

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_materialize", "_collate_fn", "_data"}

    def _get_cell(self, index: int):
        return self.data[index]

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # `index` should return a single element
        if isinstance(index, int) or isinstance(index, np.int):
            data = self._get_cell(int(index))

            # Check if the column implements materialization
            if self.materialize:
                if hasattr(data, "get"):
                    # `data` has a `get` method that can be called for retrieving the
                    # "expensive" information
                    return data.get()
                else:
                    # `data` has no `get` method, return directly
                    return data
            else:
                return data

        # `index` should return a batch
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            indices = np.arange(
                0 if index.start is None else index.start,
                len(self) if index.stop is None else index.stop,
                1 if index.step is None else index.step,
            )
        elif (isinstance(index, tuple) or isinstance(index, list)) and len(index):
            indices = np.array(index)
        elif isinstance(index, np.ndarray):
            if len(index.shape) != 1:
                raise TypeError(
                    "`np.ndarray` index must have 1 axis, not {}".format(
                        len(index.shape)
                    )
                )
            indices = index
        else:
            raise TypeError(
                "object of type {} is not a valid index".format(type(index))
            )
        return self._get_batch(indices)

    def _get_batch(self, indices: np.ndarray):
        if self.materialize:
            return self.collate([self._get_cell(int(i)) for i in indices])

        else:
            new_column = self.copy()
            new_column.visible_rows = indices
            return new_column

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        # Length of the underlying data stored in the column
        if self.data is not None:
            return len(self.data)
        return 0

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
    ) -> Optional[Union[Dict, List, AbstractColumn]]:
        """Map a function over the elements of the column."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False

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
            function = convert_to_batch_column_fn(function, with_indices=with_indices)
            batched = True
            logger.info(f"Converting `function` {function} to a batched function.")

        # # Get some information about the function
        # TODO: discuss whether this is actually required vs. doing it on first pass in
        # loop
        function_properties = self._inspect_function(
            function,
            with_indices,
            batched=batched,
        )

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        outputs = defaultdict(list) if function_properties.dict_output else []
        for i, batch in tqdm(
            enumerate(
                self.batch(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    collate=batched,
                    # TODO: collate=batched was commented out in list_column
                )
            ),
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

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    for k in output.keys():
                        outputs[k].extend(output[k])
                else:
                    outputs.extend(output)

        if not len(outputs):
            return None
        elif isinstance(outputs, dict):
            # turns the defaultdict into dict
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
    ) -> Optional[AbstractColumn]:
        """Filter the elements of the column using a function."""
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

        new_column = self.copy()
        new_column.visible_rows = indices
        return new_column

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        collate: bool = True,
        *args,
        **kwargs,
    ):
        """Batch the column.

        Args:
            batch_size: integer batch size
            drop_last_batch: drop the last batch if its smaller than batch_size
            collate: whether to collate the returned batches

        Returns:
            batches of data
        """
        if self.materialize:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                collate_fn=self.collate if collate else lambda x: x,
                drop_last=drop_last_batch,
                *args,
                **kwargs,
            )
        else:
            for i in range(0, len(self), batch_size):
                if drop_last_batch and i + batch_size > len(self):
                    continue
                yield self[i : i + batch_size]
