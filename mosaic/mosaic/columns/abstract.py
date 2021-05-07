from __future__ import annotations

import abc
import logging
import reprlib
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from pandas.core import generic

from robustnessgym.core.identifier import Identifier
from robustnessgym.core.tools import convert_to_batch_column_fn
from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.mixins.collate import CollateMixin
from robustnessgym.mosaic.mixins.copying import CopyMixin
from robustnessgym.mosaic.mixins.identifier import IdentifierMixin
from robustnessgym.mosaic.mixins.index import IndexableMixin
from robustnessgym.mosaic.mixins.inspect_fn import FunctionInspectorMixin
from robustnessgym.mosaic.mixins.mapping import MappableMixin
from robustnessgym.mosaic.mixins.materialize import MaterializationMixin
from robustnessgym.mosaic.mixins.state import StateDictMixin
from robustnessgym.mosaic.mixins.storage import ColumnStorageMixin
from robustnessgym.mosaic.mixins.visibility import VisibilityMixin
from robustnessgym.mosaic.writers.list_writer import ListWriter

logger = logging.getLogger(__name__)


class AbstractColumn(
    CollateMixin,
    ColumnStorageMixin,
    CopyMixin,
    FunctionInspectorMixin,
    IdentifierMixin,
    IndexableMixin,
    MappableMixin,
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
            return f"{self.__class__.__name__}View" f"({reprlib.repr(self.data)})"
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
        return {"_materialize", "_collate_fn", "_data", "_visible_rows"}

    def _get_cell(self, index: int):
        return self.data[index]

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # `index` should return a single element
        # np.ndarray indexed with a tuple of length 1 does not return an np.ndarray
        # so we match this behavior
        if (
            isinstance(index, int)
            or isinstance(index, np.int)
            or (isinstance(index, tuple) and len(index) == 1)
        ):
            data = self._get_cell(int(index))

            # Check if the column implements materialization
            if self.materialize:
                if isinstance(data, AbstractCell):
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
            indices = np.arange(len(self))[index]
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
        return self.__class__.from_data(self._get_batch(indices))

    def _get_batch(self, indices: np.ndarray):
        if self.materialize:
            return self.collate([self._get_cell(int(i)) for i in indices])

        else:
            new_column = self.copy()
            new_column.visible_rows = indices
            return new_column

    @staticmethod
    def _convert_to_batch_fn(function: Callable, with_indices: bool) -> callable:
        return convert_to_batch_column_fn(function=function, with_indices=with_indices)

    def __len__(self):
        # If only a subset of rows are visible
        if self.visible_rows is not None:
            return len(self.visible_rows)

        return self.full_length()

    def full_length(self):
        # Length of the underlying data stored in the column
        if self._data is not None:
            return len(self._data)
        return 0

    def _repr_pandas_(self) -> pd.Series:
        raise NotImplementedError

    def _repr_html_(self):
        # pd.Series objects do not implement _repr_html_
        return (
            self._repr_pandas_()
            .to_frame(name=f"({self.__class__.__name__})")
            ._repr_html_()
        )

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
        num_workers: int = 4,
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
        if self._get_batch.__func__ == AbstractColumn._get_batch:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                collate_fn=self.collate if collate else lambda x: x,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )
        else:
            batch_indices = []
            indices = np.arange(len(self))
            for i in range(0, len(self), batch_size):
                if drop_last_batch and i + batch_size > len(self):
                    continue
                batch_indices.append(indices[i : i + batch_size])
            return torch.utils.data.DataLoader(
                self,
                sampler=batch_indices,
                batch_size=None,
                batch_sampler=None,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )
        # if self.materialize:
        #     return torch.utils.data.DataLoader(
        #         self,
        #         batch_size=batch_size,
        #         collate_fn=self.collate if collate else lambda x: x,
        #         drop_last=drop_last_batch,
        #         *args,
        #         **kwargs,
        #     )
        # else:
        #     for i in range(0, len(self), batch_size):
        #         if drop_last_batch and i + batch_size > len(self):
        #             continue
        #         yield self[i : i + batch_size]

    @classmethod
    def get_writer(cls, mmap: bool = False):
        if mmap:
            raise ValueError("Memmapping not supported with this column type.")
        else:
            return ListWriter()

    Columnable = Union[Sequence, np.ndarray, pd.Series, torch.Tensor]

    @classmethod
    def from_data(cls, data: Union[Columnable, AbstractColumn]):
        """Convert data to a mosaic column using the appropriate Column
        type."""
        # need to import lazily to avoid circular import
        if isinstance(data, AbstractColumn):
            return data
        elif torch.is_tensor(data):
            # TODO: update this once we've added a torch.Tensor column
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data.cpu().detach().numpy())
        elif isinstance(data, np.ndarray):
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data)
        elif isinstance(data, pd.Series):
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data.values)
        elif isinstance(data, Sequence):
            from ..cells.abstract import AbstractCell

            if len(data) != 0 and isinstance(data[0], AbstractCell):
                from .cell_column import CellColumn

                return CellColumn(data)

            if len(data) != 0 and isinstance(
                data[0], (int, float, bool, np.ndarray, np.generic)
            ):
                from .numpy_column import NumpyArrayColumn

                return NumpyArrayColumn(data)

            from .list_column import ListColumn

            return ListColumn(data)
        else:
            raise ValueError(f"Cannot create column out of data of type {type(data)}")
