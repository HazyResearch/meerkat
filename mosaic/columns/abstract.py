from __future__ import annotations

import abc
import logging
import reprlib
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from mosaic.mixins.collate import CollateMixin
from mosaic.mixins.copying import CopyMixin
from mosaic.mixins.identifier import IdentifierMixin
from mosaic.mixins.index import IndexableMixin
from mosaic.mixins.inspect_fn import FunctionInspectorMixin
from mosaic.mixins.mapping import MappableMixin
from mosaic.mixins.materialize import MaterializationMixin
from mosaic.mixins.state import StateDictMixin
from mosaic.mixins.storage import ColumnStorageMixin
from mosaic.mixins.visibility import VisibilityMixin
from mosaic.tools.identifier import Identifier
from mosaic.tools.utils import convert_to_batch_column_fn
from mosaic.writers.list_writer import ListWriter

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
        collate_fn: Callable = None,
        *args,
        **kwargs,
    ):
        # Assign to data
        self._data = data

        super(AbstractColumn, self).__init__(
            n=len(data) if data is not None else 0,
            identifier=identifier,
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
        return {"_collate_fn", "_data", "_visible_rows"}

    def _get_cell(self, index: int, materialize: bool = True) -> Any:
        """Get a single cell from the column.

        Args:
            index (int): This is an index into the ALL rows, not just visible rows. In
                other words, we assume that the index passed in has already been
                remapped via `_remap_index`, if `self.visible_rows` is not `None`.
            materialize (bool, optional): Materialize and return the object. This
                argument is used by subclasses of `AbstractColumn` that hold data in an
                unmaterialized format. Defaults to False.
        """
        return self._data[index]

    def _get_batch(
        self, indices: np.ndarray, materialize: bool = True
    ) -> AbstractColumn:
        """Get a batch of cells from the column.

        Args:
            index (int): This is an index into the ALL rows, not just visible rows. In
                other words, we assume that the index passed in has already been
                remapped via `_remap_index`, if `self.visible_rows` is not `None`.
            materialize (bool, optional): Materialize and return the object. This
                argument is used by subclasses of `AbstractColumn` that hold data in an
                unmaterialized format. Defaults to False.
        """
        if materialize:
            return self.collate([self._get_cell(int(i)) for i in indices])

        else:
            new_column = self.copy()
            new_column._visible_rows = indices
            return new_column

    def _get(self, index, materialize: bool = True):
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
            return self._get_cell(int(index), materialize=materialize)

        # `index` should return a batch
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            indices = np.arange(self.full_length())[index]
        elif isinstance(index, tuple) or isinstance(index, list):
            indices = index
        elif isinstance(index, np.ndarray):
            if len(index.shape) != 1:
                raise TypeError(
                    "`np.ndarray` index must have 1 axis, not {}".format(
                        len(index.shape)
                    )
                )
            indices = np.arange(self.full_length())[index]
        elif isinstance(index, AbstractColumn):
            indices = np.arange(self.full_length())[index]
        else:
            raise TypeError(
                "Object of type {} is not a valid index".format(type(index))
            )
        return self.__class__.from_data(
            self._get_batch(indices, materialize=materialize)
        )

    def __getitem__(self, index):
        return self._get(index, materialize=True)

    @staticmethod
    def _convert_to_batch_fn(
        function: Callable, with_indices: bool, materialize: bool = True
    ) -> callable:
        return convert_to_batch_column_fn(
            function=function, with_indices=with_indices, materialize=materialize
        )

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
        materialize: bool = True,
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
            function, with_indices, batched=batched, materialize=materialize
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
            materialize=materialize,
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
        materialize: bool = True,
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
                self if materialize else self.lz,
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
                self if materialize else self.lz,
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
            return data.copy()
        elif torch.is_tensor(data):
            from .tensor_column import TensorColumn

            return TensorColumn(data)
        elif isinstance(data, np.ndarray):
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data)
        elif isinstance(data, pd.Series):
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data.values)
        elif isinstance(data, Sequence):
            from ..cells.abstract import AbstractCell

            if len(data) != 0 and isinstance(data[0], AbstractCell):
                from .image_column import ImageColumn

                return ImageColumn.from_cells(data)

            if len(data) != 0 and isinstance(data[0], AbstractCell):
                from .cell_column import CellColumn

                return CellColumn(data)

            if len(data) != 0 and isinstance(
                data[0], (int, float, bool, np.ndarray, np.generic)
            ):
                from .numpy_column import NumpyArrayColumn

                return NumpyArrayColumn(data)
            elif len(data) != 0 and torch.is_tensor(data[0]):
                from .tensor_column import TensorColumn

                return TensorColumn(torch.stack(data))

            from .list_column import ListColumn

            return ListColumn(data)
        else:
            raise ValueError(f"Cannot create column out of data of type {type(data)}")

    def head(self, n: int = 5) -> AbstractColumn:
        """Get the first `n` examples of the column."""
        return self.lz[:n]

    def tail(self, n: int = 5) -> AbstractColumn:
        """Get the last `n` examples of the column."""
        return self.lz[-n:]

    def to_pandas(self) -> pd.Series:
        return pd.Series([self.lz[int(idx)] for idx in range(len(self))])
