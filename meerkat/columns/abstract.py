from __future__ import annotations

import abc
import logging
from meerkat.mixins.cloneable import CloneableMixin
import reprlib
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from meerkat.mixins.collate import CollateMixin
from meerkat.mixins.copying import ColumnCopyMixin
from meerkat.mixins.identifier import IdentifierMixin
from meerkat.mixins.inspect_fn import FunctionInspectorMixin
from meerkat.mixins.mapping import MappableMixin
from meerkat.mixins.materialize import MaterializationMixin
from meerkat.mixins.state import StateDictMixin
from meerkat.mixins.storage import ColumnStorageMixin
from meerkat.mixins.visibility import VisibilityMixin
from meerkat.provenance import ProvenanceMixin, capture_provenance
from meerkat.tools.identifier import Identifier
from meerkat.tools.utils import convert_to_batch_column_fn
from meerkat.writers.list_writer import ListWriter

logger = logging.getLogger(__name__)


class AbstractColumn(
    CollateMixin,
    ColumnStorageMixin,
    ColumnCopyMixin,
    FunctionInspectorMixin,
    IdentifierMixin,
    MappableMixin,
    MaterializationMixin,
    ProvenanceMixin,
    StateDictMixin,
    VisibilityMixin,
    abc.ABC,
):
    """An abstract class for Meerkat columns."""

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
        return f"{self.__class__.__name__}({reprlib.repr(self.data)})"

    def streamlit(self):
        return self._repr_pandas_()

    @property
    def data(self):
        """Get the underlying data (excluding invisible rows).

        To access underlying data with invisible rows, use `_data`.
        """
        if self.visible_rows is not None:
            return self._data[self.visible_rows]
        else:
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
            new_column = self.view()
            # indices have already been remapped so we don't use the setter
            new_column._visible_rows = indices
            return new_column

    def _get(self, index, materialize: bool = True):
        index = self._translate_index(index)
        if isinstance(index, int):
            return self._get_cell(index, materialize=materialize)
        elif isinstance(index, np.ndarray):
            batch = self._get_batch(index, materialize=materialize)
            return self.__class__.from_data(batch)

    # @capture_provenance()
    def __getitem__(self, index):
        return self._get(index, materialize=True)

    def _set_cell(self, index, value):
        self._data[index] = value

    def _set_batch(self, indices: np.ndarray, values):
        for index, value in zip(indices, values):
            self._set_cell(int(index), value)

    def _set(self, index, value):
        index = self._translate_index(index)
        if isinstance(index, int):
            self._set_cell(index, value)
        elif isinstance(index, Sequence) or isinstance(index, np.ndarray):
            self._set_batch(index, value)
        else:
            raise ValueError

    def __setitem__(self, index, value):
        self._set(index, value)

    def _translate_index(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # `index` should return a single element
        # np.ndarray indexed with a tuple of length 1 does not return an np.ndarray
        # so we match this behavior
        if isinstance(index, int) or (isinstance(index, tuple) and len(index) == 1):
            return index

        # `index` should return a batch
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            # TODO (sabri): get rid of the np.arange here, very slow for large columns
            indices = np.arange(self.full_length())[index]
        elif isinstance(index, tuple) or isinstance(index, list):
            indices = np.array(index)
        elif isinstance(index, np.ndarray):
            if len(index.shape) != 1:
                raise TypeError(
                    "`np.ndarray` index must have 1 axis, not {}".format(
                        len(index.shape)
                    )
                )
            if index.dtype == bool:
                indices = np.where(index)[0]
            else:
                return index
        elif isinstance(index, AbstractColumn):
            # TODO (sabri): get rid of the np.arange here, very slow for large columns
            indices = np.arange(self.full_length())[index]
        else:
            raise TypeError(
                "Object of type {} is not a valid index".format(type(index))
            )
        return indices

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

    @capture_provenance()
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: Optional[int] = 0,
        materialize: bool = True,
        pbar: bool = False,
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
            function, with_indices, is_batched_fn=is_batched_fn, materialize=materialize
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
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

        new_column = self.view()
        new_column.visible_rows = indices
        return new_column

    def append(self, column: AbstractColumn) -> None:
        # TODO(Sabri): implement a naive `ComposedColumn` for generic append and
        # implement specific ones for ListColumn, NumpyColumn etc.
        raise NotImplementedError

    @staticmethod
    def concat(columns: Sequence[AbstractColumn]) -> None:
        # TODO(Sabri): implement a naive `ComposedColumn` for generic append and
        # implement specific ones for ListColumn, NumpyColumn etc.
        raise NotImplementedError

    def batch(
        self,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        collate: bool = True,
        num_workers: int = 0,
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
    # @capture_provenance()
    def from_data(cls, data: Union[Columnable, AbstractColumn]):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        if isinstance(data, AbstractColumn):
            return data.view()

        if torch.is_tensor(data):
            from .tensor_column import TensorColumn

            return TensorColumn(data)

        if isinstance(data, np.ndarray):
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data)

        if isinstance(data, pd.Series):
            from .numpy_column import NumpyArrayColumn

            if data.dtype != object or (len(data) > 0 and isinstance(data[0], str)):
                # if first element in object series is str, still return a
                # `NumpyArrayColumn`
                return NumpyArrayColumn(data.values)
            # otherwise, get the list of objects and proceed to `Sequence` case
            # below (e.g. `pd.Series` of AbstractCells)
            data: List = list(data.values)

        if isinstance(data, Sequence):
            from ..cells.abstract import AbstractCell
            from ..cells.imagepath import ImagePath

            if len(data) != 0 and isinstance(data[0], ImagePath):
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
