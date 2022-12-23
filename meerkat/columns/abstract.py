from __future__ import annotations

import abc
from ast import Dict
import logging
import pathlib
import reprlib
from copy import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

import meerkat.config
from meerkat.errors import ConversionError
from meerkat.interactive.node import NodeMixin
from meerkat.mixins.aggregate import AggregateMixin
from meerkat.mixins.blockable import BlockableMixin
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.mixins.collate import CollateMixin
from meerkat.mixins.deferable import DeferrableMixin
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.mixins.indexing import MaterializationMixin
from meerkat.mixins.inspect_fn import FunctionInspectorMixin
from meerkat.mixins.io import ColumnIOMixin
from meerkat.mixins.reactifiable import ReactifiableMixin
from meerkat.provenance import ProvenanceMixin, capture_provenance
from meerkat.tools.utils import convert_to_batch_column_fn, translate_index

if TYPE_CHECKING:
    from meerkat.interactive.formatter import Formatter


logger = logging.getLogger(__name__)


class Column(
    AggregateMixin,
    BlockableMixin,
    CloneableMixin,
    CollateMixin,
    ColumnIOMixin,
    FunctionInspectorMixin,
    IdentifiableMixin,
    DeferrableMixin,
    MaterializationMixin,
    NodeMixin,
    ProvenanceMixin,
    ReactifiableMixin,
    abc.ABC,
):
    """An abstract class for Meerkat columns."""

    _data: Sequence = None

    # Path to a log directory
    logdir: pathlib.Path = pathlib.Path.home() / "meerkat/"

    # Create a directory
    logdir.mkdir(parents=True, exist_ok=True)

    _self_identifiable_group: str = "columns"

    def __init__(
        self,
        data: Sequence = None,
        collate_fn: Callable = None,
        formatter: Callable = None,
        *args,
        **kwargs,
    ):
        """

        Args:
            data (Sequence, optional): [description]. Defaults to None.
            collate_fn (Callable, optional): [description]. Defaults to None.
            formatter (Callable, optional): . Defaults to None.
        """
        # Assign to data
        self._set_data(data)

        super(Column, self).__init__(
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )

        self._formatter = (
            formatter if formatter is not None else self._get_default_formatter()
        )

        # Log creation
        logger.info(f"Created `{self.__class__.__name__}` with {len(self)} rows.")

    def __repr__(self):
        return f"{self.__class__.__name__}({reprlib.repr(self.data)})"

    def __str__(self):
        return f"{self.__class__.__name__}({reprlib.repr(self.data)})"

    def streamlit(self):
        return self._repr_pandas_()

    def _set_data(self, data):
        if self.is_blockable():
            data = self._unpack_block_view(data)
        self._data = data

    def _is_valid_primary_key(self):
        """Subclasses should implement checks for ensuring that the column
        could be used as a valid primary key.

        Specifically, the check should ensure that the values in the
        column are unique. If the check does not pass, returns False. If
        the subclass has not implemented this method.
        """
        return False

    def _keyidx_to_posidx(self, keyidx: Any) -> int:
        """Get the posidx of the first occurrence of the given keyidx. Raise a
        key error if the keyidx is not found.

        Args:
            keyidx: The keyidx to search for.

        Returns:
            The posidx of the first occurrence of the given keyidx.
        """
        raise NotImplementedError()

    def _keyidxs_to_posidxs(self, keyidxs: Sequence[Any]) -> np.ndarray:
        """Get the posidxs of the given keyidxs. Raise a key error if any of
        the keyidxs are not found.

        Args:
            keyidxs: The keyidxs to search for.

        Returns:
            The posidxs of the given keyidxs.
        """
        raise NotImplementedError()

    @property
    def data(self):
        """Get the underlying data."""
        return self._data

    @data.setter
    def data(self, value):
        self._set_data(value)

    @property
    def metadata(self):
        return {}

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"_collate_fn", "_formatter"}

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

    def _get_batch(self, indices: np.ndarray, materialize: bool = True) -> Column:
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
            return self.collate(
                [self._get_cell(int(i), materialize=materialize) for i in indices]
            )

        else:
            return self.collate(
                [self._get_cell(int(i), materialize=materialize) for i in indices]
            )

    def _get(self, index, materialize: bool = True, _data: np.ndarray = None):
        index = self._translate_index(index)
        if isinstance(index, int):
            if _data is None:
                _data = self._get_cell(index, materialize=materialize)
            return _data

        elif isinstance(index, np.ndarray):
            # support for blocks
            if _data is None:
                _data = self._get_batch(index, materialize=materialize)
            return self._clone(data=_data)

    # @capture_provenance()
    def __getitem__(self, index):
        return self._get(index, materialize=False)

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

    def _is_batch_index(self, index):
        # np.ndarray indexed with a tuple of length 1 does not return an np.ndarray
        # so we match this behavior
        return not (
            isinstance(index, int) or (isinstance(index, tuple) and len(index) == 1)
        )

    def _translate_index(self, index):
        return translate_index(index, length=len(self))

    @staticmethod
    def _convert_to_batch_fn(
        function: Callable, with_indices: bool, materialize: bool = True, **kwargs
    ) -> callable:
        return convert_to_batch_column_fn(
            function=function,
            with_indices=with_indices,
            materialize=materialize,
            **kwargs,
        )

    def __len__(self):
        return self.full_length()

    def full_length(self):
        if self._data is None:
            return 0
        return len(self._data)

    def _repr_cell_(self, index) -> object:
        raise NotImplementedError

    def _get_default_formatter(self) -> "Formatter":
        # can't implement this as a class level property because then it will treat
        # the formatter as a method
        from meerkat.interactive.formatter import BasicFormatter

        return BasicFormatter()

    @property
    def formatter(self) -> "Formatter":
        return self._formatter

    @formatter.setter
    def formatter(self, formatter: "Formatter"):
        self._formatter = formatter

    def _repr_pandas_(self, max_rows: int = None) -> pd.Series:
        if max_rows is None:
            max_rows = meerkat.config.display.max_rows

        if len(self) > max_rows:
            col = pd.Series(
                [self._repr_cell(idx) for idx in range(max_rows // 2)]
                + [self._repr_cell(0)]
                + [
                    self._repr_cell(idx)
                    for idx in range(len(self) - max_rows // 2, len(self))
                ]
            )
        else:
            col = pd.Series([self._repr_cell(idx) for idx in range(len(self))])

        return col, self.formatter if self.formatter is None else self.formatter.html

    def _repr_html_(self, max_rows: int = None):
        # pd.Series objects do not implement _repr_html_
        if max_rows is None:
            max_rows = meerkat.config.display.max_rows

        if len(self) > max_rows:
            pd_index = np.concatenate(
                (
                    np.arange(max_rows // 2),
                    np.zeros(1),
                    np.arange(len(self) - max_rows // 2, len(self)),
                ),
            )
        else:
            pd_index = np.arange(len(self))

        col_name = f"({self.__class__.__name__})"
        col, formatter = self._repr_pandas_(max_rows=max_rows)
        df = col.to_frame(name=col_name)
        df = df.set_index(pd_index.astype(int))
        return df.to_html(
            max_rows=max_rows,
            formatters={col_name: formatter},
            escape=False,
        )

    def map(
        self,
        function: Callable,
        is_batched_fn: bool = False,
        batch_size: int = 1,
        inputs: Union[Mapping[str, str], Sequence[str]] = None,
        outputs: Union[Mapping[any, str], Sequence[str]] = None,
        output_type: Union[Mapping[str, type], type] = None,
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

    @capture_provenance()
    def filter(
        self,
        function: Callable,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: Optional[int] = 0,
        materialize: bool = True,
        # pbar: bool = False,
        **kwargs,
    ) -> Optional[Column]:
        """Filter the elements of the column using a function."""

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning it .")
            return self

        # Get some information about the function
        function_properties = self._inspect_function(
            function,
            with_indices,
            is_batched_fn=is_batched_fn,
            materialize=materialize,
            **kwargs,
        )
        assert function_properties.bool_output, "function must return boolean."

        # Map to get the boolean outputs and indices
        logger.info("Running `filter`, a new dataset will be returned.")
        outputs = self.map(
            function=function,
            with_indices=with_indices,
            # input_columns=input_columns,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            num_workers=num_workers,
            materialize=materialize,
            # pbar=pbar,
            **kwargs,
        )
        indices = np.where(outputs)[0]
        return self[indices]

    def sort(
        self, ascending: Union[bool, List[bool]] = True, kind: str = "quicksort"
    ) -> Column:
        """Return a sorted view of the column.

        Args:
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`. Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            AbstractColumn: A view of the column with the sorted data.
        """
        raise NotImplementedError

    def argsort(
        self, ascending: Union[bool, List[bool]] = True, kind: str = "quicksort"
    ) -> Column:
        """Return indices that would sorted the column.

        Args:
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`. Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            AbstractColumn: A view of the column with the sorted data.
        """
        raise NotImplementedError

    def sample(
        self,
        n: int = None,
        frac: float = None,
        replace: bool = False,
        weights: Union[str, np.ndarray] = None,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> Column:
        """Select a random sample of rows from Column. Roughly equivalent to
        ``sample`` in Pandas https://pandas.pydata.org/docs/reference/api/panda
        s.DataFrame.sample.html.

        Args:
            n (int): Number of samples to draw. If `frac` is specified, this parameter
                should not be passed. Defaults to 1 if `frac` is not passed.
            frac (float): Fraction of rows to sample. If `n` is specified, this
                parameter should not be passed.
            replace (bool): Sample with or without replacement. Defaults to False.
            weights (np.ndarray): Weights to use for sampling. If `None`
                (default), the rows will be sampled uniformly. If a numpy array, the
                sample will be weighted accordingly. If
                weights do not sum to 1 they will be normalized to sum to 1.
            random_state (Union[int, np.random.RandomState]): Random state or seed to
                use for sampling.

        Return:
            AbstractColumn: A random sample of rows from the DataFrame.
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

    def append(self, column: Column) -> None:
        # TODO(Sabri): implement a naive `ComposedColumn` for generic append and
        # implement specific ones for ListColumn, NumpyColumn etc.
        raise NotImplementedError

    @staticmethod
    def concat(columns: Sequence[Column]) -> None:
        # TODO(Sabri): implement a naive `ComposedColumn` for generic append and
        # implement specific ones for ListColumn, NumpyColumn etc.
        raise NotImplementedError

    def is_equal(self, other: Column) -> bool:
        """Tests whether two columns.

        Args:
            other (AbstractColumn): [description]
        """
        raise NotImplementedError()

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
        if (
            self._get_batch.__func__ == Column._get_batch
            and self._get.__func__ == Column._get
        ):
            return torch.utils.data.DataLoader(
                self.mz if materialize else self,
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
                self.mz if materialize else self,
                sampler=batch_indices,
                batch_size=None,
                batch_sampler=None,
                drop_last=drop_last_batch,
                num_workers=num_workers,
                *args,
                **kwargs,
            )

    @classmethod
    def get_writer(cls, mmap: bool = False, template: Column = None):
        from meerkat.writers.concat_writer import ConcatWriter

        if mmap:
            raise ValueError("Memmapping not supported with this column type.")
        else:
            return ConcatWriter(output_type=cls, template=template)

    Columnable = Union[Sequence, np.ndarray, pd.Series, torch.Tensor]

    @classmethod
    # @capture_provenance()
    def from_data(cls, data: Union[Columnable, Column]):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        return column(data)

    def head(self, n: int = 5) -> Column:
        """Get the first `n` examples of the column."""
        return self[:n]

    def tail(self, n: int = 5) -> Column:
        """Get the last `n` examples of the column."""
        return self[-n:]

    def to_pandas(self, allow_objects: bool = False) -> pd.Series:
        """Convert the column to a Pandas Series.

        If the column cannot be converted to a Pandas Series, this method will raise a
        `~meerkat.errors.ConversionError`.

        Returns:
            pd.Series: The column as a Pandas Series.
        """
        raise ConversionError(
            f"Cannot convert column of type {type(self)} to Pandas Series."
        )

    def to_arrow(self) -> pa.Array:
        """Convert the column to an Arrow Array.

        If the column cannot be converted to an Arrow Array, this method will raise a
        `~meerkat.errors.ConversionError`.

        Returns:
            pa.Array: The column as an Arrow Array.
        """
        raise ConversionError(
            f"Cannot convert column of type {type(self)} to Arrow Array."
        )

    def to_torch(self) -> torch.Tensor:
        """Convert the column to a PyTorch Tensor.

        If the column cannot be converted to a PyTorch Tensor, this method will raise a
        `~meerkat.errors.ConversionError`.

        Returns:
            torch.Tensor: The column as a PyTorch Tensor.
        """
        raise ConversionError(
            f"Cannot convert column of type {type(self)} to PyTorch Tensor."
        )

    def to_numpy(self) -> np.ndarray:
        """Convert the column to a Numpy array.

        If the column cannot be converted to a Numpy array, this method will raise a
        `~meerkat.errors.ConversionError`.

        Returns:
            np.ndarray: The column as a Numpy array.
        """
        raise ConversionError(
            f"Cannot convert column of type {type(self)} to Numpy array."
        )

    def _copy_data(self) -> object:
        return copy(self._data)

    def _view_data(self) -> object:
        return self._data

    @property
    def is_mmap(self):
        return False


def column(data: Sequence) -> Column:
    """Create a Meerkat column from data.

    The Meerkat column type is inferred from the type and structure of
    the data passed in.
    """
    from .scalar import ScalarColumn
    from .tensor import TensorColumn

    if isinstance(data, Column):
        # TODO: Need ton make this view but should decide where to do it exactly
        return data  # .view()

    if isinstance(data, pd.Series) or isinstance(data, pa.Array):
        return ScalarColumn(data)

    if torch.is_tensor(data):
        if len(data.shape) == 1:
            return ScalarColumn(data.cpu().detach().numpy())
        return TensorColumn(data)

    if isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            return ScalarColumn(data)
        return TensorColumn(data)

    if isinstance(data, Sequence):
        if len(data) != 0 and (
            isinstance(data[0], (np.ndarray, TensorColumn)) or torch.is_tensor(data[0])
        ):
            return TensorColumn(data)

        if len(data) != 0 and isinstance(data[0], (str, int, float, bool, np.generic)):
            from .scalar import ScalarColumn

            return ScalarColumn(data)

        from .object.base import ObjectColumn

        return ObjectColumn(data)
    else:
        raise ValueError(f"Cannot create column out of data of type {type(data)}")
