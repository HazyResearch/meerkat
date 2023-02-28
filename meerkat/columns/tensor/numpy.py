from __future__ import annotations

import abc
import functools
import logging
import numbers
import os
import shutil
from mmap import mmap
from typing import TYPE_CHECKING, Any, Callable, List, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.core._exceptions import UFuncTypeError
from yaml.representer import Representer

from meerkat.block.abstract import BlockView
from meerkat.block.numpy_block import NumPyBlock
from meerkat.columns.abstract import Column
from meerkat.mixins.aggregate import AggregationError
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.writers.concat_writer import ConcatWriter

from .abstract import TensorColumn

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch


Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, np.ndarray):
            return NumPyTensorColumn(out)
        else:
            return out

    return wrapper


class NumPyTensorColumn(
    TensorColumn,
    np.lib.mixins.NDArrayOperatorsMixin,
):

    block_class: type = NumPyBlock

    def __init__(
        self,
        data: Sequence,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, NumPyBlock):
                raise ValueError(
                    "Cannot create `NumpyArrayColumn` from a `BlockView` not "
                    "referencing a `NumpyBlock`."
                )
        elif not isinstance(data, np.memmap) and not isinstance(data, np.ndarray):
            if len(data) > 0 and isinstance(data[0], np.ndarray):
                data = np.stack(data)
            else:
                data = np.asarray(data)

        super(NumPyTensorColumn, self).__init__(data=data, *args, **kwargs)

    # TODO (sabri): need to support str here
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (NumPyTensorColumn,)) and not (
                # support for at index
                method == "at"
                and isinstance(x, list)
            ):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x.data if isinstance(x, NumPyTensorColumn) else x for x in inputs
        )

        if out:
            kwargs["out"] = tuple(
                x.data if isinstance(x, NumPyTensorColumn) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self._clone(data=result)

    def __getattr__(self, name):
        try:
            out = getattr(object.__getattribute__(self, "data"), name)
            if isinstance(out, Callable):
                return getattr_decorator(out)
            else:
                return out
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    @classmethod
    def from_array(cls, data: np.ndarray, *args, **kwargs):
        return cls(data=data, *args, **kwargs)

    def _set_batch(self, indices, values):
        self._data[indices] = values

    def _get(self, index, materialize: bool = True):
        index = NumPyBlock._convert_index(index)
        data = self._data[index]
        if self._is_batch_index(index):
            # only create a numpy array column
            return self._clone(data=data)
        else:
            return data

    def _copy_data(self) -> object:
        return self._data.copy()

    def _view_data(self) -> object:
        return self._data

    @property
    def is_mmap(self):
        # important to check if .base is a python mmap object, since a view of a mmap
        # is also a memmap object, but should not be symlinked or copied
        if len(self.data.shape) == 1:
            # if the data is a 1D array, then their is a level of indirection to the
            # the base object because we did a reshape to add an extra dimension
            return isinstance(self.data, np.memmap) and isinstance(
                self._block.data.base.base, mmap
            )
        else:
            return isinstance(self.data, np.memmap) and isinstance(
                self._block.data.base, mmap
            )

    def _write_data(self, path: str, link: bool = True) -> None:
        path = os.path.join(path, "data.npy")
        # important to check if .base is a python mmap object, since a view of a mmap
        # is also a memmap object, but should not be symlinked
        if self.is_mmap:
            if link:
                os.symlink(self.data.filename, path)
            else:
                shutil.copy(self.data.filename, path)
        else:
            np.save(path, self.data)

    @staticmethod
    def _read_data(path: str, mmap=False, *args, **kwargs) -> np.ndarray:
        data_path = os.path.join(path, "data.npy")

        if mmap:
            return np.load(data_path, mmap_mode="r")
        return np.load(data_path)

    @classmethod
    def concat(cls, columns: Sequence[NumPyTensorColumn]):
        data = np.concatenate([c.data for c in columns])
        return columns[0]._clone(data=data)

    def is_equal(self, other: Column) -> bool:
        if other.__class__ != self.__class__:
            return False
        return np.array_equal(self.data, other.data, equal_nan=True)

    @classmethod
    def get_writer(cls, mmap: bool = False, template: Column = None):
        if mmap:
            from meerkat.writers.numpy_writer import NumpyMemmapWriter

            return NumpyMemmapWriter()
        else:
            return ConcatWriter(template=template, output_type=NumPyTensorColumn)

    def _repr_cell(self, index) -> object:
        if len(self.shape) > 1:
            if len(self.shape) == 2 and self.shape[1] < 5:
                return self[index]
            return f"np.ndarray(shape={self.shape[1:]})"
        else:
            return self[index]

    def _get_default_formatters(self):
        from meerkat.interactive.formatter import (
            NumberFormatterGroup,
            TensorFormatterGroup,
            TextFormatterGroup,
        )

        if len(self) == 0:
            return NumberFormatterGroup()

        if len(self.shape) > 1:
            return TensorFormatterGroup(dtype=str(self.dtype))

        if self.dtype.type is np.str_:
            return TextFormatterGroup()

        cell = self.data[0]
        if isinstance(cell, np.generic):
            return NumberFormatterGroup(dtype=type(cell.item()).__name__)

        return TextFormatterGroup()

    def _is_valid_primary_key(self):
        if self.dtype.kind == "f":
            # can't use floats as primary keys
            return False

        if len(self.shape) != 1:
            # can't use multidimensional arrays as primary keys
            return False
        return len(np.unique(self.data)) == len(self)

    def _keyidx_to_posidx(self, keyidx: Any) -> int:
        # TODO(sabri): when we implement indices, we should use them here if we have
        # one
        where_result = np.where(self.data == keyidx)
        if len(where_result[0]) == 0:
            raise KeyError(f"keyidx {keyidx} not found in column.")

        posidx = where_result[0][0]
        return int(posidx)

    def _keyidxs_to_posidxs(self, keyidxs: Sequence[Any]) -> np.ndarray:
        posidxs = np.where(np.isin(self.data, keyidxs))[0]

        diff = np.setdiff1d(keyidxs, self.data[posidxs])
        if len(diff) > 0:
            raise KeyError(f"Key indexes {diff} not found in column.")

        return posidxs

    def sort(
        self,
        ascending: Union[bool, List[bool]] = True,
        axis: int = -1,
        kind: str = "quicksort",
        order: Union[str, List[str]] = None,
    ) -> NumPyTensorColumn:
        """Return a sorted view of the column.

        Args:
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`. Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            Column: A view of the column with the sorted data.
        """
        # calls argsort() function to retrieve ordered indices
        sorted_index = self.argsort(ascending=ascending, kind=kind)
        return self[sorted_index]

    def argsort(
        self, ascending: bool = True, kind: str = "quicksort"
    ) -> NumPyTensorColumn:
        """Return indices that would sorted the column.

        Args:
            ascending (bool): Whether to sort in ascending or
                descending order.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            NumpySeriesColumn: A view of the column with the sorted data.

        For now! Raises error when shape of input array is more than one error.
        """
        num_columns = len(np.shape(self))
        # Raise error if array has more than one column
        if num_columns > 1:
            idxs = np.lexsort(self.data)
        else:
            idxs = np.argsort(self.data, axis=0, kind=kind, order=None)

        if not ascending:
            idxs = idxs[::-1]

        return idxs

    def to_torch(self) -> "torch.Tensor":
        return torch.tensor(self.data)

    def to_pandas(self, allow_objects: bool = True) -> pd.Series:
        if len(self.shape) == 1:
            return pd.Series(self.data)
        elif allow_objects:
            # can only create a 1-D series
            return pd.Series([self[int(idx)] for idx in range(len(self))])
        else:
            return super().to_pandas()

    def to_arrow(self) -> pa.Array:
        if len(self.shape) == 1:
            return pa.array(self.data)
        else:
            return super().to_arrow()

    def to_numpy(self) -> np.ndarray:
        return self.data

    def to_json(self) -> List[Any]:
        return self.data.tolist()

    @classmethod
    def from_npy(
        cls,
        path,
        mmap_mode=None,
        allow_pickle=False,
        fix_imports=True,
        encoding="ASCII",
    ):
        data = np.load(
            path,
            mmap_mode=mmap_mode,
            allow_pickle=allow_pickle,
            fix_imports=fix_imports,
            encoding=encoding,
        )
        return cls(data)

    def mean(
        self, axis: int = None, keepdims: bool = False, **kwargs
    ) -> NumPyTensorColumn:
        try:
            return self.data.mean(axis=axis, keepdims=keepdims, **kwargs)
        except (UFuncTypeError, TypeError):
            raise AggregationError(
                "Cannot apply mean aggregation to NumPy array with "
                f" dtype '{self.data.dtype}'."
            )
