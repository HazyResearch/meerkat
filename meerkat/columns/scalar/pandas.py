from __future__ import annotations

import abc
import functools
import logging
import numbers
import os
from typing import TYPE_CHECKING, Any, Callable, List, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.accessor import CachedAccessor
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.dtypes.common import (
    is_categorical_dtype,
    is_datetime64_dtype,
    is_datetime64tz_dtype,
    is_period_dtype,
    is_timedelta64_dtype,
)
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.indexes.accessors import (
    CombinedDatetimelikeProperties,
    DatetimeProperties,
    PeriodProperties,
    TimedeltaProperties,
)
from pandas.core.strings import StringMethods
from yaml.representer import Representer

from meerkat.block.abstract import BlockView
from meerkat.block.pandas_block import PandasBlock
from meerkat.columns.abstract import Column
from meerkat.interactive.formatter.base import Formatter
from meerkat.mixins.aggregate import AggregationError
from meerkat.tools.lazy_loader import LazyLoader

from .abstract import ScalarColumn

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, pd.Series):
            return PandasScalarColumn(out)
        elif isinstance(out, pd.DataFrame):
            from meerkat import DataFrame

            # column names must be str in meerkat
            out = out.rename(mapper=str, axis="columns")
            return DataFrame.from_pandas(out)
        else:
            return out

    return wrapper


class _ReturnColumnMixin:
    def __getattribute__(self, name):
        try:
            attr = super().__getattribute__(name)
            if isinstance(attr, Callable):
                return getattr_decorator(attr)
            elif isinstance(attr, pd.Series):
                return PandasScalarColumn(attr)
            elif isinstance(attr, pd.DataFrame):
                from meerkat import DataFrame

                return DataFrame.from_pandas(attr)
            else:
                return attr
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


class _MeerkatStringMethods(_ReturnColumnMixin, StringMethods):
    pass


class _MeerkatDatetimeProperties(_ReturnColumnMixin, DatetimeProperties):
    pass


class _MeerkatTimedeltaProperties(_ReturnColumnMixin, TimedeltaProperties):
    pass


class _MeerkatPeriodProperties(_ReturnColumnMixin, PeriodProperties):
    pass


class _MeerkatCategoricalAccessor(_ReturnColumnMixin, CategoricalAccessor):
    pass


class _MeerkatCombinedDatetimelikeProperties(CombinedDatetimelikeProperties):
    def __new__(cls, data: pd.Series):
        # CombinedDatetimelikeProperties isn't really instantiated. Instead
        # we need to choose which parent (datetime or timedelta) is
        # appropriate. Since we're checking the dtypes anyway, we'll just
        # do all the validation here.

        if not isinstance(data, ABCSeries):
            raise TypeError(
                f"cannot convert an object of type {type(data)} to a datetimelike index"
            )

        orig = data if is_categorical_dtype(data.dtype) else None
        if orig is not None:
            data = data._constructor(
                orig.array,
                name=orig.name,
                copy=False,
                dtype=orig._values.categories.dtype,
            )

        if is_datetime64_dtype(data.dtype):
            obj = _MeerkatDatetimeProperties(data, orig)
        elif is_datetime64tz_dtype(data.dtype):
            obj = _MeerkatDatetimeProperties(data, orig)
        elif is_timedelta64_dtype(data.dtype):
            obj = _MeerkatTimedeltaProperties(data, orig)
        elif is_period_dtype(data.dtype):
            obj = _MeerkatPeriodProperties(data, orig)
        else:
            raise AttributeError("Can only use .dt accessor with datetimelike values")

        return obj


class PandasScalarColumn(
    ScalarColumn,
    np.lib.mixins.NDArrayOperatorsMixin,
):
    block_class: type = PandasBlock

    _HANDLED_TYPES = (np.ndarray, numbers.Number, str)

    str = CachedAccessor("str", _MeerkatStringMethods)
    dt = CachedAccessor("dt", _MeerkatCombinedDatetimelikeProperties)
    cat = CachedAccessor("cat", _MeerkatCategoricalAccessor)
    # plot = CachedAccessor("plot", pandas.plotting.PlotAccessor)
    # sparse = CachedAccessor("sparse", SparseAccessor)

    def _set_data(self, data: object):
        if isinstance(data, BlockView):
            if not isinstance(data.block, PandasBlock):
                raise ValueError(
                    "Cannot create `PandasSeriesColumn` from a `BlockView` not "
                    "referencing a `PandasBlock`."
                )
        elif isinstance(data, pd.Series):
            # Force the index to be contiguous so that comparisons between different
            # pandas series columns are always possible.
            data = data.reset_index(drop=True)
        else:
            data = pd.Series(data)

        super(PandasScalarColumn, self)._set_data(data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (PandasScalarColumn,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x.data if isinstance(x, PandasScalarColumn) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x.data if isinstance(x, PandasScalarColumn) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)  # pragma: no cover
        elif method == "at":
            # no return value
            return None  # pragma: no cover
        else:
            # one return value
            return type(self)(result)

    def __getattr__(self, name):
        if name == "__getstate__" or name == "__setstate__":
            # for pickle, it's important to raise an attribute error if __getstate__
            # or __setstate__ is called. Without this, pickle will use the __setstate__
            # and __getstate__ of the underlying pandas Series
            raise AttributeError()
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

    def _get(self, index, materialize: bool = True):
        index = self.block_class._convert_index(index)

        data = self._data.iloc[index]
        if self._is_batch_index(index):
            # only create a numpy array column
            return self._clone(data=data)
        else:
            return data

    def _set_cell(self, index, value):
        self._data.iloc[index] = value

    def _set_batch(self, indices, values):
        self._data.iloc[indices] = values

    @classmethod
    def concat(cls, columns: Sequence[PandasScalarColumn]):
        data = pd.concat([c.data for c in columns])
        return columns[0]._clone(data=data)

    def _write_data(self, path: str) -> None:
        data_path = os.path.join(path, "data.pd")
        self.data.to_pickle(data_path)

    @staticmethod
    def _read_data(
        path: str,
    ):
        data_path = os.path.join(path, "data.pd")

        # Load in the data
        return pd.read_pickle(data_path)

    def _repr_cell(self, index) -> object:
        return self[index]

    def _get_default_formatter(self) -> Formatter:
        # can't implement this as a class level property because then it will treat
        # the formatter as a method
        from meerkat.interactive.app.src.lib.component.core.scalar import (
            ScalarFormatter,
        )
        from meerkat.interactive.app.src.lib.component.core.text import TextFormatter

        if len(self) == 0:
            return ScalarFormatter()

        if self.dtype == object:
            return TextFormatter()

        if self.dtype == pd.StringDtype:
            return TextFormatter()

        cell = self[0]
        if isinstance(cell, np.generic):
            return ScalarFormatter(dtype= type(cell.item()).__name__)

        return ScalarFormatter()

    def _is_valid_primary_key(self):
        return self.data.is_unique

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
        self, ascending: Union[bool, List[bool]] = True, kind: str = "quicksort"
    ) -> PandasScalarColumn:
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
        # calls argsort() function to retrieve ordered indices
        sorted_index = self.argsort(ascending, kind)
        return self[sorted_index]

    def argsort(
        self, ascending: bool = True, kind: str = "quicksort"
    ) -> PandasScalarColumn:
        """Return indices that would sorted the column.

        Args:
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`. Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            PandasSeriesColumn: A view of the column with the sorted data.

         For now! Raises error when shape of input array is more than one error.
        """
        num_columns = len(self.shape)
        # Raise error if array has more than one column
        if num_columns > 1:
            raise Exception("No implementation for array with more than one column.")

        # returns indices of descending order of array
        if not ascending:
            return (-1 * self.data).argsort(kind=kind)

        # returns indices of ascending order of array
        return self.data.argsort(kind=kind)

    def to_tensor(self) -> "torch.Tensor":
        """Use `column.to_tensor()` instead of `torch.tensor(column)`, which is
        very slow."""
        dtype = self.data.values.dtype
        if not np.issubdtype(dtype, np.number):
            raise ValueError(
                f"Cannot convert `PandasSeriesColumn` with dtype={dtype} to tensor."
            )

        # TODO (Sabri): understand why `torch.tensor(column)` is so slow
        return torch.tensor(self.data.values)

    def to_numpy(self) -> "torch.Tensor":
        return self.values

    def to_pandas(self, allow_objects: bool = False) -> pd.Series:
        return self.data.reset_index(drop=True)

    def to_arrow(self) -> pa.Array:
        return pa.array(self.data.values)

    def is_equal(self, other: Column) -> bool:
        if other.__class__ != self.__class__:
            return False
        return (self.data.values == other.data.values).all()

    def mean(self, skipna: bool = True):
        try:
            return self.data.mean(skipna=skipna)
        except TypeError:
            raise AggregationError(
                "Cannot apply mean aggregation to Pandas Series with "
                f" dtype '{self.data.dtype}'."
            )


PandasSeriesColumn = PandasScalarColumn
