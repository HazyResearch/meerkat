from __future__ import annotations

import abc
import functools
import logging
import numbers
import os
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
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
from meerkat.columns.abstract import AbstractColumn

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, pd.Series):
            return PandasSeriesColumn(out)
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
                return PandasSeriesColumn(attr)
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


class PandasSeriesColumn(
    AbstractColumn,
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
        elif not isinstance(data, pd.Series):
            data = pd.Series(data)

        super(PandasSeriesColumn, self)._set_data(data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (PandasSeriesColumn,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(
            x.data if isinstance(x, PandasSeriesColumn) else x for x in inputs
        )
        if out:
            kwargs["out"] = tuple(
                x.data if isinstance(x, PandasSeriesColumn) else x for x in out
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
    def concat(cls, columns: Sequence[PandasSeriesColumn]):
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

    def _repr_pandas_(self) -> pd.Series:
        return self.data

    def to_tensor(self) -> torch.Tensor:
        """Use `column.to_tensor()` instead of `torch.tensor(column)`, which is
        very slow."""
        dtype = self.data.values.dtype
        if not np.issubdtype(dtype, np.number):
            raise ValueError(
                f"Cannot convert `PandasSeriesColumn` with dtype={dtype} to tensor."
            )

        # TODO (Sabri): understand why `torch.tensor(column)` is so slow
        return torch.tensor(self.data.values)

    def is_equal(self, other: AbstractColumn) -> bool:
        if other.__class__ != self.__class__:
            return False
        return (self.data.values == other.data.values).all()

    def to_pandas(self) -> pd.Series:
        return self.data
