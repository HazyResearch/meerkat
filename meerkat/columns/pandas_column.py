from __future__ import annotations

import abc
import functools
import logging
import numbers
import os
from typing import Any, Callable, Sequence

import dill
import numpy as np
import pandas as pd
import torch
import yaml
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
from pandas.core.strings.accessor import StringMethods
from yaml.representer import Representer

from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.cloneable import CloneableMixin

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
    def __init__(
        self,
        data: Sequence = None,
        dtype: str = None,
        *args,
        **kwargs,
    ):
        if isinstance(data, pd.Series):
            data = data if dtype is None else data.astype(dtype)
        elif data is not None:
            data = pd.Series(data, dtype=dtype)
        super(PandasSeriesColumn, self).__init__(data=data, *args, **kwargs)

    _HANDLED_TYPES = (np.ndarray, numbers.Number, str)

    str = CachedAccessor("str", _MeerkatStringMethods)
    dt = CachedAccessor("dt", _MeerkatCombinedDatetimelikeProperties)
    cat = CachedAccessor("cat", _MeerkatCategoricalAccessor)
    # plot = CachedAccessor("plot", pandas.plotting.PlotAccessor)
    # sparse = CachedAccessor("sparse", SparseAccessor)

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
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
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
        return self._data.iloc[index]

    def _get_batch(self, indices, materialize: bool = True):
        return self._data.iloc[indices]

    def _set_cell(self, index, value):
        self._data[index] = value

    def _set_batch(self, indices, values):
        self._data.iloc[indices] = values

    @classmethod
    def concat(cls, columns: Sequence[PandasSeriesColumn]):
        data = pd.concat([c.data for c in columns])
        if issubclass(cls, CloneableMixin):
            return columns[0]._clone(data=data)
        return cls.from_array(data)

    @classmethod
    def read(
        cls, path: str, mmap=False, dtype=None, shape=None, *args, **kwargs
    ) -> PandasSeriesColumn:
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Load in the metadata: only available if the array was stored by meerkat
        metadata_path = os.path.join(path, "meta.yaml")
        if os.path.exists(metadata_path):
            metadata = dict(yaml.load(open(metadata_path), Loader=yaml.FullLoader))
            assert metadata["dtype"] == cls

        # If the path doesn't exist, assume that `path` points to the `.npy` file
        data_path = os.path.join(path, "data.pd")
        if not os.path.exists(data_path):
            data_path = path

        # Load in the data
        data = pd.read_pickle(data_path)

        col = cls(data)

        state_path = os.path.join(path, "state.dill")
        if os.path.exists(state_path):
            state = dill.load(open(state_path, "rb"))
            col.__dict__.update(state)
        return col

    def write(self, path: str, **kwargs) -> None:
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the column state
        state = self.get_state()
        _data = state["_data"]

        # Remove the data key and put the rest of `state` into a metadata dict
        del state["_data"]
        metadata = {
            "dtype": type(self),
            "len": len(self),
            **self.metadata,
        }

        # Get the paths where metadata and data should be stored
        metadata_path = os.path.join(path, "meta.yaml")
        state_path = os.path.join(path, "state.dill")
        data_path = os.path.join(path, "data.pd")

        # Saving all cell data in a single pickle file
        _data.to_pickle(data_path)

        # Saving the metadata as a yaml
        yaml.dump(metadata, open(metadata_path, "w"))
        dill.dump(state, open(state_path, "wb"))

    def _repr_pandas_(self) -> pd.Series:
        return self.data

    def to_tensor(self) -> torch.Tensor:
        """Use `column.to_tensor()` instead of `torch.tensor(column)`, which is
        very slow."""
        # TODO (Sabri): understand why `torch.tensor(column)` is so slow
        return torch.tensor(self.data)

    def to_pandas(self) -> pd.Series:
        return self.data
