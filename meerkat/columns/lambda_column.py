from __future__ import annotations

import logging
import os
from typing import Collection, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import AbstractColumn
from meerkat.datapanel import DataPanel

logger = logging.getLogger(__name__)


class LambdaCell(AbstractCell):
    def __init__(
        self,
        fn: callable = None,
        data: any = None,
    ):
        self.fn = fn
        self._data = data

    @property
    def data(self) -> object:
        """Get the data associated with this cell."""
        return self._data

    def get(self, *args, **kwargs):
        if isinstance(self.data, AbstractCell):
            return self.fn(self.data.get())
        elif isinstance(self.data, Mapping):
            return self.fn(
                {
                    k: v.get() if isinstance(v, AbstractCell) else v
                    for k, v in self.data.items()
                }
            )
        else:
            return self.fn(self.data)

    def __eq__(self, other):
        return (
            (other.__class__ == self.__class__)
            and (self.data == other.data)
            and (self.fn == other.fn)
        )


class LambdaColumn(AbstractColumn):
    def __init__(
        self,
        data: Union[DataPanel, AbstractColumn],
        fn: callable = None,
        output_type: type = None,
        *args,
        **kwargs,
    ):
        super(LambdaColumn, self).__init__(data.view(), *args, **kwargs)
        if fn is not None:
            self.fn = fn
        self._output_type = output_type

    # TODO (Sabri): reconsider whether this is important functionality. it's not clear
    # to me that this is that useful.
    # def __getattr__(self, name):
    #     if not self._output_type:
    #         raise AttributeError(name)

    #     data = self[:2]
    #     if not hasattr(data, name):
    #         raise AttributeError(name)

    #     data = self[:]
    #     return data.__getattr__(name

    def _set(self, index, value):
        raise ValueError("Cannot setitem on a `LambdaColumn`.")

    def fn(self, data: object):
        """Subclasses like `ImageColumn` should be able to implement their own
        version."""
        raise NotImplementedError

    def _create_cell(self, data: object) -> LambdaCell:
        return LambdaCell(fn=self.fn, data=data)

    def _get_cell(self, index: int, materialize: bool = True):
        if materialize:
            return self.fn(self._data._get(index, materialize=True))
        else:
            return self._create_cell(data=self._data._get(index, materialize=False))

    def _get_batch(self, indices: np.ndarray, materialize: bool = True):
        if materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `.get`, otherwise the batch format specified by `self.collate`)
            data = self.collate(
                [self._get_cell(int(i), materialize=True) for i in indices]
            )
            if self._output_type is not None:
                data = self._output_type(data)
            return data
        else:
            return self._data.lz[indices]

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
            if materialize:
                # materialize could change the data in unknown ways, cannot clone
                return self.__class__.from_data(data=_data)
            else:
                return self._clone(data=_data)

    def _clone_kwargs(self):
        return {"fn": self.fn, "data": self._data}

    def _repr_pandas_(self) -> pd.Series:
        cell_repr = self._repr_cell_()
        if len(self) <= pd.options.display.max_rows:
            return pd.Series([cell_repr] * len(self))
        else:
            # faster than creating a full pandas series
            series = pd.Series(np.empty(len(self)), copy=False)
            series.iloc[: pd.options.display.min_rows] = cell_repr
            series.iloc[-pd.options.display.min_rows :] = cell_repr
            return series

    def _repr_cell_(self):
        name = getattr(self.fn, "__qualname__", repr(self.fn))
        return f"LambdaCell(fn={name})"

    @classmethod
    def _state_keys(cls) -> Collection:
        return super()._state_keys() | {"fn", "_output_type"}

    @staticmethod
    def concat(columns: Sequence[LambdaColumn]):
        # TODO: raise a warning if the functions don't match
        return columns[0]._clone(columns[0]._data.concat([c._data for c in columns]))

    def _write_data(self, path):
        # TODO (Sabri): avoid redundant writes in dataframes
        return self.data.write(os.path.join(path, "data"))

    def is_equal(self, other: AbstractColumn) -> bool:
        if other.__class__ != self.__class__:
            return False
        if self.fn != other.fn:
            return False

        return self.data.is_equal(other.data)

    @staticmethod
    def _read_data(path: str):
        # TODO (Sabri): make this work for dataframes underlying the lambda column
        return AbstractColumn.read(os.path.join(path, "data"))
