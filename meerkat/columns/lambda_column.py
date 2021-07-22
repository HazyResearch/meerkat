from __future__ import annotations

import logging
from typing import Collection, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import AbstractColumn
from meerkat.datapanel import DataPanel
from meerkat.mixins.cloneable import CloneableMixin

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


class LambdaColumn(CloneableMixin, AbstractColumn):
    def __init__(
        self,
        data: Union[DataPanel, AbstractColumn],
        fn: callable = None,
        output_type: type = None,
        *args,
        **kwargs
    ):
        super(LambdaColumn, self).__init__(data.view(), *args, **kwargs)
        if fn is not None:
            self.fn = fn
        self._output_type = output_type

    def __getattr__(self, name):
        if not self._output_type:
            raise AttributeError(name)

        data = self[:2]
        if not hasattr(data, name):
            raise AttributeError(name)

        data = self[:]
        return data.__getattr__(name)

    def fn(self, data: object):
        """Subclasses like `ImageColumn` should be able to implement their own
        version."""
        raise NotImplementedError

    @property
    def data(self):
        """The LambdaColumn."""
        return None

    @property
    def visible_rows(self):
        return None

    @visible_rows.setter
    def visible_rows(self, indices: Optional[Sequence]):
        if isinstance(self._data, AbstractColumn):
            self._data.visible_rows = indices
        else:
            for column in self._data._data.values():
                column.visible_rows = indices

    def _get_cell(self, index: int, materialize: bool = True):
        if materialize:
            return self.fn(self._data._get(index, materialize=True))
        else:
            return LambdaCell(
                fn=self.fn, data=self._data._get(index, materialize=False)
            )

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
            return self._clone(data=self._data.lz[indices])

    def _clone_kwargs(self):
        return {"fn": self.fn, "data": self._data}

    def _repr_pandas_(
        self,
    ) -> pd.Series:
        return pd.Series([self.fn.__repr__] * len(self))

    @classmethod
    def _state_keys(cls) -> Collection:
        return super()._state_keys() | {"fn"}

    @staticmethod
    def concat(columns: Sequence[LambdaColumn]):

        # TODO: raise a warning if the functions don't match
        return columns[0]._clone(columns[0]._data.concat([c._data for c in columns]))
