from __future__ import annotations

import logging
from typing import Sequence

import cytoolz as tz
import numpy as np

from meerkat.cells.abstract import AbstractCell
from meerkat.columns.abstract import AbstractColumn

logger = logging.getLogger(__name__)


class CellColumn(AbstractColumn):
    def __init__(
        self,
        cells: Sequence[AbstractCell] = None,
        *args,
        **kwargs,
    ):
        super(CellColumn, self).__init__(
            data=cells,
            *args,
            **kwargs,
        )

    def _get_cell(self, index: int, materialize: bool = True):
        cell = self._data[index]
        if materialize:
            return cell.get()
        else:
            return cell

    def _get_batch(self, indices: np.ndarray, materialize: bool = True):
        if materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `.get`, otherwise the batch format specified by `self.collate`)
            return self.collate([self._data[i].get() for i in indices])

        else:
            return [self._data[i] for i in indices]

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

    @classmethod
    def from_cells(cls, cells: Sequence[AbstractCell], *args, **kwargs):
        return cls(cells=cells, *args, **kwargs)

    @property
    def cells(self):
        return self.data

    def _repr_cell(self, index) -> object:
        return self.lz[index].__repr__()

    @staticmethod
    def concat(columns: Sequence[CellColumn]):
        return columns[0].__class__.from_cells(
            list(tz.concat([c.data for c in columns]))
        )

    def is_equal(self, other: AbstractColumn) -> bool:
        return (
            (self.__class__ == other.__class__)
            and (len(self) == len(other))
            and all([self.lz[idx] == other.lz[idx] for idx in range(len(self))])
        )
