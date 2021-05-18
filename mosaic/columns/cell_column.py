from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from mosaic.cells.abstract import AbstractCell
from mosaic.columns.abstract import AbstractColumn

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
            return self.__class__([self._data[i] for i in indices])

    @classmethod
    def from_cells(cls, cells: Sequence[AbstractCell], *args, **kwargs):
        return cls(cells=cells, *args, **kwargs)

    @property
    def cells(self):
        if self.visible_rows is None:
            return self.data
        else:
            return [self.data[i] for i in self.visible_rows]

    def _repr_pandas_(
        self,
    ) -> pd.Series:
        return pd.Series([cell.__repr__() for cell in self.cells])
