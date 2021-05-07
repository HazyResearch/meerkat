from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from mosaic.cells.abstract import AbstractCell
from mosaic.columns.abstract import AbstractColumn

logger = logging.getLogger(__name__)


class CellColumn(AbstractColumn):
    def __init__(
        self,
        cells: Sequence[AbstractCell] = None,
        materialize: bool = True,
        *args,
        **kwargs,
    ):
        super(CellColumn, self).__init__(
            data=cells,
            materialize=materialize,
            *args,
            **kwargs,
        )

    def _get_batch(self, indices: np.ndarray):
        if self.materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `.get`, otherwise the batch format specified by `self.collate`)
            return self.collate([self._data[i].get() for i in indices])

        else:
            return self.__class__(
                [self._data[i] for i in indices], materialize=self.materialize
            )

    @classmethod
    def from_cells(cls, cells: Sequence[AbstractCell], *args, **kwargs):
        return cls(cells=cells, *args, **kwargs)

    @property
    def cells(self):
        return self.data
