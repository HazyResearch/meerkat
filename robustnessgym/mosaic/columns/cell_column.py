from __future__ import annotations

import logging
from typing import Sequence

from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.columns.abstract import AbstractColumn

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

    @classmethod
    def from_cells(cls, cells: Sequence[AbstractCell], *args, **kwargs):
        return cls(cells=cells, *args, **kwargs)

    @property
    def cells(self):
        return self.data
