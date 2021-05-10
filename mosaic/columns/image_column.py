from __future__ import annotations

import logging
from typing import Sequence

from mosaic.cells.imagepath import ImagePath
from mosaic.columns.cell_column import CellColumn

logger = logging.getLogger(__name__)


class ImageColumn(CellColumn):
    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str] = None,
        loader: callable = None,
        transform: callable = None,
        materialize: bool = True,
        *args,
        **kwargs,
    ):
        cells = [ImagePath(fp, transform=transform, loader=loader) for fp in filepaths]

        return cls(
            cells=cells,
            materialize=materialize,
            *args,
            **kwargs,
        )
