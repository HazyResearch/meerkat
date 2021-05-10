from __future__ import annotations

import logging
from typing import Sequence

from mosaic.cells.imagepath import ImagePath
from mosaic.columns.cell_column import CellColumn

logger = logging.getLogger(__name__)


class ImageColumn(CellColumn):
    def __init__(
        self,
        filepaths: Sequence[str] = None,
        loader: callable = None,
        transform: callable = None,
        materialize: bool = True,
        *args,
        **kwargs,
    ):
        cells = [ImagePath(fp, transform=transform, loader=loader) for fp in filepaths]
        super(ImageColumn, self).__init__(
            cells=cells,
            materialize=materialize,
            *args,
            **kwargs,
        )
