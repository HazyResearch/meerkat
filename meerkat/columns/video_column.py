from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from meerkat.cells.video import VideoCell
from meerkat.columns.cell_column import CellColumn

logger = logging.getLogger(__name__)


class VideoColumn(CellColumn):
    """Interface for creating a CellColumn from VideoCell objects."""

    def __init__(self, *args, **kwargs):
        super(VideoColumn, self).__init__(*args, **kwargs)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Optional[Sequence[str]] = None,
        time_dim: Optional[int] = 1,
        # TODO: add different loaders to VideoCell
        transform: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        cells = [
            VideoCell(fp, time_dim=time_dim, transform=transform) for fp in filepaths
        ]
        return cls(
            cells=cells,
            *args,
            **kwargs,
        )
