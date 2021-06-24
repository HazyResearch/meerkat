from __future__ import annotations

import logging
from typing import Sequence

from meerkat.cells.volume import MedicalVolumeCell
from meerkat.columns.cell_column import CellColumn

logger = logging.getLogger(__name__)


class MedicalVolumeColumn(CellColumn):
    def __init__(self, *args, **kwargs):
        super(MedicalVolumeColumn, self).__init__(*args, **kwargs)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str] = None,
        loader: callable = None,
        transform: callable = None,
        *args,
        **kwargs,
    ):
        if loader is None:
            loader = MedicalVolumeCell.default_loader(filepaths)

        cells = [
            MedicalVolumeCell(fp, transform=transform, loader=loader)
            for fp in filepaths
        ]

        return cls(
            cells=cells,
            *args,
            **kwargs,
        )
