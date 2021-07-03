from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from meerkat.cells.imagepath import ImagePath
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.cell_column import CellColumn
from meerkat.columns.numpy_column import NumpyArrayColumn

logger = logging.getLogger(__name__)


class ImageColumn(AbstractColumn):
    def __init__(
        self,
        filepaths: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        *args,
        **kwargs,
    ):
        super(ImageColumn, self).__init__(filepaths, *args, **kwargs)
        self.loader = loader
        self.transform = transform

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str],
        loader: callable = None,
        transform: callable = None,
        *args,
        **kwargs,
    ):
        filepaths = pd.Series(list(filepaths)).reset_index(drop=True)
        return cls(
            filepaths=filepaths,
            loader=loader,
            transform=transform,
            *args,
            **kwargs,
        )

    def _get_cell(self, index: int, materialize: bool = True):
        cell = ImagePath(
            self._data.iloc[index], loader=self.loader, transform=self.transform
        )

        if materialize:
            return cell.get()
        else:
            return cell

    def _get_batch(self, indices: np.ndarray, materialize: bool = True):
        if materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `.get`, otherwise the batch format specified by `self.collate`)
            return self.collate(
                [self._get_cell(idx, materialize=materialize) for idx in indices]
            )

        else:
            data = self._data.iloc[indices].reset_index(drop=True)
            return self.__class__(data, loader=self.loader, transform=self.transform)

    @property
    def data(self):
        """Get the underlying data (excluding invisible rows).

        To access underlying data with invisible rows, use `_data`.
        """
        if self.visible_rows is not None:
            return self._data.iloc[self.visible_rows]
        else:
            return self._data

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return super()._state_keys() | {"transform", "loader"}

    def _repr_pandas_(
        self,
    ) -> pd.Series:
        return "ImagePathCell(" + self.data + ")"

    @staticmethod
    def concat(columns: Sequence[ImageColumn]):
        loader, transform = (
            columns[0].loader,
            columns[0].transform,
        )

        return ImageColumn(
            filepaths=pd.concat([c.data for c in columns]),
            loader=loader,
            transform=transform,
        )


class ImageCellColumn(CellColumn):
    def __init__(self, *args, **kwargs):
        super(ImageCellColumn, self).__init__(*args, **kwargs)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str] = None,
        loader: callable = None,
        transform: callable = None,
        *args,
        **kwargs,
    ):
        cells = [ImagePath(fp, transform=transform, loader=loader) for fp in filepaths]

        return cls(
            cells=cells,
            *args,
            **kwargs,
        )
