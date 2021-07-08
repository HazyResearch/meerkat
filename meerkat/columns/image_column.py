from __future__ import annotations

import logging
from typing import Collection, Sequence

from meerkat.cells.imagepath import ImagePath
from meerkat.columns.cell_column import CellColumn
from meerkat.columns.lambda_column import LambdaColumn
from meerkat.columns.numpy_column import NumpyArrayColumn

logger = logging.getLogger(__name__)


class ImageColumn(LambdaColumn):
    def __init__(
        self,
        data: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        *args,
        **kwargs,
    ):
        super(ImageColumn, self).__init__(NumpyArrayColumn(data), *args, **kwargs)
        self.loader = loader
        self.transform = transform

    @property
    def data(self):
        """The LambdaColumn"""
        return self._data.data

    def fn(self, filepath: str):
        image = self.loader(filepath)
        if self.transform is not None:
            image = self.transform(image)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str],
        loader: callable = None,
        transform: callable = None,
        *args,
        **kwargs,
    ):
        return cls(
            data=filepaths,
            loader=loader,
            transform=transform,
            *args,
            **kwargs,
        )

    @classmethod
    def _state_keys(cls) -> Collection:
        return (super()._state_keys() | {"transform", "loader"}) - {"fn"}

    def _clone_kwargs(self):
        default_kwargs = super()._clone_kwargs()
        default_kwargs.update(
            {
                "loader": self.loader,
                "transform": self.transform,
            }
        )
        return default_kwargs


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
