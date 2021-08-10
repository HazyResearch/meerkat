from __future__ import annotations

import logging
from typing import Collection, Sequence

import pandas as pd

from meerkat.cells.imagepath import ImagePath
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.cell_column import CellColumn
from meerkat.columns.lambda_column import LambdaCell, LambdaColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.tools.lazy_loader import LazyLoader

folder = LazyLoader("torchvision.datasets.folder")

logger = logging.getLogger(__name__)


class ImageCell(LambdaCell):
    def __init__(
        self,
        transform: callable = None,
        loader: callable = None,
        data: str = None,
    ):
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform
        self._data = data

    def fn(self, filepath: str):
        image = self.loader(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __eq__(self, other):
        return (
            (other.__class__ == self.__class__)
            and (self.data == other.data)
            and (self.transform == other.transform)
            and (self.loader == other.loader)
        )


class ImageColumn(LambdaColumn):
    def __init__(
        self,
        data: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        *args,
        **kwargs,
    ):
        super(ImageColumn, self).__init__(PandasSeriesColumn(data), *args, **kwargs)
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform

    def _create_cell(self, data: object) -> ImageCell:
        return ImageCell(data=data, loader=self.loader, transform=self.transform)

    def fn(self, filepath: str):
        image = self.loader(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image

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
    def default_loader(cls, *args, **kwargs):
        return folder.default_loader(*args, **kwargs)

    @classmethod
    def _state_keys(cls) -> Collection:
        return (super()._state_keys() | {"transform", "loader"}) - {"fn"}

    def _repr_pandas_(self) -> pd.Series:
        return "ImageCell(" + self.data.data.reset_index(drop=True) + ")"

    def is_equal(self, other: AbstractColumn) -> bool:
        return (
            (other.__class__ == self.__class__)
            and (self.loader == other.loader)
            and (self.transform == other.transform)
            and self.data.is_equal(other.data)
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
