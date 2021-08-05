from __future__ import annotations

import logging
from typing import Collection, Sequence

import pandas as pd

from meerkat.cells.imagepath import ImagePath
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.cell_column import CellColumn
from meerkat.columns.lambda_column import LambdaColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.tools.lazy_loader import LazyLoader

folder = LazyLoader("torchvision.datasets.folder")

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
        super(ImageColumn, self).__init__(
            PandasSeriesColumn.from_data(data), *args, **kwargs
        )
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform

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
        if other.__class__ != self.__class__:
            return False
        if self.loader != other.loader:
            return False
        if self.transform != other.transform:
            return False
        return self.data.is_equal(other.data)


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
