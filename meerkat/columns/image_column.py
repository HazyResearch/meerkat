from __future__ import annotations

import logging
from typing import Collection, Sequence

from meerkat.columns.abstract import AbstractColumn
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

    def __repr__(self):
        transform = getattr(self.transform, "__qualname__", repr(self.transform))
        dirs = self.data.split("/")
        short_path = ("" if len(dirs) <= 2 else ".../") + "/".join(dirs[-2:])
        return f"ImageCell({short_path}, transform={transform})"


class ImageColumn(LambdaColumn):
    def __init__(
        self,
        data: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        *args,
        **kwargs,
    ):
        if not isinstance(data, PandasSeriesColumn):
            data = PandasSeriesColumn(data)
        super(ImageColumn, self).__init__(data, *args, **kwargs)
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

    def is_equal(self, other: AbstractColumn) -> bool:
        return (
            (other.__class__ == self.__class__)
            and (self.loader == other.loader)
            and (self.transform == other.transform)
            and self.data.is_equal(other.data)
        )
