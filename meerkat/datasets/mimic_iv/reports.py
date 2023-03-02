from __future__ import annotations

import logging
from typing import Collection, Sequence

from meerkat.columns.deferred.base import DeferredColumn
from meerkat.columns.pandas_column import ScalarColumn

logger = logging.getLogger(__name__)


class ReportColumn(DeferredColumn):
    def __init__(
        self,
        data: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        *args,
        **kwargs,
    ):
        super(ReportColumn, self).__init__(
            ScalarColumn.from_data(data), *args, **kwargs
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
    def default_loader(cls, filepath):
        with open(filepath) as f:
            return f.read()

    @classmethod
    def _state_keys(cls) -> Collection:
        return (super()._state_keys() | {"transform", "loader"}) - {"fn"}
