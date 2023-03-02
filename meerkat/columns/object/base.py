from __future__ import annotations

import abc
import logging
from typing import Sequence

import cytoolz as tz
import pandas as pd
from yaml.representer import Representer

from meerkat.columns.abstract import Column
from meerkat.mixins.cloneable import CloneableMixin

Representer.add_representer(abc.ABCMeta, Representer.represent_name)


logger = logging.getLogger(__name__)


class ObjectColumn(Column):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if data is not None:
            data = list(data)
        super(ObjectColumn, self).__init__(data=data, *args, **kwargs)

    @classmethod
    def from_list(cls, data: Sequence):
        return cls(data=data)

    def batch(
        self,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        collate: bool = True,
        *args,
        **kwargs,
    ):
        for i in range(0, len(self), batch_size):
            if drop_last_batch and i + batch_size > len(self):
                continue
            if collate:
                yield self.collate(self[i : i + batch_size])
            else:
                yield self[i : i + batch_size]

    @classmethod
    def concat(cls, columns: Sequence[ObjectColumn]):
        data = list(tz.concat([c.data for c in columns]))
        if issubclass(cls, CloneableMixin):
            return columns[0]._clone(data=data)
        return cls.from_list(data)

    def is_equal(self, other: Column) -> bool:
        return (self.__class__ == other.__class__) and self.data == other.data

    def _repr_cell(self, index) -> object:
        return self[index]

    def _get_default_formatter(self):
        from PIL.Image import Image

        from meerkat.interactive.app.src.lib.component.core.image import ImageFormatter
        from meerkat.interactive.app.src.lib.component.core.scalar import (
            ScalarFormatter,
        )

        sample = self[0]
        if isinstance(sample, Image):
            return ImageFormatter()

        return ScalarFormatter()

    def to_pandas(self, allow_objects: bool = False) -> pd.Series:
        return pd.Series([self[int(idx)] for idx in range(len(self))])
