from __future__ import annotations

import abc
import logging
from typing import Sequence

import pandas as pd
from yaml.representer import Representer

from mosaic.columns.abstract import AbstractColumn

Representer.add_representer(abc.ABCMeta, Representer.represent_name)


logger = logging.getLogger(__name__)


# Q. how to handle collate and materialize here? Always materialized but only sometimes
# may want to collate (because collate=True should return a batch-style object, while
# collate=False should return a Column style object).


class ListColumn(AbstractColumn):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        super(ListColumn, self).__init__(data=data, *args, **kwargs)

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

    def _repr_pandas_(self) -> pd.Series:
        return pd.Series(map(repr, self))

    def __setitem__(self, index, value):
        if self.visible_rows is not None:
            # TODO (sabri): this is a stop-gap solution but won't work for fancy numpy
            # indexes, should find a way to cobine index and visible rows into one index
            index = super()._remap_index(index)
        return self._data.__setitem__(index, value)
