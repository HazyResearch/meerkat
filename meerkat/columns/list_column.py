from __future__ import annotations

import abc
import logging
from typing import Sequence

import cytoolz as tz
import pandas as pd
from yaml.representer import Representer

from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.cloneable import CloneableMixin

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
        if data is not None:
            data = list(data)
        super(ListColumn, self).__init__(data=data, *args, **kwargs)

    @classmethod
    def from_list(cls, data: Sequence):
        return cls(data=data)

    @property
    def data(self):
        """Get the underlying data (excluding invisible rows).

        To access underlying data with invisible rows, use `_data`.
        """
        return self._data

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

    @classmethod
    def concat(cls, columns: Sequence[ListColumn]):
        data = list(tz.concat([c.data for c in columns]))
        if issubclass(cls, CloneableMixin):
            return columns[0]._clone(data=data)
        return cls.from_list(data)
