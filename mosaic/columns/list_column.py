from __future__ import annotations

import abc
import logging
from typing import Sequence

import numpy as np
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

    def _get_batch(self, indices: np.ndarray, materialize: bool = True) -> ListColumn:
        """Get a batch of cells from the column.

        Args:
            index (int): This is an index into the ALL rows, not just visible rows. In
                other words, we assume that the index passed in has already been
                remapped via `_remap_index`, if `self.visible_rows` is not `None`.
            materialize (bool, optional): Materialize and return the object. This
                argument is used by subclasses of `AbstractColumn` that hold data in an
                unmaterialized format. Defaults to False.
        """
        if materialize:
            return self.collate([self._get_cell(int(i)) for i in indices])

        else:
            new_column = self.copy()
            new_column._visible_rows = indices
            return new_column

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
