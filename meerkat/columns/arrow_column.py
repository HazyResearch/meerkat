from __future__ import annotations

import os
from typing import Sequence, Set

import pyarrow as pa
import torch
from pyarrow.compute import equal

from meerkat.block.abstract import BlockView
from meerkat.block.arrow_block import ArrowBlock
from meerkat.columns.abstract import AbstractColumn
from meerkat.errors import ImmutableError


class ArrowArrayColumn(
    AbstractColumn,
):

    block_class: type = ArrowBlock

    def __init__(
        self,
        data: Sequence,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, ArrowBlock):
                raise ValueError(
                    "ArrowArrayColumn can only be initialized with ArrowBlock."
                )
        elif not isinstance(data, (pa.Array, pa.ChunkedArray)):
            data = pa.array(data)

        super(ArrowArrayColumn, self).__init__(data=data, *args, **kwargs)

    def _get(self, index, materialize: bool = True):
        index = ArrowBlock._convert_index(index)

        if isinstance(index, slice) or isinstance(index, int):
            data = self._data[index]
        elif index.dtype == bool:
            data = self._data.filter(pa.array(index))
        else:
            data = self._data.take(index)

        if self._is_batch_index(index):
            return self._clone(data=data)
        else:
            return data

    def _set(self, index, value):
        raise ImmutableError("ArrowArrayColumn is immutable.")

    def _repr_cell(self, index) -> object:
        return self.data[index]

    def is_equal(self, other: AbstractColumn) -> bool:
        if other.__class__ != self.__class__:
            return False
        return equal(self.data, other.data)

    @classmethod
    def _state_keys(cls) -> Set:
        return super()._state_keys()

    def _write_data(self, path):
        table = pa.Table.from_arrays([self.data], names=["0"])
        ArrowBlock._write_table(os.path.join(path, "data.arrow"), table)

    @staticmethod
    def _read_data(path, mmap=False):
        table = ArrowBlock._read_table(os.path.join(path, "data.arrow"), mmap=mmap)
        return table["0"]

    @classmethod
    def concat(cls, columns: Sequence[ArrowArrayColumn]):
        data = pa.concat_arrays([c.data for c in columns])
        return columns[0]._clone(data=data)

    def to_numpy(self):
        return self.data.to_numpy(zero_copy_only=False)

    def to_tensor(self):
        return torch.tensor(self.data.to_numpy())

    def to_pandas(self):
        return self.data.to_pandas()
