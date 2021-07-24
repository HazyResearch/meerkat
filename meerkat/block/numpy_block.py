from __future__ import annotations

from typing import Hashable, List, Mapping, Sequence, Tuple, Union

import numpy as np

from meerkat.block.ref import BlockRef

Index = Union[int, slice, np.ndarray, str]


class NumpyBlock:
    def __init__(self, data, *args, **kwargs):
        super(NumpyBlock, self).__init__(*args, **kwargs)
        self.data

    def view(self, _block_idx: Index):
        return self.data[_block_idx]

    @property
    def signature(self) -> Hashable:
        return ("NumpyBlock", self.data.shape[2:], self.data.dtype)

    @classmethod
    def from_column(cls, data: np.ndarray) -> Tuple[NumpyBlock, Index]:
        data = np.expand_dims(data, axis=0)
        return cls(data), 0

    @classmethod
    def from_data(
        cls, data: np.ndarray, names: Sequence[str]
    ) -> Tuple[NumpyBlock, Mapping[str, Index]]:
        if data.shape[0] != len(names):
            raise ValueError("Cannot create a `NumpyBlock` from data of shape")

        return cls(data), {name: idx for idx, name in names}

    @staticmethod
    def consolidate(blocks: Sequence[BlockRef]) -> BlockRef:
        pass

    def _get(self, index, block_ref: BlockRef):
        data = self.data[:, index]
        return self.__class__(data), {
            name: col.block_index for name, col in self.columns.values()
        }
