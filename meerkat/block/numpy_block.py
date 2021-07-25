from __future__ import annotations
from meerkat.errors import ConsolidationError

from typing import Hashable, Mapping, Sequence, Tuple
from dataclasses import dataclass

import numpy as np

from .abstract import AbstractBlock, BlockIndex


class NumpyBlock(AbstractBlock):
    @dataclass(eq=True, frozen=True)
    class Signature:
        dtype: np.dtype
        nrows: int
        shape: Tuple[int]
        klass: type

    def __init__(self, data, *args, **kwargs):
        super(NumpyBlock, self).__init__(*args, **kwargs)
        if len(data.shape) <= 1:
            raise ValueError(
                "Cannot create a `NumpyBlock` from data with less than 2 axes."
            )
        self.data = data

    @property
    def signature(self) -> Hashable:
        return self.Signature(
            klass=NumpyBlock,
            # we don't
            nrows=self.data.shape[0],
            shape=self.data.shape[2:],
            dtype=self.data.dtype,
        )
    
    def __getitem__(self, index: BlockIndex):
        return self.data[:, index]

    @classmethod
    def from_data(cls, data: np.ndarray) -> Tuple[NumpyBlock, Mapping[str, BlockIndex]]:
        """[summary]

        Args:
            data (np.ndarray): [description]
            names (Sequence[str]): [description]

        Raises:
            ValueError: [description]

        Returns:
            Tuple[NumpyBlock, Mapping[str, BlockIndex]]: [description]
        """
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)
            block_index = 0
        elif data.shape[1] == 1:
            block_index = 0
        else:
            block_index = slice(0, data.shape[1])

        return cls(data), block_index

    @classmethod
    def _consolidate(
        cls,
        blocks: Sequence[NumpyBlock],
        block_indices: Sequence[Mapping[str, BlockIndex]],
    ) -> Tuple[NumpyBlock, Mapping[str, BlockIndex]]:
        offset = 0
        new_indices = {}
        to_concat = []
        for block, curr_indices in zip(blocks, block_indices):
            for name, index in curr_indices.items():
                if isinstance(index, slice):
                    block_view = block.data[:, index]
                    new_indices[name] = slice(
                        # need to update slice offset and remove step
                        offset,
                        block_view.shape[1] + offset,
                        1,
                    )
                elif isinstance(index, int):
                    # keep block axis
                    block_view = block.data[:, index : index + 1]
                    new_indices[name] = offset
                to_concat.append(block_view)
                offset += block_view.shape[1]
        block = np.concatenate(to_concat, axis=1)
        return cls(block), new_indices

    def _get(self, index, block_indices: Mapping[str, BlockIndex]):
        data = self.data[index]
        # note that the new block may share memory with the old block
        return self.__class__(data), block_indices
