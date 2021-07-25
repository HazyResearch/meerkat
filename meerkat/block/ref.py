from __future__ import annotations

from typing import Mapping, Sequence, Union

from meerkat.block.abstract import AbstractBlock
from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.blockable import BlockableMixin


class BlockRef(Mapping):
    def __init__(self, columns: Mapping[str, AbstractColumn], block: AbstractBlock):
        self.columns: Mapping[str, AbstractColumn] = columns
        self.block: AbstractBlock = block

    def __getitem__(self, index: str):
        return self.columns[index]

    def __delitem__(self, key):
        del self.columns[key]

    def __len__(self):
        return len(self.columns)

    def __contains__(self, value):
        return value in self.columns

    def __iter__(self):
        return iter(self.columns)
    
    @property
    def block_indices(self):
        return {name: col._block_index for name, col in self.columns.items()}

    def apply(self, method_name: str = "_get", *args, **kwargs):
        # apply method to the block
        block, block_indices = getattr(self.block, method_name)(
            *args, **kwargs, block_indices=self.block_indices
        )

        # create new columns
        columns = {}
        for name, col in self.columns.items():
            block_index = block_indices[name]
            # create a new col
            new_col = col._clone(data=block[block_index])
            new_col._block_index = block_index
            new_col._block = block
            columns[name] = new_col

        # create new BlockRef from the columns
        return BlockRef(columns, block)

    def update(self, block_ref: BlockRef):
        if id(block_ref.block) != id(self.block):
            raise ValueError(
                "Can only update BlockRef with another BlockRef pointing "
                "to the same block."
            )
        self._columns.update(block_ref._columns)
