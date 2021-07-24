from __future__ import annotations
from meerkat.errors import ConsolidationError

import numpy as np
from typing import Hashable, Mapping, Sequence, Tuple, Union

# an index into a block that specifies where a column's data lives in the block
BlockIndex = Union[int, slice, str]


class AbstractBlock:
    def __init__(self, *args, **kwargs):
        super(AbstractBlock, self).__init__(*args, **kwargs)

    @property
    def signature(self) -> Hashable:
        raise NotImplementedError

    @classmethod
    def from_data(
        cls, data: object, names: Sequence[str]
    ) -> Tuple[AbstractBlock, Mapping[str, BlockIndex]]:

        return

    @classmethod
    def consolidate(
        cls,
        blocks: Sequence[AbstractBlock],
        block_indices: Sequence[Mapping[str, BlockIndex]],
    ) -> Tuple[AbstractBlock, Mapping[str, BlockIndex]]:
        if len(blocks) != len(block_indices):
            raise ConsolidationError(
                "Number blocks and block_indices passed to consolidate must be equal."
            )

        if len(blocks) == 0:
            raise ConsolidationError("Must pass at least 1 block to consolidate.")
        
        if len({block.signature for block in blocks}) != 1:
            raise ConsolidationError(
                "Can only consolidate blocks with matching signatures."
            )
        
        return cls._consolidate(blocks=blocks, block_indices=block_indices)

    @classmethod
    def _consolidate(
        cls,
        blocks: Sequence[AbstractBlock],
        block_indices: Sequence[Mapping[str, BlockIndex]],
    ) -> Tuple[AbstractBlock, Mapping[str, BlockIndex]]:
        raise NotImplementedError

    def _get():
        raise NotImplementedError
