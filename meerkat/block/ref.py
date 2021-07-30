from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence, Union

if TYPE_CHECKING:
    from meerkat.block.abstract import AbstractBlock
    from meerkat.columns.abstract import AbstractColumn


class BlockRef(Mapping):
    def __init__(self, columns: Mapping[str, AbstractColumn], block: AbstractBlock):
        self.columns: Mapping[str, AbstractColumn] = columns
        self.block: AbstractBlock = block

    def __getitem__(self, index: Union[str, Sequence[str]]):
        if isinstance(index, str):
            return self.columns[index]
        else:
            return self.__class__(
                columns={col: self.columns[col].view() for col in index},
                block=self.block,
            )

    def __delitem__(self, key):
        self.columns.pop(key)

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
        return getattr(self.block, method_name)(*args, **kwargs, block_ref=self)

    def update(self, block_ref: BlockRef):
        if id(block_ref.block) != id(self.block):
            raise ValueError(
                "Can only update BlockRef with another BlockRef pointing "
                "to the same block."
            )
        self.columns.update(block_ref.columns)
