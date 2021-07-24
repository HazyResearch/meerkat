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
        return self.name_to_spec[index]

    def __delitem__(self, key):
        del self.name_to_spec[key]

    def __len__(self):
        return len(self.name_to_spec)

    def __contains__(self, value):
        return value in self.name_to_spec

    def __iter__(self):
        return iter(self.name_to_spec)
    
    @property
    def block_indices(self):
        return {name: col.index for name, col in self.columns.items()}

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
        self._columns.update(block_ref._update)

    @classmethod
    def infer(cls, data, names: Union[Sequence[str], str] = None):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        if names is str:
            cls._infer_single(data, name=names)
        else:
            cls._infer_multiple(data, names=names)

    @classmethod
    def _infer_multiple(cls, data, names: Sequence[str] = None):
        if isinstance(data, np.ndarray):
            from .numpy_column import NumpyArrayColumn

            return NumpyArrayColumn(data)

    @classmethod
    def _infer_single(cls, data, name: str = None) -> Union[AbstractColumn, BlockRef]:
        col = AbstractColumn.from_data(data=data)

        if isinstance(col, BlockableMixin):
            return cls(names=[name], block=data.block, columns=[data])
        else:
            return col
