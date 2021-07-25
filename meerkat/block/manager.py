from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from meerkat.mixins.blockable import BlockableMixin
from meerkat.block.numpy_block import NumpyBlock
from typing import Dict, Mapping, Sequence, Union

from meerkat.columns.abstract import AbstractColumn
from .ref import BlockRef
import numpy as np


class BlockManager(MutableMapping):
    """

    This manager manages all blocks.
    """

    def __init__(self) -> None:
        self._columns: Dict[str, AbstractColumn] = {}
        self._column_to_block_id: Dict[str, int] = {}
        self._block_refs: Dict[int, BlockRef] = {}

    def add(self, block_ref: BlockRef):
        """
        Loop through all block instances, and check for a match.

        If match, then insert into that block.
        If not match, create a new block.

        Args:
            data (): a single blockable object, potentially contains multiple columns
        """
        self._columns.update(
            {name: column for name, column in block_ref.items()}
        )

        block_id = id(block_ref.block)
        # check if there already is a block_ref in the manager for this block
        if block_id in self._block_refs:
            self._block_refs[block_id].update(block_ref)
        else:
            self._block_refs[block_id] = block_ref

        self._column_to_block_id.update({name: block_id for name in block_ref.keys()})

    def apply(self, method_name: str = "_get", *args, **kwargs) -> BlockManager:
        """[summary]

        Args:
            fn (str): a function that is applied to a block and column_spec and
                returns a new block and column_spec.
        Returns:
            [type]: [description]
        """
        mgr = BlockManager()
        for block_ref in self._block_refs.values():
            new_block_ref = block_ref.apply(method_name=method_name, *args, **kwargs)
            mgr.add(new_block_ref)

        # apply method to columns not stored in block
        for name, col in self._columns.items():
            if name not in mgr:
                mgr[name] = getattr(col, method_name)(*args, **kwargs)
        return mgr

    def consolidate(self):
        block_ref_groups = defaultdict(list)
        for block_ref in self._block_refs.values():
            block_ref_groups[block_ref.block.signature].append(block_ref)

        for block_refs in block_ref_groups.values():
            if len(block_refs) == 0:
                continue 

            # remove old block_refs 
            for old_ref in block_refs:
                del self._block_refs[id(old_ref.block)]

            # consolidate group 
            block_class = block_refs[0].block.__class__
            block_indices = [ref.block_indices for ref in block_refs]
            blocks = [ref.block for ref in block_refs]
            block, block_indices = block_class.consolidate(blocks, block_indices)
            
            # create columns 
            columns = {}
            for name, block_index in block_indices.items():
                col = self._columns[name]._clone(data=block[block_index])
                col._block = block 
                col._block_index = block_index
                columns[name] = col
            self.add(BlockRef(columns=columns, block=block))

    def remove(self, name):
        if name not in self._columns:
            raise ValueError(f"Remove failed: no column '{name}' in BlockManager.")

        del self._columns[name]

        column_spec = self._column_specs[name]
        column_spec.remove(name)
        if len(column_spec) == 0:
            del column_spec
    
    @property
    def block_indices(self) -> Mapping[str, BlockIndex]:
        return {name: col.index for name, col in self._columns}

    def __getitem__(self, index: Union[str, Sequence[str]]):
        if isinstance(index, str):
            return self._columns[index]
        elif isinstance(index, Sequence):
            pass

    def __setitem__(self, index, data: Union[str, Sequence[str]]):
        self.add(column_spec=BlockRef.infer(data, index))

    def __delitem__(self, key):
        self.remove(key)

    def __len__(self):
        return len(self._columns)

    def __contains__(self, value):
        return value in self._columns

    def __iter__(self):
        return iter(self._columns)

    def add_column(self, col: AbstractColumn, name: str):
        """Convert data to a meerkat column using the appropriate Column
        type."""
        if not isinstance(col, BlockableMixin):
            col = col.view()
            self._columns[name] = col

        else:
            if col._block is None:
                block, block_index = col.block_class.from_data(col.data)
                # TODO: need to be able to set block on clone 
                col._clone(data=block[block_index])
                col._block = block 
                col._block_index = block_index
            else:
                # TODO: this should be a view and the _block should get carried
                col = col#.view()
            self.add(BlockRef(columns={name: col}, block=col._block))

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
