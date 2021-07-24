from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Tuple, Union

from torch._C import Block

from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.blockable import BlockableMixin

from .abstract import AbstractBlock
from .ref import BlockRef


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
            {name: column_spec.column for name, column_spec in block_ref.items()}
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
        for block_ref in self._block_refs().values():
            new_block_ref = getattr(block_ref.block, method_name)(
                *args,
                **kwargs,
            )
            mgr.add(new_block_ref)

        # apply method to columns not stored in block
        for name, col in self._columns():
            if name not in mgr:
                mgr[name] = getattr(col, method_name)(*args, **kwargs)
        return mgr

    def consolidate(self):
        block_ref_groups = defaultdict(list)
        for block_ref in self._block_refs:
            block_ref_groups[block_ref.block.signature] = block_ref

        self._blocks: Dict[int, BlockRef] = {}
        for block_refs in block_ref_groups.values():
            block_class = block_refs[0].block.__class__
            block_ref = block_class.consolidate(block_refs)
            self.add(block_ref)

    def remove(self, name):
        if name not in self._columns:
            raise ValueError(f"Remove failed: no column '{name}' in BlockManager.")

        del self._columns[name]

        column_spec = self._column_specs[name]
        column_spec.remove(name)
        if len(column_spec) == 0:
            del column_spec

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

    @classmethod
    def from_dict(cls, data: Mapping[str, Union[AbstractColumn, object]]):
        manager = cls()
        for index, data in data.items():
            manager.insert(data, index)
        return manager
