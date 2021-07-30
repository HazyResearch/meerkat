from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Mapping, Sequence, Tuple, Union

import yaml

from meerkat.errors import ConsolidationError

# an index into a block that specifies where a column's data lives in the block
BlockIndex = Union[int, slice, str]


if TYPE_CHECKING:
    from meerkat.block.ref import BlockRef


@dataclass
class BlockView:
    data: object
    block_index: BlockIndex
    block: AbstractBlock


class AbstractBlock:
    def __init__(self, *args, **kwargs):
        super(AbstractBlock, self).__init__(*args, **kwargs)

    def __getitem__(self, index: BlockIndex) -> BlockView:
        return BlockView(data=self._get_data(index), block_index=index, block=self)

    def _get_data(self, index: BlockIndex) -> object:
        """Must return view of the underlying data."""
        raise NotImplementedError()

    @property
    def signature(self) -> Hashable:
        raise NotImplementedError

    @classmethod
    def from_data(
        cls, data: object, names: Sequence[str]
    ) -> Tuple[AbstractBlock, Mapping[str, BlockIndex]]:
        raise NotImplementedError()

    @classmethod
    def consolidate(
        cls, block_refs: Sequence[BlockRef]
    ) -> Tuple[AbstractBlock, Mapping[str, BlockIndex]]:
        if len(block_refs) == 0:
            raise ConsolidationError("Must pass at least 1 BlockRef to consolidate.")

        if len({ref.block.signature for ref in block_refs}) != 1:
            raise ConsolidationError(
                "Can only consolidate blocks with matching signatures."
            )

        return cls._consolidate(block_refs=block_refs)

    @classmethod
    def _consolidate(cls, block_refs: Sequence[BlockRef]) -> BlockRef:
        raise NotImplementedError

    def _get(self, index, block_ref: BlockRef) -> Union[BlockRef, dict]:
        raise NotImplementedError

    def write(self, path: str, *args, **kwargs):
        os.makedirs(path, exist_ok=True)
        self._write_data(path, *args, **kwargs)
        metadata = {"klass": type(self)}
        metadata_path = os.path.join(path, "meta.yaml")
        yaml.dump(metadata, open(metadata_path, "w"))

    @classmethod
    def read(cls, path: str, *args, **kwargs):
        assert os.path.exists(path), f"`path` {path} does not exist."
        metadata_path = os.path.join(path, "meta.yaml")
        metadata = dict(yaml.load(open(metadata_path), Loader=yaml.FullLoader))

        block_class = metadata["klass"]
        data = block_class._read_data(path, *args, **kwargs)
        return block_class(data)
