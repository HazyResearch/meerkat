from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Hashable, List, Mapping, Sequence, Tuple, Union

from meerkat.errors import ConsolidationError
from meerkat.tools.utils import dump_yaml, load_yaml

# an index into a block that specifies where a column's data lives in the block
BlockIndex = Union[int, slice, str]


if TYPE_CHECKING:
    from meerkat.block.ref import BlockRef
    from meerkat.columns.abstract import Column


@dataclass
class BlockView:
    block_index: BlockIndex
    block: AbstractBlock

    @property
    def data(self):
        return self.block._get_data(self.block_index)


class AbstractBlock:
    def __init__(self, *args, **kwargs):
        super(AbstractBlock, self).__init__(*args, **kwargs)

    def __getitem__(self, index: BlockIndex) -> BlockView:
        return BlockView(block_index=index, block=self)

    def _get_data(self, index: BlockIndex) -> object:
        """Must return view of the underlying data."""
        raise NotImplementedError()

    def subblock(self, indices: List[BlockIndex]):
        raise NotImplementedError

    @property
    def signature(self) -> Hashable:
        raise NotImplementedError

    @classmethod
    def from_column_data(cls, data: object) -> Tuple[AbstractBlock, BlockView]:
        raise NotImplementedError()

    @classmethod
    def from_block_data(cls, data: object) -> Tuple[AbstractBlock, BlockView]:
        raise NotImplementedError()

    @classmethod
    def consolidate(
        cls,
        block_refs: Sequence[BlockRef],
        consolidated_inputs: Dict[int, "Column"] = None,
    ) -> Tuple[AbstractBlock, Mapping[str, BlockIndex]]:
        if len(block_refs) == 0:
            raise ConsolidationError("Must pass at least 1 BlockRef to consolidate.")

        if len({ref.block.signature for ref in block_refs}) != 1:
            raise ConsolidationError(
                "Can only consolidate blocks with matching signatures."
            )
        return cls._consolidate(
            block_refs=block_refs, consolidated_inputs=consolidated_inputs
        )

    @classmethod
    def _consolidate(cls, block_refs: Sequence[BlockRef]) -> BlockRef:
        raise NotImplementedError

    def _get(self, index, block_ref: BlockRef) -> Union[BlockRef, dict]:
        raise NotImplementedError

    @property
    def is_mmap(self):
        return False

    def write(self, path: str, *args, **kwargs):
        os.makedirs(path, exist_ok=True)
        self._write_data(path, *args, **kwargs)
        metadata = {"klass": type(self)}
        metadata_path = os.path.join(path, "meta.yaml")
        dump_yaml(metadata, metadata_path)

    @classmethod
    def read(cls, path: str, *args, **kwargs):
        assert os.path.exists(path), f"`path` {path} does not exist."
        metadata_path = os.path.join(path, "meta.yaml")
        metadata = dict(load_yaml(metadata_path))

        block_class = metadata["klass"]
        data = block_class._read_data(path, *args, **kwargs)
        return block_class(data)

    def _write_data(self, path: str, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _read_data(path: str, *args, **kwargs) -> object:
        raise NotImplementedError
