from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from mmap import mmap
from typing import Dict, Hashable, Sequence, Tuple, Union

import numpy as np

from meerkat.block.ref import BlockRef
from meerkat.columns.abstract import Column
from meerkat.errors import ConsolidationError
from meerkat.tools.lazy_loader import LazyLoader

from .abstract import AbstractBlock, BlockIndex, BlockView

torch = LazyLoader("torch")


class NumPyBlock(AbstractBlock):
    @dataclass(eq=True, frozen=True)
    class Signature:
        dtype: np.dtype
        nrows: int
        shape: Tuple[int]
        klass: type
        mmap: Union[bool, int]

    def __init__(self, data, *args, **kwargs):
        super(NumPyBlock, self).__init__(*args, **kwargs)
        if len(data.shape) <= 1:
            raise ValueError(
                "Cannot create a `NumpyBlock` from data with less than 2 axes."
            )
        self.data = data

    @property
    def signature(self) -> Hashable:
        return self.Signature(
            klass=NumPyBlock,
            # don't want to consolidate any mmaped blocks
            mmap=id(self) if isinstance(self.data, np.memmap) else False,
            nrows=self.data.shape[0],
            shape=self.data.shape[2:],
            dtype=self.data.dtype,
        )

    def _get_data(self, index: BlockIndex, materialize: bool = True) -> np.ndarray:
        return self.data[:, index]

    @classmethod
    def from_column_data(cls, data: np.ndarray) -> Tuple[NumPyBlock, BlockView]:
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
            block_index = slice(0, 1)
        else:
            block_index = slice(0, data.shape[1])

        block = cls(data)
        return BlockView(block=block, block_index=block_index)

    @classmethod
    def _consolidate(
        cls,
        block_refs: Sequence[BlockRef],
        consolidated_inputs: Dict[int, "Column"] = None,
    ) -> BlockRef:
        offset = 0
        new_indices = {}
        columns = {}
        to_concat = []
        for block_ref in block_refs:
            for name, col in block_ref.items():
                # keep track of all the columns in the block_refs
                if name in columns:
                    raise ConsolidationError(
                        "Cannot consolidate two block refs containing the same column."
                    )
                columns[name] = col

                # add block and compute new indices
                block_index = col._block_index
                if isinstance(block_index, slice):
                    block_view = col._block.data[:, block_index]
                    new_indices[name] = slice(
                        # need to update slice offset and remove step
                        offset,
                        block_view.shape[1] + offset,
                        1,
                    )
                elif isinstance(block_index, int):
                    # keep block axis
                    block_view = col._block.data[:, block_index : block_index + 1]
                    new_indices[name] = offset
                to_concat.append(block_view)
                offset += block_view.shape[1]

        block = cls(np.concatenate(to_concat, axis=1))

        # create columns
        new_columns = {
            name: columns[name]._clone(data=block[block_index])
            for name, block_index in new_indices.items()
        }

        return BlockRef(block=block, columns=new_columns)

    @staticmethod
    def _convert_index(index):
        if torch.is_tensor(index):
            # need to convert to numpy for boolean indexing
            return index.numpy()
        return index

    def _get(
        self, index, block_ref: BlockRef, materialize: bool = True
    ) -> Union[BlockRef, dict]:
        index = self._convert_index(index)
        # TODO: check if they're trying to index more than just the row dimension
        data = self.data[index]
        if isinstance(index, int):
            # if indexing a single row, we do not return a block manager, just a dict
            return {
                name: data[col._block_index] for name, col in block_ref.columns.items()
            }
        block = self.__class__(data)
        columns = {
            name: col._clone(data=block[col._block_index])
            for name, col in block_ref.columns.items()
        }
        # note that the new block may share memory with the old block
        return BlockRef(block=block, columns=columns)

    @property
    def is_mmap(self):
        # important to check if .base is a python mmap object, since a view of a mmap
        # is also a memmap object, but should not be symlinked or copied
        return isinstance(self.data, np.memmap) and isinstance(self.data.base, mmap)

    def _write_data(self, path: str, link: bool = True):
        path = os.path.join(path, "data.npy")
        if self.is_mmap:
            if link:
                os.symlink(self.data.filename, path)
            else:
                shutil.copy(self.data.filename, path)
        else:
            np.save(path, self.data)

    @staticmethod
    def _read_data(
        path: str, mmap: bool = False, read_inputs: Dict[str, Column] = None
    ):
        data_path = os.path.join(path, "data.npy")

        if mmap:
            return np.load(data_path, mmap_mode="r")
        return np.load(data_path, allow_pickle=True)
