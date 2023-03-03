from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Hashable, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from meerkat.block.ref import BlockRef
from meerkat.columns.abstract import Column
from meerkat.errors import ConsolidationError
from meerkat.tools.lazy_loader import LazyLoader

from .abstract import AbstractBlock, BlockIndex, BlockView

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch


class TorchBlock(AbstractBlock):
    @dataclass(eq=True, frozen=True)
    class Signature:
        device: "torch.device"
        dtype: "torch.dtype"
        nrows: int
        shape: Tuple[int]
        klass: type

    def __init__(self, data, *args, **kwargs):
        super(TorchBlock, self).__init__(*args, **kwargs)
        if len(data.shape) <= 1:
            raise ValueError(
                "Cannot create a `TensorBlock` from data with less than 2 axes."
            )
        self.data = data

    @property
    def signature(self) -> Hashable:
        return self.Signature(
            klass=TorchBlock,
            device=self.data.device,
            nrows=self.data.shape[0],
            shape=self.data.shape[2:],
            dtype=self.data.dtype,
        )

    def _get_data(self, index: BlockIndex) -> "torch.Tensor":
        return self.data[:, index]

    @classmethod
    def from_column_data(cls, data: "torch.Tensor") -> Tuple[TorchBlock, BlockView]:
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
            data = torch.unsqueeze(data, dim=1)
            block_index = 0
        elif data.shape[1] == 1:
            block_index = slice(0, 1)
        else:
            block_index = slice(0, data.shape[1])

        block = cls(data)
        return BlockView(block_index=block_index, block=block)

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

        block = cls(torch.cat(to_concat, dim=1))

        # create columns
        new_columns = {
            name: columns[name]._clone(data=block[block_index])
            for name, block_index in new_indices.items()
        }

        return BlockRef(block=block, columns=new_columns)

    @staticmethod
    def _convert_index(index):
        from meerkat.columns.tensor.torch import TorchTensorColumn

        if isinstance(index, TorchTensorColumn) and index.data.dtype == np.bool_:
            # needed to silence torch deprecation warning
            # DeprecationWarning: In future, it will be an error for 'np.bool_' scalars
            # to be interpreted as an index
            return torch.as_tensor(index.data)

        if isinstance(index, list):
            # convert to np.ndarray so that it will be converted to a torch tensor
            # following rules below
            index = np.array(index)

        if isinstance(index, pd.Series):
            # convert to np.ndarray so that it will be converted to a torch tensor
            index = index.values

        if isinstance(index, np.ndarray) and index.dtype == np.bool_:
            # needed to silence torch deprecation warning
            # DeprecationWarning: In future, it will be an error for 'np.bool_' scalars
            # to be interpreted as an index
            return torch.as_tensor(index)

        if isinstance(index, pd.Series):
            return torch.as_tensor(index.values)

        from meerkat.columns.scalar.pandas import PandasScalarColumn

        if (
            isinstance(index, PandasScalarColumn)
            and index.data.values.dtype == np.bool_
        ):
            # needed to silence torch deprecation warning
            # DeprecationWarning: In future, it will be an error for 'np.bool_' scalars
            # to be interpreted as an index
            return torch.as_tensor(index.data.values)

        if isinstance(index, TorchTensorColumn):
            # need to convert to numpy for boolean indexing
            return index.data
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

    def _write_data(self, path: str):
        torch.save(self.data, os.path.join(path, "data.pt"))

    @staticmethod
    def _read_data(
        path: str, mmap: bool = False, read_inputs: Dict[str, Column] = None
    ):
        return torch.load(os.path.join(path, "data.pt"))
