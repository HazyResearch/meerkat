from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from meerkat.block.ref import BlockRef
from meerkat.columns.abstract import Column
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.tools.lazy_loader import LazyLoader

from .abstract import AbstractBlock, BlockIndex, BlockView

torch = LazyLoader("torch")


class ArrowBlock(AbstractBlock):
    @dataclass(eq=True, frozen=True)
    class Signature:
        nrows: int
        klass: type
        # mmap: bool

    def __init__(self, data: pa.Table, *args, **kwargs):
        super(ArrowBlock, self).__init__(*args, **kwargs)
        self.data = data

    @property
    def signature(self) -> Hashable:
        return self.Signature(klass=ArrowBlock, nrows=len(self.data))

    def _get_data(self, index: BlockIndex) -> pa.Array:
        return self.data[index]

    @classmethod
    def from_column_data(cls, data: pa.Array) -> BlockView:
        data = pa.Table.from_pydict({"col": data})
        block = cls(data)
        return BlockView(block=block, block_index="col")

    @classmethod
    def from_block_data(cls, data: pa.Table) -> List[BlockView]:
        block = cls(data)
        return [
            BlockView(block=block, block_index=column) for column in data.column_names
        ]

    @classmethod
    def _consolidate(
        cls,
        block_refs: Sequence[BlockRef],
        consolidated_inputs: Dict[int, "Column"] = None,
    ) -> BlockRef:
        table = pa.Table.from_pydict(
            # need to ignore index when concatenating
            {
                name: ref.block.data[col._block_index]
                for ref in block_refs
                for name, col in ref.items()
            }
        )
        block = cls(table)

        # pull out the block columns from all the block_refs
        columns = {}
        for ref in block_refs:
            columns.update(ref)

        new_columns = {
            name: col._clone(data=block[name]) for name, col in columns.items()
        }

        return BlockRef(block=block, columns=new_columns)

    @staticmethod
    def _convert_index(index):
        if isinstance(index, list):
            return np.array(index)
        if torch.is_tensor(index):
            # need to convert to numpy for boolean indexing
            return index.numpy()
        if isinstance(index, NumPyTensorColumn):
            return index.data
        if isinstance(index, TorchTensorColumn):
            # need to convert to numpy for boolean indexing
            return index.data.numpy()
        if isinstance(index, pd.Series):
            # need to convert to numpy for boolean indexing
            return index.values

        from meerkat.columns.scalar.pandas import PandasScalarColumn

        if isinstance(index, PandasScalarColumn):
            return index.data.values

        from meerkat.columns.scalar.arrow import ArrowScalarColumn

        if isinstance(index, ArrowScalarColumn):
            return index.to_numpy()

        return index

    def _get(
        self, index, block_ref: BlockRef, materialize: bool = True
    ) -> Union[BlockRef, dict]:
        index = self._convert_index(index)
        # TODO: check if they're trying to index more than just the row dimension

        if isinstance(index, int):
            # if indexing a single row, we do not return a block manager, just a dict
            # Convert to Python object for consistency with other ScalarColumn
            # implementations.
            return {
                name: self.data[col._block_index][index].as_py()
                for name, col in block_ref.columns.items()
            }

        if isinstance(index, slice):
            data = self.data[index]
        elif index.dtype == bool:
            data = self.data.filter(pa.array(index))
        else:
            # we do not want to use ``data = self.data.take(index)``
            # because it can't handle ChunkedArrays that don't fit in an Array
            # https://issues.apache.org/jira/browse/ARROW-9773
            # TODO (Sabri): Huggingface gets around this in a similar manner but
            # applies the slices to the record batches, because this allows them to do
            # the batch lookup in numpy, which is faster than pure python, which is
            # presumably why Table.slice does
            # noqa E501, https://github.com/huggingface/datasets/blob/491dad8507792f6f51077867e22412af7cd5c2f1/src/datasets/table.py#L110
            data = pa.concat_tables(self.data.slice(i, 1) for i in index)

        block = self.__class__(data)

        columns = {
            name: col._clone(data=block[col._block_index])
            for name, col in block_ref.columns.items()
        }
        # note that the new block may share memory with the old block
        return BlockRef(block=block, columns=columns)

    @staticmethod
    def _write_table(path: str, table: pa.Table):
        # noqa E501, source: huggingface implementation https://github.com/huggingface/datasets/blob/92304b42cf0cc6edafc97832c07de767b81306a6/src/datasets/table.py#L50
        with open(path, "wb") as sink:
            writer = pa.RecordBatchStreamWriter(sink=sink, schema=table.schema)
            batches: List[pa.RecordBatch] = table.to_batches()
            for batch in batches:
                writer.write_batch(batch)
            writer.close()
            return sum(batch.nbytes for batch in batches)

    @staticmethod
    def _read_table(path: str, mmap: bool = False):
        if mmap:
            return pa.ipc.open_stream(pa.memory_map(path)).read_all()
        else:
            return pa.ipc.open_stream(pa.input_stream(path)).read_all()

    def _write_data(self, path: str):
        self._write_table(os.path.join(path, "data.arrow"), self.data)

    @staticmethod
    def _read_data(
        path: str, mmap: bool = False, read_inputs: Dict[str, Column] = None
    ):
        return ArrowBlock._read_table(os.path.join(path, "data.arrow"), mmap=mmap)
