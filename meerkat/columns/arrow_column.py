from typing import Sequence, Set

import pyarrow as pa

from meerkat.block.abstract import BlockView
from meerkat.block.arrow_block import ArrowBlock
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.columns.pandas_column import PandasSeriesColumn


class ArrowArrayColumn(
    AbstractColumn,
):

    block_class: type = ArrowBlock

    def __init__(
        self,
        data: Sequence,
        formatter=None,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, ArrowBlock):
                raise ValueError(
                    "ArrowArrayColumn can only be initialized with ArrowBlock."
                )
        elif not isinstance(data, (pa.Array, pa.ChunkedArray)):
            data = pa.array(data)

        super(ArrowArrayColumn, self).__init__(data=data, *args, **kwargs)

        self._formatter = formatter

    def _get(self, index, materialize: bool = True):
        index = ArrowBlock._convert_index(index)
        if isinstance(index, slice) or isinstance(index, int):
            data = self._data[index]
        else:
            data = self._data.take(index)
        if self._is_batch_index(index):
            if self._formatter is None:
                return self._clone(data=data)
            elif self._formatter == "numpy":
                return NumpyArrayColumn(data.to_numpy())
            elif self._formatter == "pandas":
                return PandasSeriesColumn(data.to_pandas())
            else:
                raise ValueError(f"Formatter '{self._formatter}' not supported.")
        else:
            return data

    def _set(self, index, value):
        raise NotImplementedError("ArrowArrayColumn is immutable.")

    def _repr_cell(self, index) -> object:
        return self.data[index]

    @classmethod
    def _state_keys(cls) -> Set:
        return super()._state_keys() | {"_formatter"}

    def _write_data(self, path):
        # with open(filename, "wb") as sink:
        #     writer = pa.RecordBatchStreamWriter(sink=sink, schema=table.schema)
        #     batches: List[pa.RecordBatch] = table.to_batches()
        #     for batch in batches:
        #         writer.write_batch(batch)
        #     writer.close()
        #     return sum(batch.nbytes for batch in batches)
        pass

    def _read_data(self, path, mmap=False):
        # if mmap:
        #     self._data = pa.ipc.open_stream(pa.memory_map(filename)).read_all()
        # else:
        #     self._data = pa.ipc.open_stream(pa.input_stream(filename)).read_all()
        pass
