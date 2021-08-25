from typing import Sequence

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

    def __init__(formatter=NumpyArrayColumn):
        """
        Initialize a new ArrowArrayColumn.
        """

        pass

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
        elif not isinstance(data, pa.Array):
            data = pa.array(data)

        super(ArrowArrayColumn, self).__init__(data=data, *args, **kwargs)

        self._formatter = formatter

    def _get(self, index, materialize: bool = True):
        index = ArrowBlock._convert_index(index)
        data = self._data[index]
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
