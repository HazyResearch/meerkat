from meerkat.columns.numpy_column import NumpyArrayColumn
import pyarrow as pa

from meerkat.columns.abstract import AbstractColumn
from meerkat.block.arrow_block import ArrowBlock

class ArrowArrayColumn(
    AbstractColumn,
):

    block_class: type = ArrowBlock

    def __init__(column_type=NumpyArrayColumn):
        pass 

