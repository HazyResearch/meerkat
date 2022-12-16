from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import torch

from meerkat.block.abstract import BlockView
from meerkat.block.arrow_block import ArrowBlock
from meerkat.block.pandas_block import PandasBlock

from ..abstract import Column

ScalarColumnTypes = Union[np.ndarray, torch.TensorType, pd.Series, List]


class ScalarColumn(Column):
    def __new__(cls, data: ScalarColumnTypes = None, backend: str = None):
        from .arrow import ArrowScalarColumn
        from .pandas import PandasScalarColumn

        if (cls is not ScalarColumn) or (data is None):
            return super().__new__(cls)

        backends = {"arrow": ArrowScalarColumn, "pandas": PandasScalarColumn}
        if backend is not None:
            if backend not in backends:
                raise ValueError(
                    f"Cannot create `ScalarColumn` with backend '{backend}'. "
                    f"Expected one of {list(backends.keys())}"
                )
            else:
                return super().__new__(backends[backend])

        if isinstance(data, BlockView):
            if isinstance(data.block, PandasBlock):
                return super().__new__(PandasScalarColumn)
            elif isinstance(data.block, ArrowBlock):
                return super().__new__(ArrowScalarColumn)

        if isinstance(data, (np.ndarray, torch.TensorType, pd.Series, List, Tuple)):
            return super().__new__(PandasScalarColumn)
        elif isinstance(data, pa.Array):
            return super().__new__(ArrowScalarColumn)
        else:
            raise ValueError(
                f"Cannot create `ScalarColumn` from object of type {type(data)}."
            )
