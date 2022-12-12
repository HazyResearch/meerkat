from typing import List, Union

import numpy as np
import torch
import pandas as pd

from ..abstract import Column

ScalarColumnTypes = Union[np.ndarray, torch.TensorType, pd.Series, List]


class ScalarColumn(Column):
    def __new__(cls, data: ScalarColumnTypes=None):
        if data is None:
            return super().__new__(cls)
            
        if isinstance(data, (np.ndarray, torch.TensorType, pd.Series, List)):
            from .pandas import PandasScalarColumn

            return super().__new__(PandasScalarColumn)
        else:
            raise ValueError(
                f"Cannot create `ScalarColumn` from object of type {type(data)}."
            )