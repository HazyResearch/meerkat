from typing import List, Union

import numpy as np
import torch

from ..abstract import Column

TensorColumnTypes = Union[np.ndarray, torch.TensorType]


class TensorColumn(Column):
    def __new__(cls, data: TensorColumnTypes):

        if isinstance(data, (np.ndarray, List)):
            from .numpy import NumPyTensorColumn

            return super().__new__(NumPyTensorColumn)
        elif torch.is_tensor(data):
            from .torch import TorchTensorColumn

            return super().__new__(TorchTensorColumn)

        else:
            raise ValueError(
                f"Cannot create `TensorColumn` from object of type {type(data)}."
            )
