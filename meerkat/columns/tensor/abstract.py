from typing import List, Union

import numpy as np
import torch

from ..abstract import Column

TensorColumnTypes = Union[np.ndarray, torch.TensorType]


class TensorColumn(Column):
    def __new__(cls, data: TensorColumnTypes = None):
        
        if (cls is not TensorColumn) or (data is None):
            return super().__new__(cls)

        if isinstance(data, np.ndarray):
            from .numpy import NumPyTensorColumn

            return super().__new__(NumPyTensorColumn)
        elif torch.is_tensor(data):
            from .torch import TorchTensorColumn

            return super().__new__(TorchTensorColumn)
        elif isinstance(data, List):
            if len(data) == 0:
                raise ValueError(
                    "Cannot create `TensorColumn` from empty list of tensors."
                )
            elif torch.is_tensor(data[0]):
                from .torch import TorchTensorColumn

                return super().__new__(TorchTensorColumn)
            else:
                from .numpy import NumPyTensorColumn

                return super().__new__(NumPyTensorColumn)

        else:
            raise ValueError(
                f"Cannot create `TensorColumn` from object of type {type(data)}."
            )
