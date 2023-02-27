from typing import TYPE_CHECKING, List, Union

import numpy as np

from meerkat.block.abstract import BlockView
from meerkat.block.numpy_block import NumPyBlock
from meerkat.block.torch_block import TorchBlock
from meerkat.tools.lazy_loader import LazyLoader

from ..abstract import Column

torch = LazyLoader("torch")

if TYPE_CHECKING:
    from torch import TensorType

    TensorColumnTypes = Union[np.ndarray, TensorType]


class TensorColumn(Column):
    def __new__(cls, data: "TensorColumnTypes" = None, backend: str = None):
        from .numpy import NumPyTensorColumn
        from .torch import TorchTensorColumn

        backends = {"torch": TorchTensorColumn, "numpy": NumPyTensorColumn}

        if backend is not None:
            if backend not in backends:
                raise ValueError(
                    f"Backend {backend} not supported. "
                    f"Expected one of {list(backends.keys())}"
                )
            else:
                return super().__new__(backends[backend])

        if isinstance(data, BlockView):
            if isinstance(data.block, TorchBlock):
                backend = TorchTensorColumn
            elif isinstance(data.block, NumPyBlock):
                backend = NumPyTensorColumn

        if (cls is not TensorColumn) or (data is None):
            return super().__new__(cls)

        if isinstance(data, BlockView):
            if isinstance(data.block, TorchBlock):
                return super().__new__(TorchTensorColumn)
            elif isinstance(data.block, NumPyBlock):
                return super().__new__(NumPyTensorColumn)

        if isinstance(data, np.ndarray):
            return super().__new__(NumPyTensorColumn)
        elif torch.is_tensor(data):
            return super().__new__(TorchTensorColumn)
        elif isinstance(data, List):
            if len(data) == 0:
                raise ValueError(
                    "Cannot create `TensorColumn` from empty list of tensors."
                )
            elif torch.is_tensor(data[0]):
                return super().__new__(TorchTensorColumn)
            else:
                return super().__new__(NumPyTensorColumn)

        else:
            raise ValueError(
                f"Cannot create `TensorColumn` from object of type {type(data)}."
            )

    # def _get_default_formatters(self):
    #     from meerkat.interactive.formatter import TensorFormatterGroup

    #     return TensorFormatterGroup()
