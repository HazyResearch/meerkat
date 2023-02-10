from __future__ import annotations

import abc
import functools
import logging
import os
from typing import TYPE_CHECKING, Callable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from yaml.representer import Representer

from meerkat.block.abstract import BlockView
from meerkat.block.torch_block import TorchBlock
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.writers.concat_writer import ConcatWriter
from meerkat.writers.numpy_writer import NumpyMemmapWriter

from ..abstract import Column
from .abstract import TensorColumn

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

Columnable = Union[Sequence, np.ndarray, pd.Series, "torch.Tensor"]

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            if out.ndim == 0:
                return out.clone().detach()
            return TorchTensorColumn(out)
        else:
            return out

    return wrapper


def _as_tensor(data: Union["torch.Tensor", np.ndarray, pd.Series]) -> "torch.Tensor":
    """Overloaded as_tensor function to support other data types."""
    if not isinstance(data, (np.ndarray, torch.Tensor)):
        data = np.asarray(data)
    return torch.as_tensor(data)


class TorchTensorColumn(
    np.lib.mixins.NDArrayOperatorsMixin,
    TensorColumn,
):
    block_class: type = TorchBlock

    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, TorchBlock):
                raise ValueError(
                    "Cannot create `TensorColumn` from a `BlockView` not "
                    "referencing a `TensorBlock`."
                )
        elif data is not None and not isinstance(data, TorchTensorColumn):
            if isinstance(data, Sequence) and len(data) > 0:
                # TODO: We need to apply this check and do proper conversion of every
                # element in the sequence.
                # e.g. a list of mixed ndarrays and torch tensors
                # [np.array, torch.Tensor] should work.
                if torch.is_tensor(data[0]):
                    # np.asarray supports a list of numpy arrays (it simply stacks them
                    # before putting them into an array) but torch.as_tensor does not.
                    # we want to support this for consistency and because it is
                    # important for map
                    data = torch.stack(data)
                else:
                    data = np.asarray(data)
            data = _as_tensor(data)
        super(TorchTensorColumn, self).__init__(data=data, *args, **kwargs)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        def _process_arg(arg):
            if isinstance(arg, type(self)):
                return arg.data
            elif isinstance(arg, (List, Tuple)):
                # Specifically use list and tuple because these are
                # expected types for arguments in torch operations.
                return type(arg)([_process_arg(_a) for _a in arg])
            elif isinstance(arg, Mapping):
                # All mappings can be converted to dictionaries
                # when processed by torch operations.
                return {_k: _process_arg(_a) for _k, _a in arg.items()}
            else:
                return arg

        def _process_ret(ret):
            # This function may need to be refactored into an instance method
            # because the from_data implementation is different for each
            # class.
            if isinstance(ret, torch.Tensor):
                if ret.ndim == 0:
                    return ret.clone().detach()
                return self.from_data(ret)
            elif isinstance(ret, (List, Tuple)):
                return type(ret)([_process_arg(_a) for _a in ret])
            elif isinstance(ret, Mapping):
                return {_k: _process_arg(_a) for _k, _a in ret.items()}
            else:
                return ret

        if kwargs is None:
            kwargs = {}
        args = [_process_arg(a) for a in args]
        ret = func(*args, **kwargs)
        return _process_ret(ret)

    def __getattr__(self, name):
        try:
            out = getattr(object.__getattribute__(self, "data"), name)
            if isinstance(out, Callable):
                return getattr_decorator(out)
            else:
                return out
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def _get(self, index, materialize: bool = True):
        index = self.block_class._convert_index(index)

        data = self._data[index]
        if self._is_batch_index(index):
            # only create a numpy array column
            return self._clone(data=data)
        else:
            return data

    def _set_batch(self, indices, values):
        self._data[indices] = values

    @classmethod
    def concat(cls, columns: Sequence[TorchTensorColumn]):
        data = torch.cat([c.data for c in columns])
        if issubclass(cls, CloneableMixin):
            return columns[0]._clone(data=data)
        return cls(data)

    @classmethod
    def get_writer(cls, mmap: bool = False, template: Column = None):
        if mmap:
            return NumpyMemmapWriter()
        else:
            return ConcatWriter(template=template, output_type=TorchTensorColumn)

    def _repr_cell(self, index) -> object:
        if len(self.shape) > 1:
            if len(self.shape) == 2 and self.shape[1] < 5:
                return self[index]
            return f"torch.Tensor(shape={self.shape[1:]})"
        else:
            return self[index]

    @staticmethod
    def _get_default_formatter() -> Callable:
        from meerkat.interactive.app.src.lib.component.core.scalar import (
            ScalarFormatter,
        )

        return ScalarFormatter()

    @classmethod
    def from_data(cls, data: Union[Columnable, Column]):
        """Convert data to an EmbeddingColumn."""
        if torch.is_tensor(data):
            return cls(data)
        else:
            return super(TorchTensorColumn, cls).from_data(data)

    def _copy_data(self) -> "torch.Tensor":
        return self._data.clone()

    def _view_data(self) -> object:
        return self._data

    def _write_data(self, path: str) -> None:
        # Saving all cell data in a single pickle file
        torch.save(self.data, os.path.join(path, "data.pt"))

    @staticmethod
    def _read_data(path: str) -> "torch.Tensor":
        return torch.load(os.path.join(path, "data.pt"))

    def sort(
        self, ascending: Union[bool, List[bool]] = True, kind: str = "quicksort"
    ) -> TorchTensorColumn:
        """Return a sorted view of the column.

        Args:
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`. Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            Column: A view of the column with the sorted data.
        """
        # calls argsort() function to retrieve ordered indices

        sorted_index = self.argsort(ascending=ascending, kind=kind)
        return self[sorted_index]

    def argsort(
        self, ascending: Union[bool, List[bool]] = True, kind: str = "quicksort"
    ) -> TorchTensorColumn:
        """Return indices that would sorted the column.

        Args:
            ascending (Union[bool, List[bool]]): Whether to sort in ascending or
                descending order. If a list, must be the same length as `by`. Defaults
                to True.
            kind (str): The kind of sort to use. Defaults to 'quicksort'. Options
                include 'quicksort', 'mergesort', 'heapsort', 'stable'.
        Return:
            TensorColumn: A view of the column with the sorted data.

        For now! Raises error when shape of input array is more than one error.
        """
        try:
            self.size()[1]

        except IndexError:  # Case 1: The array only has one column
            # returns indices of descending order of array
            if not ascending:
                return torch.argsort(self.data, dim=-1, descending=True)
            # returns indices of ascending order of array
            return torch.argsort(self.data, dim=-1, descending=False)

        else:  # Case 2: The array has more than one column, raise error.
            raise Exception("No implementation for array with more than one column.")

    def is_equal(self, other: Column) -> bool:
        return (other.__class__ == self.__class__) and (self.data == other.data).all()

    def to_tensor(self) -> "torch.Tensor":
        return self.data

    def to_pandas(self, allow_objects: bool = True) -> pd.Series:
        if len(self.shape) == 1:
            return pd.Series(self.to_numpy())
        elif allow_objects:
            # can only create a 1-D series
            data = self.to_numpy()
            return pd.Series([data[int(idx)] for idx in range(len(self))])
        else:
            # can only create a 1-D series
            return super().to_pandas()

    def to_numpy(self) -> pd.Series:
        return self.data.detach().cpu().numpy()

    def to_arrow(self) -> pa.Array:
        if len(self.shape) == 1:
            return pa.array(self.to_numpy())
        else:
            return super().to_arrow()

    def mean(
        self, dim: int = None, keepdim: bool = False, *args, **kwargs
    ) -> "torch.Tensor":
        # torch only supports mean for floating point dtypes
        if self.data.dtype not in [
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
            torch.half,
            torch.bfloat16,
        ]:
            data = self.data.float()
        else:
            data = self.data
        if dim is not None:
            return data.mean(*args, dim=dim, keepdim=keepdim, **kwargs)
        else:
            return data.mean(*args, **kwargs).numpy().item()
