from __future__ import annotations

import abc
import functools
import logging
import os
from typing import Callable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from yaml.representer import Representer

from meerkat.block.abstract import BlockView
from meerkat.block.tensor_block import TensorBlock
from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.writers.concat_writer import ConcatWriter
from meerkat.writers.numpy_writer import NumpyMemmapWriter

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

Columnable = Union[Sequence, np.ndarray, pd.Series, torch.Tensor]

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            if out.ndim == 0:
                return out.clone().detach()
            return TensorColumn(out)
        else:
            return out

    return wrapper


class TensorColumn(
    np.lib.mixins.NDArrayOperatorsMixin,
    AbstractColumn,
):
    block_class: type = TensorBlock

    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, TensorBlock):
                raise ValueError(
                    "Cannot create `TensorColumn` from a `BlockView` not "
                    "referencing a `TensorBlock`."
                )
        elif data is not None and not isinstance(data, TensorColumn):
            if (
                isinstance(data, Sequence)
                and len(data) > 0
                and torch.is_tensor(data[0])
            ):
                # np.asarray supports a list of numpy arrays (it simply stacks them
                # before putting them into an array) but torch.as_tensor does not.
                # we want to support this for consistency and because it is important
                # for map
                data = torch.stack(data)
            data = torch.as_tensor(data)
        super(TensorColumn, self).__init__(data=data, *args, **kwargs)

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
    def concat(cls, columns: Sequence[TensorColumn]):
        data = torch.cat([c.data for c in columns])
        if issubclass(cls, CloneableMixin):
            return columns[0]._clone(data=data)
        return cls(data)

    @classmethod
    def get_writer(cls, mmap: bool = False, template: AbstractColumn = None):
        if mmap:
            return NumpyMemmapWriter()
        else:
            return ConcatWriter(template=template, output_type=TensorColumn)

    def _repr_pandas_(self) -> pd.Series:
        if len(self.shape) > 1:
            return pd.Series([f"torch.Tensor(shape={self.shape[1:]})"] * len(self))
        else:
            return pd.Series(self.data)

    @classmethod
    def from_data(cls, data: Union[Columnable, AbstractColumn]):
        """Convert data to an EmbeddingColumn."""
        if torch.is_tensor(data):
            return cls(data)
        else:
            return super(TensorColumn, cls).from_data(data)

    def _copy_data(self) -> torch.Tensor:
        return self._data.clone()

    def _view_data(self) -> object:
        return self._data

    def _write_data(self, path: str) -> None:
        # Saving all cell data in a single pickle file
        torch.save(self.data, os.path.join(path, "data.pt"))

    @staticmethod
    def _read_data(path: str) -> torch.Tensor:
        return torch.load(os.path.join(path, "data.pt"))

    def is_equal(self, other: AbstractColumn) -> bool:
        return (other.__class__ == self.__class__) and (self.data == other.data).all()

    def to_tensor(self) -> torch.Tensor:
        return self.data
