from __future__ import annotations

import abc
import functools
import logging
import numbers
import os
import shutil
from mmap import mmap
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from yaml.representer import Representer

from meerkat.block.abstract import BlockView
from meerkat.block.numpy_block import NumpyBlock
from meerkat.columns.abstract import AbstractColumn
from meerkat.writers.concat_writer import ConcatWriter

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, np.ndarray):
            return NumpyArrayColumn(out)
        else:
            return out

    return wrapper


class NumpyArrayColumn(
    AbstractColumn,
    np.lib.mixins.NDArrayOperatorsMixin,
):

    block_class: type = NumpyBlock

    def __init__(
        self,
        data: Sequence,
        *args,
        **kwargs,
    ):
        if isinstance(data, BlockView):
            if not isinstance(data.block, NumpyBlock):
                raise ValueError(
                    "Cannot create `NumpyArrayColumn` from a `BlockView` not "
                    "referencing a `NumpyBlock`."
                )
        elif not isinstance(data, np.memmap):
            data = np.asarray(data)
        super(NumpyArrayColumn, self).__init__(data=data, *args, **kwargs)

    # TODO (sabri): need to support str here
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc: np.ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (NumpyArrayColumn,)) and not (
                # support for at index
                method == "at"
                and isinstance(x, list)
            ):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.data if isinstance(x, NumpyArrayColumn) else x for x in inputs)

        if out:
            kwargs["out"] = tuple(
                x.data if isinstance(x, NumpyArrayColumn) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self._clone(data=result)

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

    @classmethod
    def from_array(cls, data: np.ndarray, *args, **kwargs):
        return cls(data=data, *args, **kwargs)

    def _set_batch(self, indices, values):
        self._data[indices] = values

    def _get(self, index, materialize: bool = True):
        index = NumpyBlock._convert_index(index)
        data = self._data[index]
        if self._is_batch_index(index):
            # only create a numpy array column
            return self._clone(data=data)
        else:
            return data

    def _copy_data(self) -> object:
        return self._data.copy()

    def _view_data(self) -> object:
        return self._data

    @property
    def is_mmap(self):
        # important to check if .base is a python mmap object, since a view of a mmap
        # is also a memmap object, but should not be symlinked or copied
        return isinstance(self.data, np.memmap) and isinstance(self.data.base, mmap)

    def _write_data(self, path: str, link: bool = True) -> None:
        path = os.path.join(path, "data.npy")
        # important to check if .base is a python mmap object, since a view of a mmap
        # is also a memmap object, but should not be symlinked
        if self.is_mmap:
            if link:
                os.symlink(self.data.filename, path)
            else:
                shutil.copy(self.data.filename, path)
        else:
            np.save(path, self.data)

    @staticmethod
    def _read_data(path: str, mmap=False, *args, **kwargs) -> np.ndarray:
        data_path = os.path.join(path, "data.npy")

        if mmap:
            return np.load(data_path, mmap_mode="r")
        return np.load(data_path)

    @classmethod
    def concat(cls, columns: Sequence[NumpyArrayColumn]):
        data = np.concatenate([c.data for c in columns])
        return columns[0]._clone(data=data)

    def is_equal(self, other: AbstractColumn) -> bool:
        if other.__class__ != self.__class__:
            return False
        return np.array_equal(self.data, other.data, equal_nan=True)

    @classmethod
    def get_writer(cls, mmap: bool = False, template: AbstractColumn = None):
        if mmap:
            from meerkat.writers.numpy_writer import NumpyMemmapWriter

            return NumpyMemmapWriter()
        else:
            return ConcatWriter(template=template, output_type=NumpyArrayColumn)

    def _repr_cell(self, index) -> object:
        if len(self.shape) > 1:
            return f"np.ndarray(shape={self.shape[1:]})"
        else:
            return self[index]

    def to_tensor(self) -> torch.Tensor:
        """Use `column.to_tensor()` instead of `torch.tensor(column)`, which is
        very slow."""
        # TODO (Sabri): understand why `torch.tensor(column)` is so slow
        return torch.tensor(self.data)

    def to_pandas(self) -> pd.Series:
        if len(self.shape) == 1:
            return pd.Series(self.data)
        else:
            # can only create a 1-D series
            return super().to_pandas()

    @classmethod
    def from_npy(
        cls,
        path,
        mmap_mode=None,
        allow_pickle=False,
        fix_imports=True,
        encoding="ASCII",
    ):
        data = np.load(
            path,
            mmap_mode=mmap_mode,
            allow_pickle=allow_pickle,
            fix_imports=fix_imports,
            encoding=encoding,
        )
        return cls(data)
