from __future__ import annotations

import abc
import functools
import logging
import numbers
import os
from typing import Callable, Sequence

import dill
import numpy as np
import pandas as pd
import torch
import yaml
from yaml.representer import Representer

from mosaic.columns.abstract import AbstractColumn
from mosaic.writers.numpy_writer import NumpyMemmapWriter, NumpyWriter

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
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if data is not None:
            data = np.asarray(data)
        super(NumpyArrayColumn, self).__init__(data=data, *args, **kwargs)

    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (NumpyArrayColumn,)):
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
            return type(self)(result)

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

    def _get_batch(self, indices, materialize: bool = True):
        return self._data[indices]

    def _set_batch(self, indices, values):
        self._data[indices] = values

    @staticmethod
    def concat(columns: Sequence[NumpyArrayColumn]):
        return NumpyArrayColumn.from_array(np.concatenate([c.data for c in columns]))

    @classmethod
    def get_writer(cls, mmap: bool = False):
        if mmap:
            return NumpyMemmapWriter()
        else:
            return NumpyWriter()

    @classmethod
    def read(
        cls, path: str, mmap=False, dtype=None, shape=None, *args, **kwargs
    ) -> NumpyArrayColumn:
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Load in the metadata: only available if the array was stored by mosaic
        metadata_path = os.path.join(path, "meta.yaml")
        if os.path.exists(metadata_path):
            metadata = dict(yaml.load(open(metadata_path), Loader=yaml.FullLoader))
            assert metadata["dtype"] == cls

        # If the path doesn't exist, assume that `path` points to the `.npy` file
        data_path = os.path.join(path, "data.npy")
        if not os.path.exists(data_path):
            data_path = path

        # Load in the data
        if mmap:
            # assert dtype is not None and shape is not None
            # data = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
            data = np.load(data_path, mmap_mode="r")
        else:
            data = np.load(data_path, allow_pickle=True)

        col = cls(data)

        state_path = os.path.join(path, "state.dill")
        if os.path.exists(state_path):
            state = dill.load(open(state_path, "rb"))
            col.__dict__.update(state)
        return col

    def write(self, path: str, **kwargs) -> None:
        # Make all the directories to the path
        os.makedirs(path, exist_ok=True)

        # Get the column state
        state = self.get_state()
        _data = state["_data"]

        # Remove the data key and put the rest of `state` into a metadata dict
        del state["_data"]
        metadata = {
            "dtype": type(self),
            "len": len(self),
            **self.metadata,
        }

        # Get the paths where metadata and data should be stored
        metadata_path = os.path.join(path, "meta.yaml")
        state_path = os.path.join(path, "state.dill")
        data_path = os.path.join(path, "data.npy")

        # Saving all cell data in a single pickle file
        np.save(data_path, _data)

        # Saving the metadata as a yaml
        yaml.dump(metadata, open(metadata_path, "w"))
        dill.dump(state, open(state_path, "wb"))

    def _repr_pandas_(self) -> pd.Series:
        if len(self.shape) > 1:
            return pd.Series([f"np.ndarray(shape={self.shape[1:]})"] * len(self))
        else:
            return pd.Series(self.data)

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
