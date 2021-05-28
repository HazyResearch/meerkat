from __future__ import annotations

import abc
import functools
import logging
import os
from typing import Callable, Sequence

import dill
import numpy as np
import pandas as pd
import torch
import yaml
from yaml.representer import Representer

from mosaic.columns.abstract import AbstractColumn
from mosaic.writers.numpy_writer import NumpyMemmapWriter
from mosaic.writers.torch_writer import TorchWriter

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


def getattr_decorator(fn: Callable):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            if out.ndim == 0:
                return torch.tensor(out)
            return TensorColumn(out)
        else:
            return out

    return wrapper


class TensorColumn(
    np.lib.mixins.NDArrayOperatorsMixin,
    AbstractColumn,
):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if data is not None and not isinstance(data, TensorColumn):
            data = torch.as_tensor(data)
        super(TensorColumn, self).__init__(data=data, *args, **kwargs)

    @property
    def data(self):
        if self.visible_rows is not None:
            return self._data[self.visible_rows]
        else:
            return self._data

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        args = [a._t if hasattr(a, "_t") else a for a in args]
        ret = func(*args, **kwargs)
        if ret.ndim == 0:
            return torch.tensor(ret)
        return TensorColumn(ret)

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

    def __setitem__(self, index, value):
        if self.visible_rows is not None:
            # TODO (sabri): this is a stop-gap solution but won't work for fancy numpy
            # indexes, should find a way to cobine index and visible rows into one index
            index = super()._remap_index(index)
        return self._data.__setitem__(index, value)

    def _remap_index(self, index):
        # don't remap index since we take care of visibility with self.data
        return index

    def _get_cell(self, index: int, materialize: bool = True):
        return self.data[index]

    def _get_batch(self, indices, materialize: bool = True):
        return self.data[indices]

    @classmethod
    def get_writer(cls, mmap: bool = False):
        if mmap:
            return NumpyMemmapWriter()
        else:
            return TorchWriter()

    @classmethod
    def read(
        cls, path: str, mmap=False, dtype=None, shape=None, *args, **kwargs
    ) -> TensorColumn:
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
