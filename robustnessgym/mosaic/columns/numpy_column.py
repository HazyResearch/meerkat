from __future__ import annotations

import abc
import logging
import os
from typing import Sequence

import pandas as pd
import numpy as np
import numpy.lib.mixins
import yaml
from yaml.representer import Representer

from robustnessgym.mosaic.columns.abstract import AbstractColumn
from robustnessgym.mosaic.mixins.collate import identity_collate
from robustnessgym.mosaic.writers.numpy_writer import NumpyMemmapWriter, NumpyWriter

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


class NumpyArrayColumn(
    AbstractColumn,
    np.ndarray,
    numpy.lib.mixins.NDArrayOperatorsMixin,
):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        super(NumpyArrayColumn, self).__init__(data=np.asarray(data), *args, **kwargs)

    def __array__(self, *args, **kwargs):
        return np.asarray(self._data)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # Convert the inputs to np.ndarray
        inputs = [
            input_.view(np.ndarray) if isinstance(input_, self.__class__) else input_
            for input_ in inputs
        ]

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, self.__class__):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs["out"] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        # Apply ufunc, method
        results = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = tuple(
            (
                np.asarray(result).view(self.__class__)
                if result.ndim > 0
                else np.asarray([result]).view(self.__class__)
                if output is None
                else output
            )
            for result, output in zip(results, outputs)
        )

        if results and isinstance(results[0], self.__class__):
            results[0]._data = np.asarray(results[0])
            results[0]._materialize = self._materialize
            results[0].collate = self.collate
            results[0].visible_rows = self.visible_rows

        return results[0] if len(results) == 1 else results

    def __new__(cls, data, *args, **kwargs):
        return np.asarray(data).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return

        self._data = getattr(obj, "_data", None)
        self._materialize = getattr(obj, "_materialize", True)
        self.collate = getattr(obj, "collate", identity_collate)
        self.visible_rows = getattr(obj, "visible_rows", None)

    @classmethod
    def from_array(cls, data: np.ndarray, *args, **kwargs):
        return cls(data=data, *args, **kwargs)

    def _get_batch(self, indices):
        return self.from_array(self.__array__()[indices])
    
    def get_writer(mmap: bool = False):
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
            assert dtype is not None and shape is not None
            data = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
        else:
            data = np.load(data_path)

        return cls(data)

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
            "state": state,
            **self.metadata,
        }

        # Get the paths where metadata and data should be stored
        metadata_path = os.path.join(path, "meta.yaml")
        data_path = os.path.join(path, "data.npy")

        # Saving all cell data in a single pickle file
        np.save(data_path, _data)

        # Saving the metadata as a yaml
        yaml.dump(metadata, open(metadata_path, "w"))
    
    def _repr_pandas_(self) -> pd.Series:
        if len(self.shape) > 1:
            return pd.Series([f"np.ndarray(shape={self.shape[1:]})"] * len(self))
        else:
            return pd.Series(self)
