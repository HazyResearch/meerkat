from __future__ import annotations

import abc
import logging
import os
from typing import Sequence

import numpy as np
import numpy.lib.mixins
import yaml
from yaml.representer import Representer

from robustnessgym.mosaic.columns.abstract import AbstractColumn
from robustnessgym.mosaic.mixins.collate import identity_collate

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

    def __getitem__(self, index):
        if self.visible_rows is not None:
            # Remap the index if only some rows are visible
            index = self._remap_index(index)

        # indices that return a single cell
        if (
            isinstance(index, int)
            or isinstance(index, np.int)
            # np.ndarray indexed with a tuple of length 1 does not return an np.ndarray
            # but the element at the index
            # TODO: interestingly, np.ndarray indexed with a list of length 1 DOES
            # return a np.ndarray. Discuss how we want to handle this for columns in RG,
            # ideally all columns should share the same behavior w.r.t. this.
            or (isinstance(index, tuple) and len(index) == 1)
        ):
            return self._data[index]

        # indices that return batches
        if isinstance(index, slice):
            # int or slice index => standard list slicing
            data = self._data[index]
        elif isinstance(index, tuple) and len(index):
            data = self.__array__()[index]
        elif isinstance(index, list) and len(index):
            data = [self._data[i] for i in index]
        elif isinstance(index, np.ndarray) and len(index.shape) == 1:
            data = [self._data[int(i)] for i in index]
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

        # TODO(karan): do we need collate in NumpyArrayColumn
        # if self._materialize:
        #     # return a batch
        #     return self.collate([element for element in data])
        # else:
        # if not materializing, return a new NumpyArrayColumn
        # return self.from_list(data)

        # need to check if data has `ndim`, in case data is str or other object
        if hasattr(data, "ndim") and data.ndim > 0:
            return self.from_array(data)
        return self.from_array([data])

    def batch(
        self,
        batch_size: int = 32,
        drop_last_batch: bool = False,
        collate: bool = True,
        *args,
        **kwargs,
    ):
        for i in range(0, len(self), batch_size):
            if drop_last_batch and i + batch_size > len(self):
                continue
            if collate:
                yield self.collate(self[i : i + batch_size])
            else:
                yield self[i : i + batch_size]

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> NumpyArrayColumn:
        # Assert that the path exists
        assert os.path.exists(path), f"`path` {path} does not exist."

        # Load in the metadata
        metadata = dict(
            yaml.load(
                open(os.path.join(path, "meta.yaml")),
                Loader=yaml.FullLoader,
            )
        )
        assert metadata["dtype"] == cls

        # Load in the data
        data = np.load(os.path.join(path, "data.npy"))

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
