"""Unittests for NumpyColumn."""
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from meerkat import ArrowScalarColumn
from meerkat.block.tensor_block import TensorBlock

from ..abstract import AbstractColumnTestBed, column_parametrize


def to_numpy(array: Union[pa.Array, pa.ChunkedArray]):
    """For non-chunked arrays, need to pass zero_copy_only=False."""
    if isinstance(array, pa.ChunkedArray):
        return array.to_numpy()
    return array.to_numpy(zero_copy_only=False)


class ArrowScalarColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "dtype": ["float", "int", "str"],
    }

    marks = pytest.mark.arrow_col

    def __init__(
        self,
        length: int = 16,
        dtype="float",
        seed: int = 123,
        tmpdir: str = None,
    ):
        self.dtype = dtype
        np.random.seed(seed)
        array = np.random.random(length) * 10
        if dtype == "float":
            array = pa.array(array, type=pa.float64())
        elif dtype == "int":
            array = array.astype(int)
            array = pa.array(array, type=pa.int64())
        elif dtype == "str":
            array = pd.Series(array).astype("str")
            array = pa.array(array, type=pa.string())
        else:
            raise ValueError(f"dtype {dtype} not supported.")

        self.col = ArrowScalarColumn(array)
        self.data = array

    def get_map_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        salt: int = 1,
        kwarg: int = 0,
    ):
        salt = salt if self.dtype != "str" else str(salt)
        kwarg = kwarg if self.dtype != "str" else str(kwarg)
        if batched:
            return {
                "fn": lambda x, k=0: pa.array(
                    to_numpy(x.data) + salt + (k if self.dtype != "str" else str(k))
                ),
                "expected_result": ArrowScalarColumn(
                    to_numpy(self.col.data) + salt + kwarg
                ),
                "output_type": ArrowScalarColumn,
            }

        else:
            return {
                "fn": lambda x, k=0: x.as_py()
                + salt
                + (k if self.dtype != "str" else str(k)),
                "expected_result": ArrowScalarColumn(
                    to_numpy(self.col.data) + salt + kwarg
                ),
                "output_type": ArrowScalarColumn,
            }

    def get_filter_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        salt: int = 1,
        kwarg: int = 0,
    ):
        salt = 3 + salt if self.dtype != "str" else str(3 + salt)
        kwarg = kwarg if self.dtype != "str" else str(kwarg)
        if batched:
            return {
                "fn": lambda x, k=0: to_numpy(x.data)
                > salt + (k if self.dtype != "str" else str(k)),
                "expected_result": self.col[to_numpy(self.col.data) > salt + kwarg],
            }

        else:
            return {
                "fn": lambda x, k=0: x.as_py()
                > salt + (k if self.dtype != "str" else str(k)),
                "expected_result": self.col[to_numpy(self.col.data) > salt + kwarg],
            }

    def get_data(self, index, materialize: bool = True):
        if isinstance(index, slice) or isinstance(index, int):
            data = self.data[index]
        elif index.dtype == bool:
            data = self.data.filter(pa.array(index))
        else:
            data = self.data.take(index)
        return data

    def get_data_to_set(self, data_index):
        if isinstance(data_index, int):
            return 0
        return pd.Series(np.zeros(len(self.data)))

    @staticmethod
    def assert_data_equal(data1: pa.Array, data2: pa.Array):
        if isinstance(data1, (pa.Array, pa.ChunkedArray)):
            assert (to_numpy(data1) == to_numpy(data2)).all()
        else:
            assert data1 == data2


@pytest.fixture(**column_parametrize([ArrowScalarColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


def test_init_block():
    block_view = TensorBlock(torch.zeros(10, 10))[0]
    with pytest.raises(ValueError):
        ArrowScalarColumn(block_view)


def test_to_numpy(testbed):
    col, _ = testbed.col, testbed.data
    array = col.to_numpy()

    assert isinstance(array, np.ndarray)
    assert (col.data.to_numpy(zero_copy_only=False) == array).all()


def test_to_tensor(testbed):
    col, _ = testbed.col, testbed.data
    if testbed.dtype == "str":
        with pytest.raises(ValueError):
            col.to_tensor()
    else:
        tensor = col.to_tensor()

        assert torch.is_tensor(tensor)
        assert (col.data.to_numpy() == tensor.numpy()).all()


def test_to_pandas(testbed):
    col, _ = testbed.col, testbed.data
    series = col.to_pandas()
    assert isinstance(series, pd.Series)
    assert (col.data.to_pandas() == series.values).all()


def test_repr_pandas(testbed):
    series = testbed.col.to_pandas()
    assert isinstance(series, pd.Series)
