import os

import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest
import torch
from numpy.lib.format import open_memmap

from meerkat import NumPyTensorColumn, TorchTensorColumn
from meerkat.block.numpy_block import NumPyBlock

from ....utils import product_parametrize
from ..abstract import AbstractColumnTestBed, column_parametrize


class NumPyTensorColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "num_dims": [1, 2, 3],
        "dim_length": [1, 5],
        "dtype": ["float", "int"],
        "mmap": [True, False],
    }

    marks = pytest.mark.numpy_col

    def __init__(
        self,
        length: int = 16,
        num_dims: int = True,
        dim_length: int = 5,
        dtype="float",
        mmap: bool = False,
        seed: int = 123,
        tmpdir: str = None,
    ):
        self.dtype = dtype
        np.random.seed(seed)
        array = (
            np.random.random((length, *[dim_length for _ in range(num_dims - 1)])) * 10
        )
        array = array.astype(dtype)
        if mmap:
            mmap = open_memmap(
                filename=os.path.join(tmpdir, "mmap"),
                dtype=array.dtype,
                shape=array.shape,
                mode="w+",
            )
            mmap[:] = array
            self.col = NumPyTensorColumn.from_array(mmap)
        else:
            self.col = NumPyTensorColumn.from_array(array)
        self.data = array

    def get_map_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        kwarg: int = 0,
        salt: int = 1,
    ):
        return {
            "fn": lambda x, k=0: x + salt + k,
            "expected_result": NumPyTensorColumn.from_array(
                self.col.data + salt + kwarg
            ),
            "output_type": NumPyTensorColumn
        }

    def get_filter_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        kwarg: int = 0,
        salt: int = 1,
    ):
        return {
            "fn": lambda x, k=0: x > 3 + k + salt,
            "expected_result": self.col[self.col.data > 3 + salt + kwarg],
        }

    def get_data(self, index, materialize=True):
        return self.data[index]

    def get_data_to_set(self, data_index):
        return np.zeros_like(self.get_data(data_index))

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        assert (data1 == data2).all()


@pytest.fixture(**column_parametrize([NumPyTensorColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


def test_init_block():
    block_view = NumPyBlock(np.zeros((10, 10)))[0]
    with pytest.raises(ValueError):
        TorchTensorColumn(block_view)


@product_parametrize(params={"batched": [True, False]})
def test_map_return_single_mmap(tmpdir, testbed: AbstractColumnTestBed, batched: bool):
    col = testbed.col
    map_spec = testbed.get_map_spec(batched=batched)

    def func(x):
        out = map_spec["fn"](x)
        return out

    mmap_path = os.path.join(tmpdir, "mmap_path")

    result = col.map(
        func,
        batch_size=4,
        mmap=True,
        mmap_path=mmap_path,
        is_batched_fn=batched,
        output_type=map_spec.get("output_type", None),
    )
    assert result.is_equal(map_spec["expected_result"])

    assert isinstance(result.data, np.memmap)
    assert result.data.filename == mmap_path


@product_parametrize(params={"link": [True, False], "mmap": [True, False]})
def test_io_mmap(tmp_path, testbed, link, mmap):
    col = testbed.col

    path = os.path.join(tmp_path, "test")
    col.write(path, link=link)

    assert os.path.islink(os.path.join(path, "data.npy")) == (link and col.is_mmap)

    new_col = NumPyTensorColumn.read(path, mmap=mmap)

    assert isinstance(new_col, NumPyTensorColumn)
    assert col.is_equal(new_col)
    assert new_col.is_mmap == mmap


def test_to_tensor(testbed):
    col, _ = testbed.col, testbed.data

    tensor = col.to_tensor()

    assert torch.is_tensor(tensor)
    assert (col == tensor.numpy()).all()


def test_from_array():
    # Build a dataset from a batch
    array = np.random.rand(10, 3, 3)
    col = NumPyTensorColumn.from_array(array)

    assert (col == array).all()
    np_test.assert_equal(len(col), 10)


def test_to_pandas(testbed):
    series = testbed.col.to_pandas()

    assert isinstance(series, pd.Series)
 
    if testbed.col.shape == 1:
        assert (series.values == testbed.col.data).all()
    else:
        for idx in range(len(testbed.col)):
            assert (series.iloc[idx] == testbed.col[idx]).all()


def test_repr_pandas(testbed):
    series = testbed.col.to_pandas()
    assert isinstance(series, pd.Series)


def test_ufunc_out():
    out = np.zeros(3)
    a = NumPyTensorColumn([1, 2, 3])
    b = NumPyTensorColumn([1, 2, 3])
    result = np.add(a, b, out=out)
    assert (result.data == out).all()


def test_ufunc_at():
    a = NumPyTensorColumn([1, 2, 3])
    result = np.add.at(a, [0, 1, 1], 1)
    assert result is None
    assert a.is_equal(NumPyTensorColumn([2, 4, 3]))


def test_ufunc_unhandled():
    a = NumPyTensorColumn([1, 2, 3])
    with pytest.raises(TypeError):
        a == "a"
