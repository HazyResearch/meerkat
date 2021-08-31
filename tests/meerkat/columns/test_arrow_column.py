"""Unittests for NumpyColumn."""


import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from meerkat import ArrowArrayColumn
from meerkat.block.tensor_block import TensorBlock

from .abstract import AbstractColumnTestBed, TestAbstractColumn


class ArrowArrayColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "dtype": ["float", "int", "str"],
    }

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

        self.col = ArrowArrayColumn(array)
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
                    x.data.to_numpy() + salt + (k if self.dtype != "str" else str(k))
                ),
                "expected_result": ArrowArrayColumn(
                    self.col.data.to_numpy(zero_copy_only=False) + salt + kwarg
                ),
                "output_type": ArrowArrayColumn,
            }

        else:
            return {
                "fn": lambda x, k=0: x + salt + (k if self.dtype != "str" else str(k)),
                "expected_result": ArrowArrayColumn(
                    self.col.data.to_numpy(zero_copy_only=False) + salt + kwarg
                ),
                "output_type": ArrowArrayColumn,
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
        return {
            "fn": lambda x, k=0: x.to_numpy()
            > salt + (k if self.dtype != "str" else str(k)),
            "expected_result": self.col[self.col.data > salt + kwarg],
        }

    def get_data(self, index, materialize: bool = True):
        return self.data[index]

    @staticmethod
    def assert_data_equal(data1: pa.Array, data2: pa.Array):
        if isinstance(data1, (pa.Array, pa.ChunkedArray)):
            assert data1.equals(data2)
        else:
            assert data1 == data2


@pytest.fixture
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


class TestArrowArrayColumn(TestAbstractColumn):
    __test__ = True
    testbed_class: type = ArrowArrayColumnTestBed
    column_class: type = ArrowArrayColumn

    def test_init_block(self):
        block_view = TensorBlock(torch.zeros(10, 10))[0]
        with pytest.raises(ValueError):
            ArrowArrayColumn(block_view)

    def _get_data_to_set(self, testbed, data_index):
        if isinstance(data_index, int):
            return 0
        return pd.Series(np.zeros_like(testbed.get_data(data_index).values))

    @ArrowArrayColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_set_item(self, testbed, index_type: type):
        return super().test_set_item(testbed, index_type=index_type)

    @ArrowArrayColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_getitem(self, testbed, index_type: type):
        return super().test_getitem(testbed, index_type=index_type)

    @ArrowArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_filter_1(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_filter_1(testbed, batched, materialize=True)

    @ArrowArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_multiple(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_map_return_multiple(testbed, batched, materialize=True)

    @ArrowArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_map_return_single(testbed, batched, materialize=True)

    @ArrowArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single_w_kwarg(
        self, testbed: AbstractColumnTestBed, batched: bool
    ):
        return super().test_map_return_single_w_kwarg(
            testbed, batched, materialize=True
        )

    @ArrowArrayColumnTestBed.parametrize(params={"n": [1, 2, 3]})
    def test_concat(self, testbed: AbstractColumnTestBed, n: int):
        return super().test_concat(testbed, n=n)

    @ArrowArrayColumnTestBed.parametrize()
    def test_copy(self, testbed: AbstractColumnTestBed):
        return super().test_copy(testbed)

    @ArrowArrayColumnTestBed.parametrize()
    def test_io(self, tmp_path, testbed):
        super().test_io(tmp_path, testbed)

    @ArrowArrayColumnTestBed.parametrize()
    def test_pickle(self, testbed):
        super().test_pickle(testbed)

    @ArrowArrayColumnTestBed.parametrize()
    def test_to_tensor(self, testbed):
        col, _ = testbed.col, testbed.data
        if testbed.dtype == "str":
            with pytest.raises(ValueError):
                col.to_tensor()
        else:
            tensor = col.to_tensor()

            assert torch.is_tensor(tensor)
            assert (col == tensor.numpy()).all()

    @ArrowArrayColumnTestBed.parametrize()
    def test_to_pandas(self, testbed):
        col, _ = testbed.col, testbed.data
        series = col.to_pandas()
        assert isinstance(series, pd.Series)
        assert (col.data.values == series.values).all()

    @ArrowArrayColumnTestBed.parametrize()
    def test_repr_pandas(self, testbed):
        series = testbed.col.to_pandas()
        assert isinstance(series, pd.Series)

    def test_ufunc_out(self):
        out = np.zeros(3)
        a = ArrowArrayColumn([1, 2, 3])
        b = ArrowArrayColumn([1, 2, 3])
        np.add(a, b, out=out)
        assert (out == np.array([2, 4, 6])).all()
