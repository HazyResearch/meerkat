import os

import numpy as np
import numpy.testing as np_test
import pandas as pd
import pytest
import torch
from numpy.lib.format import open_memmap

from meerkat import NumpyArrayColumn
from meerkat.block.tensor_block import TensorBlock

from .abstract import AbstractColumnTestBed, TestAbstractColumn


class NumpyArrayColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "num_dims": [1, 2, 3],
        "dim_length": [1, 5],
        "dtype": ["float", "int"],
        "mmap": [True, False],
    }

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
            self.col = NumpyArrayColumn.from_array(mmap)
        else:
            self.col = NumpyArrayColumn.from_array(array)
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
            "expected_result": NumpyArrayColumn.from_array(
                self.col.data + salt + kwarg
            ),
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

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        assert (data1 == data2).all()


@pytest.fixture
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


class TestNumpyArrayColumn(TestAbstractColumn):

    __test__ = True
    testbed_class: type = NumpyArrayColumnTestBed
    column_class: type = NumpyArrayColumn

    def test_init_block(self):
        block_view = TensorBlock(torch.zeros(10, 10))[0]
        with pytest.raises(ValueError):
            NumpyArrayColumn(block_view)

    def _get_data_to_set(self, testbed, data_index):
        return np.zeros_like(testbed.get_data(data_index))

    @NumpyArrayColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_set_item(self, testbed, index_type: type):
        return super().test_set_item(testbed, index_type=index_type)

    @NumpyArrayColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_getitem(self, testbed, index_type: type):
        return super().test_getitem(testbed, index_type=index_type)

    @NumpyArrayColumnTestBed.parametrize(
        config={"num_dims": [1], "dim_length": [1]}, params={"batched": [True, False]}
    )
    def test_filter_1(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_filter_1(testbed, batched, materialize=True)

    @NumpyArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_multiple(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_map_return_multiple(testbed, batched, materialize=True)

    @NumpyArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_map_return_single(testbed, batched, materialize=True)

    @NumpyArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single_w_kwarg(
        self, testbed: AbstractColumnTestBed, batched: bool
    ):
        return super().test_map_return_single_w_kwarg(
            testbed, batched, materialize=True
        )

    @NumpyArrayColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single_mmap(
        self, tmpdir, testbed: AbstractColumnTestBed, batched: bool
    ):
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

    @NumpyArrayColumnTestBed.parametrize(params={"n": [1, 2, 3]})
    def test_concat(self, testbed: AbstractColumnTestBed, n: int):
        return super().test_concat(testbed, n=n)

    @NumpyArrayColumnTestBed.parametrize()
    def test_copy(self, testbed: AbstractColumnTestBed):
        return super().test_copy(testbed)

    @NumpyArrayColumnTestBed.parametrize()
    def test_io(self, tmp_path, testbed):
        super().test_io(tmp_path, testbed)

    @NumpyArrayColumnTestBed.parametrize(
        params={"link": [True, False], "mmap": [True, False]}
    )
    def test_io_mmap(self, tmp_path, testbed, link, mmap):
        col = testbed.col

        path = os.path.join(tmp_path, "test")
        col.write(path, link=link)

        assert os.path.islink(os.path.join(path, "data.npy")) == (link and col.is_mmap)

        new_col = self.column_class.read(path, mmap=mmap)

        assert isinstance(new_col, self.column_class)
        assert col.is_equal(new_col)
        assert new_col.is_mmap == mmap

    @NumpyArrayColumnTestBed.parametrize()
    def test_pickle(self, testbed):
        super().test_pickle(testbed)

    @NumpyArrayColumnTestBed.parametrize()
    def test_to_tensor(self, testbed):
        col, _ = testbed.col, testbed.data

        tensor = col.to_tensor()

        assert torch.is_tensor(tensor)
        assert (col == tensor.numpy()).all()

    def test_from_array(self):
        # Build a dataset from a batch
        array = np.random.rand(10, 3, 3)
        col = NumpyArrayColumn.from_array(array)

        assert (col == array).all()
        np_test.assert_equal(len(col), 10)

    @NumpyArrayColumnTestBed.parametrize()
    def test_to_pandas(self, testbed):
        series = testbed.col.to_pandas()

        assert isinstance(series, pd.Series)

        if testbed.col.shape == 1:
            assert (series.values == testbed.col.data).all()
        else:
            for idx in range(len(testbed.col)):
                assert (series.iloc[idx] == testbed.col[idx]).all()

    @NumpyArrayColumnTestBed.parametrize()
    def test_repr_pandas(self, testbed):
        series = testbed.col.to_pandas()
        assert isinstance(series, pd.Series)

    def test_ufunc_out(self):
        out = np.zeros(3)
        a = NumpyArrayColumn([1, 2, 3])
        b = NumpyArrayColumn([1, 2, 3])
        result = np.add(a, b, out=out)
        assert result.data is out

    def test_ufunc_at(self):
        a = NumpyArrayColumn([1, 2, 3])
        result = np.add.at(a, [0, 1, 1], 1)
        assert result is None
        assert a.is_equal(NumpyArrayColumn([2, 4, 3]))

    def test_ufunc_unhandled(self):
        a = NumpyArrayColumn([1, 2, 3])
        with pytest.raises(TypeError):
            a == "a"
