import numpy as np
import pandas as pd
import pytest
import torch

from meerkat import TensorColumn
from meerkat.block.numpy_block import NumpyBlock

from .abstract import AbstractColumnTestBed, TestAbstractColumn


class TensorColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "num_dims": [1, 2, 3],
        "dim_length": [1, 5],
        "dtype": ["float", "int"],
    }

    def __init__(
        self,
        length: int = 16,
        num_dims: int = True,
        dim_length: int = 5,
        dtype="float",
        seed: int = 123,
        tmpdir: str = None,
    ):
        self.dtype = dtype
        np.random.seed(seed)
        array = (
            np.random.random((length, *[dim_length for _ in range(num_dims - 1)])) * 10
        )
        array = torch.tensor(array).to({"int": torch.int, "float": torch.float}[dtype])

        self.col = TensorColumn(array)
        self.data = array

    def get_map_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        salt: int = 1,
        kwarg: int = 0,
    ):
        return {
            "fn": lambda x, k=0: x + salt + k,
            "expected_result": TensorColumn(self.col.data + salt + kwarg),
        }

    def get_filter_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        salt: int = 1,
        kwarg: int = 0,
    ):
        return {
            "fn": lambda x, k=0: (
                (x > 3 + salt + k).to(dtype=bool) if batched else (x > 3 + salt + k)
            ),
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


class TestTensorColumn(TestAbstractColumn):

    __test__ = True
    testbed_class: type = TensorColumnTestBed
    column_class: type = TensorColumn

    def test_init_block(self):
        block_view = NumpyBlock(np.zeros((10, 10)))[0]
        with pytest.raises(ValueError):
            TensorColumn(block_view)

    def _get_data_to_set(self, testbed, data_index):
        return torch.zeros_like(testbed.get_data(data_index))

    @TensorColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_set_item(self, testbed, index_type: type):
        return super().test_set_item(testbed, index_type=index_type)

    @TensorColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_getitem(self, testbed, index_type: type):
        return super().test_getitem(testbed, index_type=index_type)

    @TensorColumnTestBed.parametrize(
        config={"num_dims": [1], "dim_length": [1]}, params={"batched": [True, False]}
    )
    def test_filter_1(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_filter_1(testbed, batched, materialize=True)

    @TensorColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_multiple(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_map_return_multiple(testbed, batched, materialize=True)

    @TensorColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_map_return_single(testbed, batched, materialize=True)

    @TensorColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_single_w_kwarg(
        self, testbed: AbstractColumnTestBed, batched: bool
    ):
        return super().test_map_return_single_w_kwarg(
            testbed, batched, materialize=True
        )

    @TensorColumnTestBed.parametrize(params={"n": [1, 2, 3]})
    def test_concat(self, testbed: AbstractColumnTestBed, n: int):
        return super().test_concat(testbed, n=n)

    @TensorColumnTestBed.parametrize()
    def test_copy(self, testbed: AbstractColumnTestBed):
        return super().test_copy(testbed)

    @TensorColumnTestBed.parametrize()
    def test_io(self, tmp_path, testbed):
        super().test_io(tmp_path, testbed)

    @TensorColumnTestBed.parametrize()
    def test_pickle(self, testbed):
        super().test_pickle(testbed)

    @TensorColumnTestBed.parametrize()
    def test_to_tensor(self, testbed):
        col, _ = testbed.col, testbed.data

        tensor = col.to_tensor()

        assert torch.is_tensor(tensor)
        assert (col == tensor.numpy()).all()

    @TensorColumnTestBed.parametrize()
    def test_to_pandas(self, testbed):
        series = testbed.col.to_pandas()

        assert isinstance(series, pd.Series)

        if testbed.col.shape == 1:
            assert (series.values == testbed.col.data).all()
        else:
            for idx in range(len(testbed.col)):
                assert (series.iloc[idx] == testbed.col[idx]).all()

    @TensorColumnTestBed.parametrize()
    def test_repr_pandas(self, testbed):
        series = testbed.col.to_pandas()
        assert isinstance(series, pd.Series)

    def test_ufunc_unhandled(self):
        a = TensorColumn([1, 2, 3])
        with pytest.raises(TypeError):
            a == "a"
