import numpy as np
import pandas as pd
import pytest
import torch

from meerkat import TensorColumn, TorchTensorColumn
from meerkat.block.numpy_block import NumPyBlock

from ..abstract import AbstractColumnTestBed, column_parametrize


class TensorColumnTestBed(AbstractColumnTestBed):
    DEFAULT_CONFIG = {
        "num_dims": [1, 2, 3],
        "dim_length": [1, 5],
        "dtype": ["float", "int"],
    }

    marks = pytest.mark.tensor_col

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

        self.col = TorchTensorColumn(array)
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
            "expected_result": TorchTensorColumn(self.col.data + salt + kwarg),
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

    def get_data_to_set(self, data_index):
        return torch.zeros_like(self.get_data(data_index))

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        assert (data1 == data2).all()


@pytest.fixture(**column_parametrize([TensorColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


def test_init_block():
    block_view = NumPyBlock(np.zeros((10, 10)))[0]
    with pytest.raises(ValueError):
        TorchTensorColumn(block_view)


def test_to_tensor(testbed):
    col, _ = testbed.col, testbed.data

    tensor = col.to_tensor()

    assert torch.is_tensor(tensor)
    assert (col == tensor.numpy()).all()


def test_to_pandas(testbed):
    series = testbed.col.to_pandas()

    assert isinstance(series, pd.Series)

    if testbed.col.shape == 1:
        assert (series.values == testbed.col.data).all()
    else:
        for idx in range(len(testbed.col)):
            assert (series.iloc[idx] == testbed.col[idx].numpy()).all()


def test_repr_pandas(testbed):
    series = testbed.col.to_pandas()
    assert isinstance(series, pd.Series)


def test_ufunc_unhandled():
    a = TorchTensorColumn([1, 2, 3])
    with pytest.raises(TypeError):
        a == "a"


@pytest.mark.parametrize(
    "data",
    [[1, 2, 3], np.asarray([1, 2, 3]), torch.tensor([1, 2, 3]), pd.Series([1, 2, 3])],
)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_backend(data, backend: str):
    col = TensorColumn(data, backend=backend)

    expected_type = {"numpy": np.ndarray, "torch": torch.Tensor}[backend]
    assert isinstance(col.data, expected_type)

    col_data = col.data
    if isinstance(col_data, torch.Tensor):
        col_data = col_data.numpy()
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    col_data = np.asarray(col_data)
    data = np.asarray(data)
    assert (col_data == data).all()
