"""Unittests for NumpyColumn."""
import os
import pickle
from itertools import product

import numpy as np
import numpy.testing as np_test
import pytest
import torch

from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel


def _get_data(multiple_dim: bool = True, dtype="float", use_visible_rows=False):
    if multiple_dim:
        array = np.array(
            [
                [
                    [0.5565041, 1.51486395, 0],
                    [123, 0.60526485, 0.7246723],
                ],
                [
                    [0.3156991, 0.82733837, 45],
                    [0.71086498, 0, 0],
                ],
                [
                    [0, 0.17152445, 0.06989294],
                    [0.59578079, 0.03738921, 0],
                ],
                [
                    [0.49596023, 0, 0.56062833],
                    [0.31457122, 0.19126629, 16],
                ],
            ]
            * 4  # shape (16, 2, 3)
        )
    else:
        array = np.array([0.3969655, 23.26084479, 0, 123] * 4)
    array = array.astype(dtype)
    col = TensorColumn(array)

    if use_visible_rows:
        visible_rows = [0, 4, 6, 10]
        col.visible_rows = visible_rows
        array = array[visible_rows]

    return col, array


def test_from_array():
    # Build a dataset from a batch
    array = np.random.rand(10, 3, 3)
    col = TensorColumn(array)

    assert (col == array).all()
    np_test.assert_equal(len(col), 10)


@pytest.mark.parametrize(
    "dtype,batched",
    product(["float", "int"], [True, False]),
)
def test_map_return_single(dtype, batched):
    """`map`, single return,"""
    col, array = _get_data(
        dtype=dtype,
    )

    def func(x):
        out = x.type(torch.FloatTensor).mean(axis=-1)
        return out

    result = col.map(
        func, batch_size=4, is_batched_fn=batched, output_type=TensorColumn
    )
    assert isinstance(result, TensorColumn)
    np_test.assert_equal(len(result), len(array))
    assert np.allclose(result.numpy(), array.mean(axis=-1))


@pytest.mark.parametrize(
    "dtype, batched",
    product(["float", "int"], [True, False]),
)
def test_map_return_multiple(dtype, batched):
    """`map`, multiple return."""
    col, array = _get_data(
        dtype=dtype,
    )

    def func(x):
        return {
            "mean": x.type(torch.FloatTensor).mean(axis=-1),
            "std": x.type(torch.FloatTensor).std(axis=-1),
        }

    result = col.map(func, batch_size=4, is_batched_fn=batched)
    assert isinstance(result, DataPanel)
    assert isinstance(result["std"], TensorColumn)
    assert isinstance(result["mean"], TensorColumn)
    np_test.assert_equal(len(result), len(array))
    assert np.allclose(result["mean"].numpy(), array.mean(axis=-1))
    assert np.allclose(result["std"].numpy(), array.std(axis=-1, ddof=1))


@pytest.mark.parametrize(
    "multiple_dim,dtype",
    product([True, False], ["float", "int"]),
)
def test_set_item_1(multiple_dim, dtype):
    col, array = _get_data(
        multiple_dim=multiple_dim,
        dtype=dtype,
    )
    index = [0, 3]
    not_index = [i for i in range(col.shape[0]) if i not in index]
    col[index] = 0

    assert (col[not_index] == array[not_index]).all()
    assert (col[index] == 0).all()


@pytest.mark.parametrize(
    "multiple_dim,dtype",
    product([True, False], ["float", "int"]),
)
def test_set_item_2(
    multiple_dim,
    dtype,
):
    col, array = _get_data(
        multiple_dim=multiple_dim,
        dtype=dtype,
    )
    index = 0
    not_index = [i for i in range(col.shape[0]) if i != index]
    col[index] = 0

    assert (col[not_index] == array[not_index]).all()
    assert (col[index] == 0).all()


@pytest.mark.parametrize(
    "multiple_dim, dtype",
    product([True, False], ["float", "int"]),
)
def test_pickle(
    multiple_dim,
    dtype,
):
    # important for dataloader
    col, _ = _get_data(
        multiple_dim=multiple_dim,
        dtype=dtype,
    )
    buf = pickle.dumps(col)
    new_col = pickle.loads(buf)

    assert isinstance(new_col, TensorColumn)
    assert (col == new_col).all()


@pytest.mark.parametrize(
    "multiple_dim, dtype",
    product([True, False], ["float", "int"]),
)
def test_io(
    tmp_path,
    multiple_dim,
    dtype,
):
    # uses the tmp_path fixture which will provide a
    # temporary directory unique to the test invocation,
    # important for dataloader
    col, _ = _get_data(
        multiple_dim=multiple_dim,
        dtype=dtype,
    )
    path = os.path.join(tmp_path, "test")
    col.write(path)

    new_col = TensorColumn.read(path)

    assert isinstance(new_col, TensorColumn)
    assert (col == new_col).all()


@pytest.mark.parametrize(
    "multiple_dim,dtype",
    product([True, False], ["float", "int"]),
)
def test_copy(
    multiple_dim,
    dtype,
):
    col, _ = _get_data(
        multiple_dim=multiple_dim,
        dtype=dtype,
    )
    col_copy = col.copy()

    assert isinstance(col_copy, TensorColumn)
    assert (col == col_copy).all()


def test_tensor_ops():
    """Test prototype tensor operations on tensor columns."""
    col = TensorColumn(torch.ones(4, 3))

    assert torch.all(torch.sum(col, dim=1) == 3)
    assert torch.all(col.sum() == torch.sum(col))
    assert torch.all(col.sum(dim=1) == torch.sum(col, dim=1))

    assert torch.cat([col, col]).shape == (8, 3)
    assert torch.vstack([col, col]).shape == (8, 3)
    assert torch.cat([col, col, col], dim=1).shape == (4, 9)

    assert torch.stack([col, col], dim=0).shape == (2, 4, 3)

    chunk1, chunk2 = torch.chunk(col, chunks=2, dim=0)
    assert chunk1.shape == (2, 3)
    assert chunk2.shape == (2, 3)

    col_nd = TensorColumn(torch.ones(4, 3, 5, 6))
    assert col_nd.permute(3, 2, 1, 0).shape == col_nd.shape[::-1]
