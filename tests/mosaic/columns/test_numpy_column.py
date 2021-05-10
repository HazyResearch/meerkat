"""Unittests for NumpyColumn."""
import os
import pickle
from itertools import product

import numpy as np
import numpy.testing as np_test
import pytest
import torch

from mosaic import NumpyArrayColumn
from mosaic.datapanel import DataPanel


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
    col = NumpyArrayColumn.from_array(array)

    if use_visible_rows:
        visible_rows = [0, 4, 6, 10]
        col.visible_rows = visible_rows
        array = array[visible_rows]

    return col, array


def test_from_array():
    # Build a dataset from a batch
    array = np.random.rand(10, 3, 3)
    col = NumpyArrayColumn.from_array(array)

    assert (col == array).all()
    np_test.assert_equal(len(col), 10)


@pytest.mark.parametrize(
    "dtype,use_visible_rows,batched",
    product(["float", "int"], [True, False], [True, False]),
)
def test_map_return_single(dtype, use_visible_rows, batched):
    """`map`, single return,"""
    col, array = _get_data(dtype=dtype, use_visible_rows=use_visible_rows)

    def func(x):
        out = x.mean(axis=-1)
        return out

    result = col.map(func, batch_size=4, batched=batched)
    assert isinstance(result, NumpyArrayColumn)
    np_test.assert_equal(len(result), len(array))
    assert (result == array.mean(axis=-1)).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows, batched",
    product(["float", "int"], [True, False], [True, False]),
)
def test_map_return_multiple(dtype, use_visible_rows, batched):
    """`map`, multiple return."""
    col, array = _get_data(dtype=dtype, use_visible_rows=use_visible_rows)

    def func(x):
        return {"mean": x.mean(axis=-1), "std": x.std(axis=-1)}

    result = col.map(func, batch_size=4, batched=batched)
    assert isinstance(result, DataPanel)
    assert isinstance(result["std"], NumpyArrayColumn)
    assert isinstance(result["mean"], NumpyArrayColumn)
    np_test.assert_equal(len(result), len(array))
    assert (result["mean"] == array.mean(axis=-1)).all()
    assert (result["std"] == array.std(axis=-1)).all()


@pytest.mark.parametrize(
    "multiple_dim,dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_set_item_1(multiple_dim, dtype, use_visible_rows):
    col, array = _get_data(
        multiple_dim=multiple_dim, dtype=dtype, use_visible_rows=use_visible_rows
    )
    index = [0, 3]
    not_index = [i for i in range(col.shape[0]) if i not in index]
    col[index] = 0

    assert (col[not_index] == array[not_index]).all()
    assert (col[index] == 0).all()


@pytest.mark.parametrize(
    "multiple_dim,dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_set_item_2(multiple_dim, dtype, use_visible_rows):
    col, array = _get_data(
        multiple_dim=multiple_dim, dtype=dtype, use_visible_rows=use_visible_rows
    )
    index = 0
    not_index = [i for i in range(col.shape[0]) if i != index]
    col[index] = 0

    assert (col[not_index] == array[not_index]).all()
    assert (col[index] == 0).all()


@pytest.mark.parametrize(
    "use_visible_rows,dtype,batched",
    product([True, False], ["float", "int"], [True, False]),
)
def test_filter_1(use_visible_rows, dtype, batched):
    """multiple_dim=False."""
    col, array = _get_data(
        multiple_dim=False, dtype=dtype, use_visible_rows=use_visible_rows
    )

    def func(x):
        return x > 20

    result = col.filter(func, batch_size=4, batched=batched)
    assert isinstance(result, NumpyArrayColumn)
    assert len(result) == (array > 20).sum()


@pytest.mark.parametrize(
    "multiple_dim, dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_pickle(multiple_dim, dtype, use_visible_rows):
    # important for dataloader
    col, _ = _get_data(
        multiple_dim=multiple_dim, dtype=dtype, use_visible_rows=use_visible_rows
    )
    buf = pickle.dumps(col)
    new_col = pickle.loads(buf)

    assert isinstance(new_col, NumpyArrayColumn)
    assert (col == new_col).all()


@pytest.mark.parametrize(
    "multiple_dim, dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_io(tmp_path, multiple_dim, dtype, use_visible_rows):
    # uses the tmp_path fixture which will provide a
    # temporary directory unique to the test invocation,
    # important for dataloader
    col, _ = _get_data(
        multiple_dim=multiple_dim, dtype=dtype, use_visible_rows=use_visible_rows
    )
    path = os.path.join(tmp_path, "test")
    col.write(path)

    new_col = NumpyArrayColumn.read(path)

    assert isinstance(new_col, NumpyArrayColumn)
    assert (col == new_col).all()


@pytest.mark.parametrize(
    "multiple_dim,dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_copy(multiple_dim, dtype, use_visible_rows):
    col, _ = _get_data(
        multiple_dim=multiple_dim, dtype=dtype, use_visible_rows=use_visible_rows
    )
    col_copy = col.copy()

    assert isinstance(col_copy, NumpyArrayColumn)
    assert (col == col_copy).all()


@pytest.mark.parametrize(
    "multiple_dim,dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_to_tensor(multiple_dim, dtype, use_visible_rows):
    col, _ = _get_data(
        multiple_dim=multiple_dim, dtype=dtype, use_visible_rows=use_visible_rows
    )
    tensor = col.to_tensor()

    assert torch.is_tensor(tensor)
    assert (col == tensor.numpy()).all()
