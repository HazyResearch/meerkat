"""Unittests for NumpyColumn."""
import os
import pickle
from functools import wraps
from itertools import product

import numpy as np
import numpy.testing as np_test
import pytest
import torch

from meerkat import NumpyArrayColumn
from meerkat.datapanel import DataPanel


class NumpyArrayColumnTestBed:

    DEFAULT_CONFIG = {
        "num_dims": [1, 2, 3],
        "dim_length": [1, 5],
        "dtype": ["float", "int"],
    }

    def __init__(
        self, length: int = 16, num_dims: int = True, dim_length: int = 5, dtype="float"
    ):
        np.random.seed(123)
        array = (
            np.random.random((length, *[dim_length for _ in range(num_dims - 1)])) * 10
        )
        array = array.astype(dtype)

        self.col = NumpyArrayColumn.from_array(array)
        self.data = array

    @classmethod
    def get_params(cls, config: dict = None, params: dict = None):
        updated_config = cls.DEFAULT_CONFIG.copy()
        if config is not None:
            updated_config.update(config)
        configs = list(
            map(
                dict,
                product(*[[(k, v) for v in vs] for k, vs in updated_config.items()]),
            )
        )
        if params is None:
            return "config", configs
        else:
            return "config," + ",".join(params.keys()), product(
                configs, *params.values()
            )

    @classmethod
    @wraps(pytest.mark.parametrize)
    def parametrize(cls, config: dict = None, params: dict = None):
        return pytest.mark.parametrize(
            *NumpyArrayColumnTestBed.get_params(config=config, params=params)
        )


def test_from_array():
    # Build a dataset from a batch
    array = np.random.rand(10, 3, 3)
    col = NumpyArrayColumn.from_array(array)

    assert (col == array).all()
    np_test.assert_equal(len(col), 10)


@NumpyArrayColumnTestBed.parametrize(
    config={"num_dims": [2]},
    params={"batched": [True, False], "use_kwargs": [True, False]},
)
def test_map_return_single(config, batched, use_kwargs):
    """`map`, single return,"""
    testbed = NumpyArrayColumnTestBed(**config)
    col, array = testbed.col, testbed.data

    def func(x, bias=0):
        out = x.mean(axis=-1) + bias
        return out

    bias = 1 if use_kwargs else 0
    kwargs = {"bias": bias} if use_kwargs else {}

    result = col.map(func, batch_size=4, is_batched_fn=batched, **kwargs)

    assert isinstance(result, NumpyArrayColumn)
    np_test.assert_equal(len(result), len(array))
    assert (result == array.mean(axis=-1) + bias).all()


@NumpyArrayColumnTestBed.parametrize(
    config={"num_dims": [2]},
    params={"batched": [True, False], "use_kwargs": [True, False]},
)
def test_map_return_multiple(config, batched, use_kwargs):
    """`map`, multiple return."""
    testbed = NumpyArrayColumnTestBed(**config)
    col, array = testbed.col, testbed.data

    def func(x, bias=0):
        return {"mean": x.mean(axis=-1) + bias, "std": x.std(axis=-1) + bias}

    bias = 1 if use_kwargs else 0
    kwargs = {"bias": bias} if use_kwargs else {}

    result = col.map(func, batch_size=4, is_batched_fn=batched, **kwargs)
    assert isinstance(result, DataPanel)
    assert isinstance(result["std"], NumpyArrayColumn)
    assert isinstance(result["mean"], NumpyArrayColumn)
    np_test.assert_equal(len(result), len(array))
    assert (result["mean"] == array.mean(axis=-1) + bias).all()
    assert (result["std"] == array.std(axis=-1) + bias).all()


@NumpyArrayColumnTestBed.parametrize()
def test_set_item_1(config):
    testbed = NumpyArrayColumnTestBed(**config)
    col, array = testbed.col, testbed.data
    index = [0, 3]
    not_index = [i for i in range(col.shape[0]) if i not in index]
    col[index] = 0

    assert (col[not_index] == array[not_index]).all()
    assert (col[index] == 0).all()


@NumpyArrayColumnTestBed.parametrize()
def test_set_item_2(config):
    testbed = NumpyArrayColumnTestBed(**config)
    col, array = testbed.col, testbed.data
    index = 0
    not_index = [i for i in range(col.shape[0]) if i != index]
    col[index] = 0

    assert (col[not_index] == array[not_index]).all()
    assert (col[index] == 0).all()


@NumpyArrayColumnTestBed.parametrize(
    config={"num_dims": [1]},
    params={"batched": [True, False], "use_kwargs": [True, False]},
)
def test_filter_1(config, batched, use_kwargs):
    """multiple_dim=False."""
    testbed = NumpyArrayColumnTestBed(**config)
    col, array = testbed.col, testbed.data

    def func(x, thresh=20):
        return x > thresh

    thresh = 10 if use_kwargs else 20
    kwargs = {"thresh": thresh} if use_kwargs else {}

    result = col.filter(func, batch_size=4, is_batched_fn=batched, **kwargs)
    assert isinstance(result, NumpyArrayColumn)
    assert len(result) == (array > thresh).sum()


@NumpyArrayColumnTestBed.parametrize()
def test_pickle(config):
    # important for dataloader
    testbed = NumpyArrayColumnTestBed(**config)
    col, _ = testbed.col, testbed.data
    buf = pickle.dumps(col)
    new_col = pickle.loads(buf)

    assert isinstance(new_col, NumpyArrayColumn)
    assert (col == new_col).all()


@NumpyArrayColumnTestBed.parametrize()
def test_io(tmp_path, config):
    # uses the tmp_path fixture which will provide a
    # temporary directory unique to the test invocation,
    # important for dataloader
    testbed = NumpyArrayColumnTestBed(**config)
    col, _ = testbed.col, testbed.data

    path = os.path.join(tmp_path, "test")
    col.write(path)

    new_col = NumpyArrayColumn.read(path)

    assert isinstance(new_col, NumpyArrayColumn)
    assert (col == new_col).all()


@NumpyArrayColumnTestBed.parametrize()
def test_copy(config):
    testbed = NumpyArrayColumnTestBed(**config)
    col, _ = testbed.col, testbed.data
    col_copy = col.copy()

    assert isinstance(col_copy, NumpyArrayColumn)
    assert (col == col_copy).all()


@NumpyArrayColumnTestBed.parametrize()
def test_to_tensor(config):
    testbed = NumpyArrayColumnTestBed(**config)
    col, _ = testbed.col, testbed.data

    tensor = col.to_tensor()

    assert torch.is_tensor(tensor)
    assert (col == tensor.numpy()).all()
