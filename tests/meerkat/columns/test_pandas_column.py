"""Unittests for NumpyColumn."""
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meerkat import NumpyArrayColumn, PandasSeriesColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel

from ...testbeds import MockAnyColumn, MockColumn, MockStrColumn


def test_str_accessor():
    testbed = MockStrColumn(col_type=PandasSeriesColumn)
    col = testbed.col

    upper_col = col.str.upper()
    assert isinstance(upper_col, PandasSeriesColumn)
    assert (
        upper_col.values == np.array([f"ROW_{idx}" for idx in testbed.visible_rows])
    ).all()


def test_dt_accessor():
    testbed = MockAnyColumn(
        data=[f"01/{idx+1}/2001" for idx in range(16)],
        col_type=PandasSeriesColumn,
    )
    col = testbed.col
    col = pd.to_datetime(col)
    day_col = col.dt.day
    assert isinstance(day_col, PandasSeriesColumn)
    assert (day_col.values == np.array(testbed.visible_rows) + 1).all()


def test_cat_accessor():
    categories = ["a", "b", "c", "d"]
    testbed = MockAnyColumn(
        data=categories * 4,
        col_type=PandasSeriesColumn,
        use_visible_rows=False,
    )
    col = testbed.col.astype("category")

    assert (np.array(categories) == col.cat.categories.values).all()


@pytest.mark.parametrize(
    "dtype,index_type",
    product(
        ["float", "int", "str"], [NumpyArrayColumn, PandasSeriesColumn, TensorColumn]
    ),
)
def test_getitem(dtype, index_type):
    """`map`, single return,"""
    if dtype == "str":
        testbed = MockStrColumn(col_type=PandasSeriesColumn)
    else:
        testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)

    col = testbed.col

    assert testbed.array[testbed.visible_rows[1]] == col[1]

    assert (testbed.array[testbed.visible_rows[2:4]] == col[2:4].values).all()

    bool_index = (np.arange(len(col)) % 2).astype(bool)
    bool_index_col = index_type(bool_index)
    assert (testbed.array[bool_index] == col[bool_index_col].values).all()


@pytest.mark.parametrize(
    "dtype",
    ["float", "int", "str"],
)
def test_ops(dtype):
    """`map`, single return,"""
    col = PandasSeriesColumn(["a", "b", "c", "d"])
    col == "a"
    if dtype == "str":
        testbed = MockStrColumn(col_type=PandasSeriesColumn)
    else:
        testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)

    col = testbed.col

    assert testbed.array[testbed.visible_rows[1]] == col[1]

    assert (testbed.array[testbed.visible_rows[2:4]] == col[2:4].values).all()


@pytest.mark.parametrize(
    "dtype,batched,use_kwargs",
    product(["float", "int"], [True, False], [True, False]),
)
def test_map_return_single(dtype, batched, use_kwargs):
    """`map`, single return,"""
    testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)
    col, array = testbed.col, testbed.array

    def func(x, bias=1):
        out = x + bias
        return out

    bias = 2 if use_kwargs else 1
    kwargs = {"bias": bias} if use_kwargs else {}

    result = col.map(
        func,
        batch_size=2,
        is_batched_fn=batched,
        output_type=PandasSeriesColumn,
        **kwargs,
    )
    assert isinstance(result, PandasSeriesColumn)
    assert len(result) == len(array[testbed.visible_rows])
    assert (result.values == array[testbed.visible_rows] + bias).all()


@pytest.mark.parametrize(
    "dtype, batched, use_kwargs",
    product(["float", "int"], [True, False], [True, False]),
)
def test_map_return_multiple(dtype, batched, use_kwargs):
    """`map`, multiple return."""
    testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)
    col, array = testbed.col, testbed.array

    def func(x, bias=1):
        return {"a": x + bias, "b": x - bias}

    bias = 2 if use_kwargs else 1
    kwargs = {"bias": bias} if use_kwargs else {}

    result = col.map(
        func,
        batch_size=2,
        is_batched_fn=batched,
        output_type=PandasSeriesColumn,
        **kwargs,
    )
    assert isinstance(result, DataPanel)
    assert len(result) == len(array[testbed.visible_rows])

    assert isinstance(result["a"], PandasSeriesColumn)
    assert (result["a"].values == array[testbed.visible_rows] + bias).all()

    assert isinstance(result["b"], PandasSeriesColumn)
    assert (result["b"].values == array[testbed.visible_rows] - bias).all()


@pytest.mark.parametrize(
    "dtype",
    ["float", "int", "str"],
)
def test_set_item_1(dtype):

    if dtype == "str":
        testbed = MockStrColumn(col_type=PandasSeriesColumn)
    else:
        testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)

    col = testbed.col

    index = [0, 3]
    not_index = [i for i in range(col.shape[0]) if i not in index]
    col[index] = 0
    assert (col[not_index] == testbed.array[testbed.visible_rows[not_index]]).all()
    assert (col[index] == 0).all()


@pytest.mark.parametrize(
    "dtype",
    ["float", "int", "str"],
)
def test_set_item_2(dtype):
    if dtype == "str":
        testbed = MockStrColumn(col_type=PandasSeriesColumn)
    else:
        testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)

    col = testbed.col

    index = 1
    not_index = [i for i in range(col.shape[0]) if i != index]
    col[index] = 0
    assert (
        col[not_index].values == testbed.array[testbed.visible_rows[not_index]]
    ).all()
    assert col[index] == 0


@pytest.mark.parametrize(
    "dtype,batched,use_kwargs",
    product(["float", "int"], [True, False], [True, False]),
)
def test_filter_1(dtype, batched, use_kwargs):
    """multiple_dim=False."""
    testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)
    col, array = testbed.col, testbed.array

    def func(x, thresh=10):
        return x > thresh

    thresh = 5 if use_kwargs else 10
    kwargs = {"thresh": thresh} if use_kwargs else {}

    result = col.filter(func, batch_size=4, is_batched_fn=batched, **kwargs)
    assert isinstance(result, PandasSeriesColumn)
    assert len(result) == (array[testbed.visible_rows] > thresh).sum()


@pytest.mark.parametrize(
    "multiple_dim, dtype,use_visible_rows",
    product([True, False], ["float", "int"], [True, False]),
)
def test_pickle(multiple_dim, dtype, use_visible_rows):
    # important for dataloader
    testbed = MockColumn(
        dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
    )
    col = testbed.col
    buf = pickle.dumps(col)
    new_col = pickle.loads(buf)

    assert isinstance(new_col, PandasSeriesColumn)
    assert (col.values == new_col.values).all()


@pytest.mark.parametrize(
    "dtype",
    ["float", "int", "str"],
)
def test_io(
    tmp_path,
    dtype,
):
    # uses the tmp_path fixture which will provide a
    # temporary directory unique to the test invocation,
    # important for dataloader
    if dtype == "str":
        testbed = MockStrColumn(col_type=PandasSeriesColumn)
    else:
        testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)
    col = testbed.col
    path = os.path.join(tmp_path, "test")
    col.write(path)

    new_col = PandasSeriesColumn.read(path)

    assert isinstance(new_col, PandasSeriesColumn)
    assert (col == new_col).all()


@pytest.mark.parametrize(
    "dtype",
    ["float", "int", "str"],
)
def test_copy(dtype):
    if dtype == "str":
        testbed = MockStrColumn(col_type=PandasSeriesColumn)
    else:
        testbed = MockColumn(dtype=dtype, col_type=PandasSeriesColumn)
    col = testbed.col
    col_copy = col.copy()

    assert isinstance(col_copy, PandasSeriesColumn)
    assert (col == col_copy).all()
