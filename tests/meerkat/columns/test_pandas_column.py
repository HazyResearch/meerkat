"""Unittests for NumpyColumn."""
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.datapanel import DataPanel

from ...testbeds import MockAnyColumn, MockColumn, MockStrColumn


@pytest.mark.parametrize(
    "use_visible_rows",
    [True, False],
)
def test_str_accessor(use_visible_rows):
    testbed = MockStrColumn(
        col_type=PandasSeriesColumn, use_visible_rows=use_visible_rows
    )
    col = testbed.col

    upper_col = col.str.upper()
    assert isinstance(upper_col, PandasSeriesColumn)
    assert (
        upper_col.values == np.array([f"ROW_{idx}" for idx in testbed.visible_rows])
    ).all()


@pytest.mark.parametrize(
    "use_visible_rows",
    [True, False],
)
def test_dt_accessor(use_visible_rows):
    testbed = MockAnyColumn(
        data=[f"01/{idx+1}/2001" for idx in range(16)],
        col_type=PandasSeriesColumn,
        use_visible_rows=use_visible_rows,
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
    "dtype,use_visible_rows",
    product(["float", "int", "str"], [True, False]),
)
def test_getitem(dtype, use_visible_rows):
    """`map`, single return,"""
    if dtype == "str":
        testbed = MockStrColumn(
            use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    else:
        testbed = MockColumn(
            dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )

    col = testbed.col

    assert testbed.array[testbed.visible_rows[1]] == col[1]

    assert (testbed.array[testbed.visible_rows[2:4]] == col[2:4].values).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows",
    product(["float", "int", "str"], [True, False]),
)
def test_ops(dtype, use_visible_rows):
    """`map`, single return,"""
    col = PandasSeriesColumn(["a", "b", "c", "d"])
    col == "a"
    if dtype == "str":
        testbed = MockStrColumn(
            use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    else:
        testbed = MockColumn(
            dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )

    col = testbed.col

    assert testbed.array[testbed.visible_rows[1]] == col[1]

    assert (testbed.array[testbed.visible_rows[2:4]] == col[2:4].values).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows,batched",
    product(["float", "int"], [True, False], [True, False]),
)
def test_map_return_single(dtype, use_visible_rows, batched):
    """`map`, single return,"""
    testbed = MockColumn(
        dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
    )
    col, array = testbed.col, testbed.array

    def func(x):
        out = x + 1
        return out

    result = col.map(
        func, batch_size=2, is_batched_fn=batched, output_type=PandasSeriesColumn
    )
    assert isinstance(result, PandasSeriesColumn)
    assert len(result) == len(array[testbed.visible_rows])
    assert (result.values == array[testbed.visible_rows] + 1).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows, batched",
    product(["float", "int"], [True, False], [True, False]),
)
def test_map_return_multiple(dtype, use_visible_rows, batched):
    """`map`, multiple return."""
    testbed = MockColumn(
        dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
    )
    col, array = testbed.col, testbed.array

    def func(x):
        return {"a": x + 1, "b": x - 1}

    result = col.map(
        func, batch_size=2, is_batched_fn=batched, output_type=PandasSeriesColumn
    )
    assert isinstance(result, DataPanel)
    assert len(result) == len(array[testbed.visible_rows])

    assert isinstance(result["a"], PandasSeriesColumn)
    assert (result["a"].values == array[testbed.visible_rows] + 1).all()

    assert isinstance(result["b"], PandasSeriesColumn)
    assert (result["b"].values == array[testbed.visible_rows] - 1).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows",
    product(["float", "int", "str"], [True, False]),
)
def test_set_item_1(dtype, use_visible_rows):

    if dtype == "str":
        testbed = MockStrColumn(
            use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    else:
        testbed = MockColumn(
            dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )

    col = testbed.col

    index = [0, 3]
    not_index = [i for i in range(col.shape[0]) if i not in index]
    col[index] = 0
    assert (col[not_index] == testbed.array[testbed.visible_rows[not_index]]).all()
    assert (col[index] == 0).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows",
    product(["float", "int", "str"], [True, False]),
)
def test_set_item_2(dtype, use_visible_rows):
    if dtype == "str":
        testbed = MockStrColumn(
            use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    else:
        testbed = MockColumn(
            dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )

    col = testbed.col

    index = 1
    not_index = [i for i in range(col.shape[0]) if i != index]
    col[index] = 0
    assert (
        col[not_index].values == testbed.array[testbed.visible_rows[not_index]]
    ).all()
    assert col[index] == 0


@pytest.mark.parametrize(
    "use_visible_rows,dtype,batched",
    product([True, False], ["float", "int"], [True, False]),
)
def test_filter_1(use_visible_rows, dtype, batched):
    """multiple_dim=False."""
    testbed = MockColumn(
        dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
    )
    col, array = testbed.col, testbed.array

    def func(x):
        return x > 10

    result = col.filter(func, batch_size=4, is_batched_fn=batched)
    assert isinstance(result, PandasSeriesColumn)
    assert len(result) == (array[testbed.visible_rows] > 10).sum()


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
    "dtype,use_visible_rows",
    product(["float", "int", "str"], [True, False]),
)
def test_io(tmp_path, dtype, use_visible_rows):
    # uses the tmp_path fixture which will provide a
    # temporary directory unique to the test invocation,
    # important for dataloader
    if dtype == "str":
        testbed = MockStrColumn(
            use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    else:
        testbed = MockColumn(
            dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    col = testbed.col
    path = os.path.join(tmp_path, "test")
    col.write(path)

    new_col = PandasSeriesColumn.read(path)

    assert isinstance(new_col, PandasSeriesColumn)
    assert (col == new_col).all()


@pytest.mark.parametrize(
    "dtype,use_visible_rows",
    product(["float", "int", "str"], [True, False]),
)
def test_copy(dtype, use_visible_rows):
    if dtype == "str":
        testbed = MockStrColumn(
            use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    else:
        testbed = MockColumn(
            dtype=dtype, use_visible_rows=use_visible_rows, col_type=PandasSeriesColumn
        )
    col = testbed.col
    col_copy = col.copy()

    assert isinstance(col_copy, PandasSeriesColumn)
    assert (col == col_copy).all()
