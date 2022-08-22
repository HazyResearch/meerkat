import os
import pickle
from functools import wraps
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meerkat.datapanel import DataPanel
from meerkat.ops.concat import concat

from ...utils import product_parametrize
from .abstract import AbstractColumnTestBed, column_parametrize
from .test_arrow_column import ArrowArrayColumnTestBed
from .test_lambda_column import LambdaColumn, LambdaColumnTestBed
from .test_numpy_column import NumpyArrayColumnTestBed
from .test_pandas_column import PandasSeriesColumnTestBed
from .test_tensor_column import TensorColumnTestBed


@pytest.fixture(
    **column_parametrize(
        [
            NumpyArrayColumnTestBed,
            PandasSeriesColumnTestBed,
            TensorColumnTestBed,
            LambdaColumnTestBed,
            ArrowArrayColumnTestBed,
        ]
    )
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@product_parametrize(params={"index_type": [np.array, list]})
def test_getitem(column_testbed, index_type: type):
    col = column_testbed.col

    column_testbed.assert_data_equal(column_testbed.get_data(1), col[1])

    for index in [
        slice(2, 4, 1),
        (np.arange(len(col)) % 2).astype(bool),
        np.array([0, 3, 5, 6]),
    ]:
        col_index = index_type(index) if not isinstance(index, slice) else index
        data = column_testbed.get_data(index)
        result = col[col_index]
        column_testbed.assert_data_equal(data, result.data)

        if type(result) == type(col):
            # if the getitem returns a column of the same type, enforce that all the
            # attributes were cloned over appropriately. We don't want to check this
            # for columns that return columns of different type from getitem
            # (e.g. LambdaColumn)
            assert col._clone(data=data).is_equal(result)


# def test_set_item(self, testbed, index_type: type = np.array):
#     col = testbed.col

#     for index in [
#         1,
#         slice(2, 4, 1),
#         (np.arange(len(col)) % 2).astype(bool),
#         np.array([0, 3, 5, 6]),
#     ]:
#         col_index = index_type(index) if isinstance(index, np.ndarray) else index
#         data_to_set = self._get_data_to_set(testbed, index)
#         col[col_index] = data_to_set
#         if isinstance(index, int):
#             testbed.assert_data_equal(data_to_set, col.lz[col_index])
#         else:
#             testbed.assert_data_equal(data_to_set, col.lz[col_index].data)


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_single(
    column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, LambdaColumn) or materialize):
        # skip columns for which materialize has no effect
        return

    col = column_testbed.col

    map_spec = column_testbed.get_map_spec(batched=batched, materialize=materialize)

    def func(x):
        out = map_spec["fn"](x)
        return out

    result = col.map(
        func,
        batch_size=4,
        is_batched_fn=batched,
        materialize=materialize,
        output_type=map_spec.get("output_type", None),
    )
    assert result.is_equal(map_spec["expected_result"])


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_single_w_kwarg(
    column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, LambdaColumn) or materialize):
        # skip columns for which materialize has no effect
        return

    col = column_testbed.col
    kwarg = 2
    map_spec = column_testbed.get_map_spec(
        batched=batched, materialize=materialize, kwarg=kwarg
    )

    def func(x, k=0):
        out = map_spec["fn"](x, k=k)
        return out

    result = col.map(
        func,
        batch_size=4,
        is_batched_fn=batched,
        materialize=materialize,
        output_type=map_spec.get("output_type", None),
        k=kwarg,
    )
    assert result.is_equal(map_spec["expected_result"])


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_multiple(
    column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, LambdaColumn) or materialize):
        # skip columns for which materialize has no effect
        return

    col = column_testbed.col
    map_specs = {
        "map1": column_testbed.get_map_spec(
            batched=batched, materialize=materialize, salt=1
        ),
        "map2": column_testbed.get_map_spec(
            batched=batched, materialize=materialize, salt=2
        ),
    }

    def func(x):
        out = {key: map_spec["fn"](x) for key, map_spec in map_specs.items()}
        return out

    result = col.map(
        func,
        batch_size=4,
        is_batched_fn=batched,
        materialize=materialize,
        output_type=list(map_specs.values())[0].get("output_type", None),
    )
    assert isinstance(result, DataPanel)
    for key, map_spec in map_specs.items():
        assert result[key].is_equal(map_spec["expected_result"])
