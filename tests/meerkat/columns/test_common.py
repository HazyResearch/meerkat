import os
import pickle
from functools import wraps
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meerkat import LambdaColumn, NumpyArrayColumn, PandasSeriesColumn, TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.errors import ImmutableError
from meerkat.ops.concat import concat

from ...utils import product_parametrize
from .abstract import AbstractColumnTestBed, column_parametrize
from .test_arrow_column import ArrowArrayColumnTestBed
from .test_lambda_column import LambdaColumnTestBed
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


@pytest.fixture(
    **column_parametrize(
        [
            NumpyArrayColumnTestBed,
            PandasSeriesColumnTestBed,
            TensorColumnTestBed,
            LambdaColumnTestBed,
            ArrowArrayColumnTestBed,
        ],
        single=True,
    ),
)
def single_column_testbed(request, tmpdir):
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


@product_parametrize(params={"index_type": [np.array, list, pd.Series]})
def test_set_item(column_testbed, index_type: type):
    MUTABLE_COLUMNS = (NumpyArrayColumn, TensorColumn, PandasSeriesColumn)

    col = column_testbed.col

    for index in [
        1,
        slice(2, 4, 1),
        (np.arange(len(col)) % 2).astype(bool),
        np.array([0, 3, 5, 6]),
    ]:
        col_index = index_type(index) if isinstance(index, np.ndarray) else index
        data_to_set = column_testbed.get_data_to_set(index)
        if isinstance(col, MUTABLE_COLUMNS):
            col[col_index] = data_to_set
            if isinstance(index, int):
                column_testbed.assert_data_equal(data_to_set, col.lz[col_index])
            else:
                column_testbed.assert_data_equal(data_to_set, col.lz[col_index].data)
        else:
            with pytest.raises(ImmutableError):
                col[col_index] = data_to_set


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


def test_copy(column_testbed: AbstractColumnTestBed):
    col, _ = column_testbed.col, column_testbed.data
    col_copy = col.copy()

    assert isinstance(col_copy, type(col))
    assert col.is_equal(col_copy)


def test_pickle(column_testbed):
    import dill as pickle  # needed so that it works with lambda functions

    # important for dataloader
    col = column_testbed.col
    buf = pickle.dumps(col)
    new_col = pickle.loads(buf)

    assert isinstance(new_col, type(col))

    if isinstance(new_col, LambdaColumn):
        # the lambda function isn't exactly the same after reading
        new_col.data.fn = col.fn
    assert col.is_equal(new_col)


def test_io(tmp_path, column_testbed: AbstractColumnTestBed):
    # uses the tmp_path fixture which will provide a
    # temporary directory unique to the test invocation,
    # important for dataloader
    col, _ = column_testbed.col, column_testbed.data

    path = os.path.join(tmp_path, "test")
    col.write(path)

    new_col = type(col).read(path)

    assert isinstance(new_col, type(col))

    if isinstance(new_col, LambdaColumn):
        # the lambda function isn't exactly the same after reading
        new_col.data.fn = col.fn

    assert col.is_equal(new_col)


def test_head(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    length = 10
    result = testbed.col.head(length)
    assert len(result) == length
    assert result.is_equal(testbed.col.lz[:length])


def test_tail(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    length = 10
    result = testbed.col.tail(length)
    assert len(result) == length
    assert result.is_equal(testbed.col.lz[-length:])


def test_repr_html(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    testbed.col._repr_html_()


def test_str(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    result = str(testbed.col)
    assert isinstance(result, str)


def test_repr(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    result = repr(testbed.col)
    assert isinstance(result, str)


def test_streamlit(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    testbed.col.streamlit()


def test_repr_pandas(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    series, _ = testbed.col._repr_pandas_()
    assert isinstance(series, pd.Series)


def test_to_pandas(single_column_testbed: AbstractColumnTestBed):
    testbed = single_column_testbed
    series = testbed.col.to_pandas()
    assert isinstance(series, pd.Series)
