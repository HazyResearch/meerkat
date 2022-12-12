import os

import numpy as np
import pandas as pd
import pytest

from meerkat import (
    DeferredColumn,
    TorchTensorColumn,
    ScalarColumn,
    NumPyTensorColumn,
)
from meerkat.errors import ImmutableError

from ...utils import product_parametrize
from .abstract import AbstractColumnTestBed, column_parametrize
from .scalar.test_arrow import ArrowScalarColumnTestBed
from .deferred.test_audio import AudioColumnTestBed
from .deferred.test_file import FileColumnTestBed
from .deferred.test_image import ImageColumnTestBed
from .deferred.test_deferred import DeferredColumnTestBed
from .tensor.test_numpy import NumPyTensorColumnTestBed
from .scalar.test_pandas import PandasScalarColumnTestBed
from .tensor.test_torch import TorchTensorColumnTestBed


@pytest.fixture(
    **column_parametrize(
        [
            NumPyTensorColumnTestBed,
            PandasScalarColumnTestBed,
            TorchTensorColumnTestBed,
            # DeferredColumnTestBed,
            ArrowScalarColumnTestBed,
            # FileColumnTestBed,
            # ImageColumnTestBed,
            # AudioColumnTestBed,
        ]
    )
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@pytest.fixture(
    **column_parametrize(
        [
            NumPyTensorColumnTestBed,
            PandasScalarColumnTestBed,
            TorchTensorColumnTestBed,
            DeferredColumnTestBed,
            ArrowScalarColumnTestBed,
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
    MUTABLE_COLUMNS = (TorchTensorColumn, TorchTensorColumn, ScalarColumn)

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

    if isinstance(new_col, DeferredColumn):
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

    if isinstance(new_col, DeferredColumn):
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
