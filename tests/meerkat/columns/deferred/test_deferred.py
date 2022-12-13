"""Unittests for LambdaColumn."""
from typing import Type

import numpy as np
import pytest

import meerkat as mk
from meerkat import DeferredColumn, ObjectColumn, NumPyTensorColumn
from meerkat.errors import ConcatWarning

from ....testbeds import MockColumn, MockDatapanel
from ..abstract import AbstractColumnTestBed, column_parametrize


class DeferredColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "batched": [True, False],
        "from_df": [True, False],
        "multiple_outputs": [True, False],
    }

    marks = pytest.mark.lambda_col

    def __init__(
        self,
        batched: bool,
        from_df: bool,
        multiple_outputs: bool,
        length: int = 16,
        seed: int = 123,
        tmpdir: str = None,
    ):
        to_lambda_kwargs = {
            "is_batched_fn": batched,
            "batch_size": 4 if batched else 1,
        }

        np.random.seed(seed)
        array = np.random.random(length) * 10
        self.col = mk.NumPyTensorColumn(array).defer(
            function=lambda x: x + 2, **to_lambda_kwargs
        )
        self.data = array + 2

    def get_map_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        kwarg: int = 0,
        salt: int = 1,
    ):
        if materialize:
            return {
                "fn": lambda x, k=0: x + salt + k,
                "expected_result": NumPyTensorColumn.from_array(
                    self.data + salt + kwarg
                ),
            }
        else:
            if batched:
                return {
                    "fn": lambda x, k=0: np.array([cell.get() for cell in x.lz])
                    + salt
                    + k,
                    "expected_result": NumPyTensorColumn.from_array(
                        self.data + salt + kwarg
                    ),
                }
            else:
                return {
                    "fn": lambda x, k=0: x.get() + salt + k,
                    "expected_result": NumPyTensorColumn.from_array(
                        self.data + salt + kwarg
                    ),
                }

    def get_data(self, index, materialize: bool = True):
        if materialize:
            return self.data[index]
        else:
            raise NotImplementedError()

    def get_data_to_set(self, index):
        return 0

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        if isinstance(data1, np.ndarray):
            assert (data1 == data2).all()
        else:
            assert data1 == data2


@pytest.fixture(**column_parametrize([DeferredColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@pytest.mark.parametrize("col_type", [NumPyTensorColumn, NumPyTensorColumn, ObjectColumn])
def test_column_to_lambda(col_type: Type):
    testbed = MockColumn(col_type=col_type)
    col = testbed.col

    # Build a dataset from a batch
    lambda_col = col.to_lambda(lambda x: x + 1)

    assert isinstance(lambda_col, DeferredColumn)
    assert (lambda_col() == testbed.array[testbed.visible_rows] + 1).all()


@pytest.mark.parametrize(
    "use_visible_columns",
    [True, False],
)
def test_df_to_lambda(use_visible_columns: bool):
    length = 16
    testbed = MockDatapanel(
        use_visible_columns=use_visible_columns,
        length=length,
    )
    df = testbed.df

    # Build a dataset from a batch
    lambda_col = df.defer(lambda x: x["a"] + 1)

    assert isinstance(lambda_col, DeferredColumn)
    assert (lambda_col().data == np.arange(length)[testbed.visible_rows] + 1).all()


@pytest.mark.parametrize(
    "col_type",
    [NumPyTensorColumn, NumPyTensorColumn, ObjectColumn],
)
def test_composed_lambda_columns(col_type: Type):
    testbed = MockColumn(col_type=col_type)

    # Build a dataset from a batch
    lambda_col = testbed.col.to_lambda(lambda x: x + 1)
    lambda_col = lambda_col.to_lambda(lambda x: x + 1)

    assert (lambda_col() == testbed.array[testbed.visible_rows] + 2).all()


def test_df_concat():
    length = 16
    testbed = MockDatapanel(length=length)
    df = testbed.df

    def fn(x):
        return x["a"] + 1

    col_a = df.defer(fn)
    col_b = df.defer(fn)

    out = mk.concat([col_a, col_b])

    assert isinstance(out, DeferredColumn)
    assert (out().data == np.concatenate([np.arange(length) + 1] * 2)).all()

    col_a = df.defer(fn)
    col_b = df.defer(lambda x: x["a"])
    with pytest.warns(ConcatWarning):
        out = mk.concat([col_a, col_b])


@pytest.mark.parametrize("col_type", [NumPyTensorColumn, ObjectColumn])
def test_col_concat(col_type):
    testbed = MockColumn(col_type=col_type)
    col = testbed.col
    length = len(col)

    def fn(x):
        return x + 1

    col_a = col.to_lambda(fn)
    col_b = col.to_lambda(fn)

    out = mk.concat([col_a, col_b])

    assert isinstance(out, DeferredColumn)
    assert (out().data == np.concatenate([np.arange(length) + 1] * 2)).all()

    col_a = col.to_lambda(fn)
    col_b = col.to_lambda(lambda x: x)
    with pytest.warns(ConcatWarning):
        out = mk.concat([col_a, col_b])
