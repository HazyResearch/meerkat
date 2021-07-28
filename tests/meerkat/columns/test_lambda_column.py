"""Unittests for LambdaColumn."""
from typing import Type

import numpy as np
import pytest

from meerkat import LambdaColumn, ListColumn, NumpyArrayColumn, TensorColumn

from ...testbeds import MockColumn, MockDatapanel


@pytest.mark.parametrize("col_type", [NumpyArrayColumn, TensorColumn, ListColumn])
def test_column_to_lambda(col_type: Type):
    testbed = MockColumn(col_type=col_type)
    col = testbed.col

    # Build a dataset from a batch
    lambda_col = col.to_lambda(lambda x: x + 1)

    assert isinstance(lambda_col, LambdaColumn)
    assert (lambda_col[:] == testbed.array[testbed.visible_rows] + 1).all()


@pytest.mark.parametrize(
    "use_visible_columns",
    [True, False],
)
def test_dp_to_lambda(use_visible_columns: bool):
    length = 16
    testbed = MockDatapanel(
        use_visible_columns=use_visible_columns,
        length=length,
    )
    dp = testbed.dp

    # Build a dataset from a batch
    lambda_col = dp.to_lambda(lambda x: x["a"] + 1)

    assert isinstance(lambda_col, LambdaColumn)
    assert (lambda_col[:].data == np.arange(length)[testbed.visible_rows] + 1).all()


@pytest.mark.parametrize(
    "col_type",
    [NumpyArrayColumn, TensorColumn, ListColumn],
)
def test_composed_lambda_columns(col_type: Type):
    testbed = MockColumn(col_type=col_type)

    # Build a dataset from a batch
    lambda_col = testbed.col.to_lambda(lambda x: x + 1)
    lambda_col = lambda_col.to_lambda(lambda x: x + 1)

    assert (lambda_col[:] == testbed.array[testbed.visible_rows] + 2).all()
