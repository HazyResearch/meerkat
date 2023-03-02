"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest

from meerkat import concat
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.errors import ConcatError

from ...testbeds import AbstractColumnTestBed, MockDatapanel
from ...utils import product_parametrize
from ..columns.abstract import AbstractColumnTestBed, column_parametrize
from ..columns.deferred.test_deferred import DeferredColumnTestBed
from ..columns.deferred.test_image import ImageColumnTestBed
from ..columns.scalar.test_arrow import ArrowScalarColumnTestBed
from ..columns.scalar.test_pandas import PandasScalarColumnTestBed
from ..columns.tensor.test_numpy import NumPyTensorColumnTestBed
from ..columns.tensor.test_torch import TorchTensorColumnTestBed

# flake8: noqa: E501


@pytest.fixture(
    **column_parametrize(
        [
            NumPyTensorColumnTestBed,
            PandasScalarColumnTestBed,
            TorchTensorColumnTestBed,
            DeferredColumnTestBed,
            ArrowScalarColumnTestBed,
            ImageColumnTestBed,
        ]
    )
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@pytest.mark.parametrize(
    "use_visible_columns,n",
    product([True, False], [1, 2, 3]),
)
def test_dataframe_row_concat(use_visible_columns, n):
    mock_df = MockDatapanel(
        length=16,
        use_visible_columns=use_visible_columns,
    )

    out = concat([mock_df.df] * n, axis="rows")

    assert len(out) == len(mock_df.visible_rows) * n
    assert isinstance(out, DataFrame)
    assert set(out.columns) == set(mock_df.df.columns)
    assert (out["a"].data == np.concatenate([mock_df.visible_rows] * n)).all()
    assert out["b"].data == list(np.concatenate([mock_df.visible_rows] * n))


def test_dataframe_column_concat():
    mock_df = MockDatapanel(
        length=16,
        use_visible_columns=False,
    )

    out = concat([mock_df.df[["a"]], mock_df.df[["b"]]], axis="columns")

    assert len(out) == len(mock_df.visible_rows)
    assert isinstance(out, DataFrame)
    assert set(out.columns) == {"a", "b"}
    assert list(out["a"].data) == out["b"].data


@product_parametrize(params={"n": [1, 2, 3]})
def test_concat(column_testbed: AbstractColumnTestBed, n: int):
    col = column_testbed.col
    out = concat([col] * n)

    assert len(out) == len(col) * n
    assert isinstance(out, type(col))
    for i in range(n):
        assert out[i * len(col) : (i + 1) * len(col)].is_equal(col)


def test_concat_same_columns():
    a = DataFrame.from_batch({"a": [1, 2, 3]})
    b = DataFrame.from_batch({"a": [2, 3, 4]})

    out = concat([a, b], axis="columns", suffixes=["_a", "_b"])
    assert out.columns == ["a_a", "a_b"]
    assert list(out["a_a"].data) == [1, 2, 3]
    assert list(out["a_b"].data) == [2, 3, 4]


def test_concat_different_type():
    a = NumPyTensorColumn.from_array([1, 2, 3])
    b = ObjectColumn.from_list([1, 2, 3])
    with pytest.raises(ConcatError):
        concat([a, b])


def test_concat_unsupported_type():
    a = [1, 2, 3]
    b = [4, 5, 6]
    with pytest.raises(ConcatError):
        concat([a, b])


def test_concat_unsupported_axis():
    a = DataFrame.from_batch({"a": [1, 2, 3]})
    b = DataFrame.from_batch({"b": [1, 2, 3]})
    with pytest.raises(ConcatError):
        concat([a, b], axis="abc")


def test_concat_different_column_names():
    a = DataFrame.from_batch({"a": [1, 2, 3]})
    b = DataFrame.from_batch({"b": [1, 2, 3]})
    with pytest.raises(ConcatError):
        concat([a, b], axis="rows")


def test_concat_different_lengths():
    a = DataFrame.from_batch({"a": [1, 2, 3]})
    b = DataFrame.from_batch({"b": [1, 2, 3, 4]})

    with pytest.raises(ConcatError):
        concat([a, b], axis="columns")


def test_empty_concat():
    out = concat([])
    assert isinstance(out, DataFrame)
