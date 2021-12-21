"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest

from meerkat import concat
from meerkat.columns.list_column import ListColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.datapanel import DataPanel
from meerkat.errors import ConcatError
from meerkat.ml.prediction_column import ClassificationOutputColumn

from ...testbeds import MockDatapanel


@pytest.mark.parametrize(
    "use_visible_columns,n",
    product([True, False], [1, 2, 3]),
)
def test_datapanel_row_concat(use_visible_columns, n):

    mock_dp = MockDatapanel(
        length=16,
        use_visible_columns=use_visible_columns,
    )

    out = concat([mock_dp.dp] * n, axis="rows")

    assert len(out) == len(mock_dp.visible_rows) * n
    assert isinstance(out, DataPanel)
    assert set(out.columns) == set(mock_dp.dp.columns)
    assert (out["a"].data == np.concatenate([mock_dp.visible_rows] * n)).all()
    assert out["b"].data == list(np.concatenate([mock_dp.visible_rows] * n))


def test_datapanel_column_concat():

    mock_dp = MockDatapanel(
        length=16,
        use_visible_columns=False,
    )

    out = concat([mock_dp.dp[["a"]], mock_dp.dp[["b"]]], axis="columns")

    assert len(out) == len(mock_dp.visible_rows)
    assert isinstance(out, DataPanel)
    assert set(out.columns) == {"a", "b"}
    assert list(out["a"].data) == out["b"].data


def test_concat_same_columns():
    a = DataPanel.from_batch({"a": [1, 2, 3]})
    b = DataPanel.from_batch({"a": [2, 3, 4]})

    out = concat([a, b], axis="columns", suffixes=["_a", "_b"])
    assert out.columns == ["a_a", "a_b"]
    assert list(out["a_a"].data) == [1, 2, 3]
    assert list(out["a_b"].data) == [2, 3, 4]


def test_concat_different_type():
    a = NumpyArrayColumn.from_array([1, 2, 3])
    b = ListColumn.from_list([1, 2, 3])
    with pytest.raises(ConcatError):
        concat([a, b])


def test_concat_unsupported_type():
    a = [1, 2, 3]
    b = [4, 5, 6]
    with pytest.raises(ConcatError):
        concat([a, b])


def test_concat_unsupported_axis():
    a = DataPanel.from_batch({"a": [1, 2, 3]})
    b = DataPanel.from_batch({"b": [1, 2, 3]})
    with pytest.raises(ConcatError):
        concat([a, b], axis="abc")


def test_concat_different_column_names():
    a = DataPanel.from_batch({"a": [1, 2, 3]})
    b = DataPanel.from_batch({"b": [1, 2, 3]})
    with pytest.raises(ConcatError):
        concat([a, b], axis="rows")


def test_concat_different_lengths():
    a = DataPanel.from_batch({"a": [1, 2, 3]})
    b = DataPanel.from_batch({"b": [1, 2, 3, 4]})

    with pytest.raises(ConcatError):
        concat([a, b], axis="columns")


def test_concat_maintains_subclass():
    col = ClassificationOutputColumn(logits=[0, 1, 0, 1], num_classes=2)
    out = concat([col, col])
    assert isinstance(out, ClassificationOutputColumn)


def test_empty_concat():
    out = concat([])
    assert isinstance(out, DataPanel)
