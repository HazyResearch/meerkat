"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest

from mosaic import concat
from mosaic.columns.image_column import ImageColumn
from mosaic.columns.list_column import ListColumn
from mosaic.columns.numpy_column import NumpyArrayColumn
from mosaic.columns.tensor_column import TensorColumn
from mosaic.datapanel import DataPanel
from mosaic.errors import ConcatError

from ...testbeds import MockColumn, MockDatapanel, MockImageColumn


@pytest.mark.parametrize(
    "use_visible_rows,col_type,n",
    product([True, False], [ListColumn, NumpyArrayColumn, TensorColumn], [1, 2, 3]),
)
def test_column_concat(use_visible_rows, col_type, n):
    mock_col = MockColumn(use_visible_rows=use_visible_rows, col_type=col_type)
    out = concat([mock_col.col] * n)

    assert len(out) == len(mock_col.visible_rows) * n
    assert isinstance(out, col_type)
    assert list(out.data) == list(np.concatenate([mock_col.visible_rows] * n))


@pytest.mark.parametrize(
    "use_visible_rows,use_visible_columns,n",
    product([True, False], [True, False], [1, 2, 3]),
)
def test_datapanel_row_concat(use_visible_rows, use_visible_columns, n):

    mock_dp = MockDatapanel(
        length=16,
        use_visible_rows=use_visible_rows,
        use_visible_columns=use_visible_columns,
    )

    out = concat([mock_dp.dp] * n, axis="rows")

    assert len(out) == len(mock_dp.visible_rows) * n
    assert isinstance(out, DataPanel)
    assert set(out.visible_columns) == set(mock_dp.visible_columns)
    assert (out["a"].data == np.concatenate([mock_dp.visible_rows] * n)).all()
    assert out["b"].data == list(np.concatenate([mock_dp.visible_rows] * n))


@pytest.mark.parametrize(
    "use_visible_rows",
    product([True, False]),
)
def test_datapanel_column_concat(use_visible_rows):

    mock_dp = MockDatapanel(
        length=16,
        use_visible_rows=use_visible_rows,
        use_visible_columns=False,
    )

    out = concat([mock_dp.dp[["a"]], mock_dp.dp[["b"]]], axis="columns")

    assert len(out) == len(mock_dp.visible_rows)
    assert isinstance(out, DataPanel)
    assert set(out.visible_columns) == {"a", "b", "index"}
    assert list(out["a"].data) == out["b"].data


@pytest.mark.parametrize(
    "use_visible_rows",
    product([True, False]),
)
def test_image_column(use_visible_rows, tmpdir):

    mock = MockImageColumn(length=16, tmpdir=tmpdir)

    out = concat([mock.col.lz[:5], mock.col.lz[5:10]])

    assert len(out) == 10
    assert isinstance(out, ImageColumn)
    assert [str(cell.filepath) for cell in out.lz] == mock.image_paths[:10]


def test_concat_different_type():
    a = NumpyArrayColumn.from_array([1, 2, 3])
    b = ListColumn.from_list([1, 2, 3])
    with pytest.raises(ConcatError):
        concat([a, b])


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


def test_concat_same_columns():
    a = DataPanel.from_batch({"a": [1, 2, 3]})
    b = DataPanel.from_batch({"a": [1, 2, 3]})

    with pytest.raises(ConcatError):
        concat([a, b], axis="columns")


def test_empty_concat():
    out = concat([])
    assert isinstance(out, DataPanel)
