"""Unittests for Datasets."""
import os
from itertools import product

import numpy as np
import pytest
import torch

from mosaic import NumpyArrayColumn
from mosaic.datapanel import DataPanel


def _get_datapanel(
    use_visible_rows: bool = False,
    use_visible_columns: bool = False,
):
    batch = {
        "a": np.arange(16),
        "b": list(np.arange(16)),
        "c": [{"a": 2}] * 16,
    }
    dp = DataPanel.from_batch(batch)

    visible_rows = [0, 4, 6, 11] if use_visible_rows else None
    if use_visible_rows:
        dp.visible_rows = visible_rows

    visible_columns = ["a", "b"] if use_visible_columns else None
    if use_visible_columns:
        dp.visible_columns = visible_columns

    return dp, visible_rows, visible_columns


def test_from_batch():
    # Build a dataset from a batch
    datapanel = DataPanel.from_batch(
        {
            "a": [1, 2, 3],
            "b": [True, False, True],
            "c": ["x", "y", "z"],
            "d": [{"e": 2}, {"e": 3}, {"e": 4}],
            "e": torch.ones(3),
            "f": np.ones(3),
        },
    )
    assert set(datapanel.column_names) == {"a", "b", "c", "d", "e", "f", "index"}
    assert len(datapanel) == 3


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns",
    product([True, False], [True, False]),
)
def test_map_1(use_visible_rows, use_visible_columns):
    """`map`, mixed datapanel, single return, `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows, use_visible_columns=use_visible_columns
    )

    def func(x):
        out = (x["a"] + np.array(x["b"])) * 2
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)
    result = dp.map(func, batch_size=4, batched=True)
    assert isinstance(result, NumpyArrayColumn)
    assert len(result) == len(visible_rows)
    assert (result == np.array(visible_rows) * 4).all()


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns",
    product([True, False], [True, False]),
)
def test_map_2(use_visible_rows, use_visible_columns):
    """`map`, mixed datapanel, return multiple, `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows, use_visible_columns=use_visible_columns
    )

    def func(x):
        out = {
            "x": (x["a"] + np.array(x["b"])) * 2,
            "y": np.array([x["c"][i]["a"] for i in range(len(x["c"]))]),
        }
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)
    result = dp.map(func, batch_size=4, batched=True)
    assert isinstance(result, DataPanel)
    assert len(result["x"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.ones(len(visible_rows)) * 2).all()


def test_update_1():
    """`update`, mixed datapanel, return single, new columns, `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=False, use_visible_columns=False
    )

    # mixed datapanel (i.e. has multiple colummn types)
    def func(x):
        out = {"x": (x["a"] + np.array(x["b"])) * 2}
        return out

    result = dp.update(func, batch_size=4, batched=True)
    assert isinstance(result, DataPanel)
    assert set(result.column_names) == set(["a", "b", "c", "x", "index"])
    assert len(result["x"]) == 16
    assert (result["x"] == np.arange(16) * 4).all()


def test_update_2():
    """`update`, mixed datapanel, return multiple, new columns,
    `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=False, use_visible_columns=False
    )

    def func(x):
        out = {
            "x": (x["a"] + np.array(x["b"])) * 2,
            "y": np.array([x["c"][i]["a"] for i in range(len(x["c"]))]),
        }
        return out

    result = dp.update(func, batch_size=4, batched=True)
    assert isinstance(result, DataPanel)
    assert set(result.column_names) == set(["a", "b", "c", "x", "y", "index"])
    assert len(result["x"]) == 16
    assert len(result["y"]) == 16
    assert (result["x"] == np.arange(16) * 4).all()
    assert (result["y"] == np.ones(16) * 2).all()


def test_update_3():
    """`update`, mixed datapanel, return multiple, replace existing column,
    `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=False, use_visible_columns=False
    )

    def func(x):
        out = {
            "a": (x["a"] + np.array(x["b"])) * 2,
            "y": np.array([x["c"][i]["a"] for i in range(len(x["c"]))]),
        }
        return out

    result = dp.update(func, batch_size=4, batched=True)
    assert isinstance(result, DataPanel)
    assert set(result.column_names) == set(["a", "b", "c", "y", "index"])
    assert len(result["a"]) == 16
    assert len(result["y"]) == 16
    assert (result["a"] == np.arange(16) * 4).all()
    assert (result["y"] == np.ones(16) * 2).all()


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_filter_1(use_visible_rows, use_visible_columns, batched):
    """`filter`, mixed datapanel."""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows, use_visible_columns=use_visible_columns
    )

    def func(x):
        return (x["a"] % 2) == 0

    if visible_rows is None:
        visible_rows = np.arange(16)
    result = dp.filter(func, batch_size=4, batched=batched)
    assert isinstance(result, DataPanel)
    assert len(result) == (np.array(visible_rows) % 2 == 0).sum()

    if visible_columns is not None:
        assert result.visible_columns == visible_columns
        assert result.all_columns == dp.all_columns


@pytest.mark.parametrize(
    "write_together,use_visible_rows, use_visible_columns",
    product([True, False], [True, False], [True, False]),
)
def test_io(tmp_path, write_together, use_visible_rows, use_visible_columns):
    """`map`, mixed datapanel, return multiple, `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows, use_visible_columns=use_visible_columns
    )
    path = os.path.join(tmp_path, "test")
    dp.write(path, write_together=write_together)

    new_dp = DataPanel.read(path)

    assert isinstance(new_dp, DataPanel)
    assert len(new_dp) == len(dp)
    assert len(new_dp["a"]) == len(dp["a"])
    assert len(new_dp["b"]) == len(dp["b"])
    assert len(new_dp["c"]) == len(dp["c"])

    assert (dp["a"] == new_dp["a"]).all()
    assert (dp["b"] == new_dp["b"]).all()

    assert dp.visible_columns == new_dp.visible_columns

    visible_rows = None if dp.visible_rows is None else set(dp.visible_rows)
    new_visible_rows = None if new_dp.visible_rows is None else set(dp.visible_rows)
    assert visible_rows == new_visible_rows


def test_repr_html_():
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=False, use_visible_columns=False
    )
    dp._repr_html_()
