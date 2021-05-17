"""Unittests for Datasets."""
import os
from itertools import product

import numpy as np
import pytest
import torch

from mosaic import ImagePath, NumpyArrayColumn
from mosaic.columns.image_column import ImageColumn
from mosaic.columns.list_column import ListColumn
from mosaic.datapanel import DataPanel

from ..testbeds import MockDatapanel


def _get_datapanel(*args, **kwargs):
    test_bed = MockDatapanel(length=16, *args, **kwargs)
    return test_bed.dp, test_bed.visible_rows, test_bed.visible_columns


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
    "use_visible_rows",
    product([True, False]),
)
def test_lz_getitem(tmpdir, use_visible_rows):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        use_visible_rows=use_visible_rows,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    visible_rows = (
        np.arange(length)
        if test_bed.visible_rows is None
        else np.array(test_bed.visible_rows)
    )

    # int index => single row (dict)
    index = 2
    row = dp.lz[index]
    assert isinstance(row["img"], ImagePath)
    assert str(row["img"].filepath) == test_bed.img_col.image_paths[visible_rows[index]]
    assert row["a"] == visible_rows[index]
    assert row["b"] == visible_rows[index]

    # slice index => multiple row selection (DataPanel)
    # tuple or list index => multiple row selection (DataPanel)
    # np.array indeex => multiple row selection (DataPanel)
    for rows, indices in (
        (dp.lz[1:3], visible_rows[1:3]),
        (dp.lz[[0, 2]], visible_rows[[0, 2]]),
        (dp.lz[np.array((0,))], visible_rows[np.array((0,))]),
        (dp.lz[np.array((1, 1))], visible_rows[np.array((1, 1))]),
    ):
        assert isinstance(rows["img"], ImageColumn)
        assert list(map(lambda x: str(x.filepath), rows["img"].data)) == [
            test_bed.img_col.image_paths[i] for i in indices
        ]
        assert (rows["a"].data == indices).all()
        assert (rows["b"].data == indices).all()


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns, num_workers",
    product([True, False], [True, False], [0, 2]),
)
def test_map_1(use_visible_rows, use_visible_columns, num_workers):
    """`map`, mixed datapanel, single return, `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows, use_visible_columns=use_visible_columns
    )

    def func(x):
        out = (x["a"] + np.array(x["b"])) * 2
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)
    result = dp.map(func, batch_size=4, batched=True, num_workers=num_workers)
    assert isinstance(result, NumpyArrayColumn)
    assert len(result) == len(visible_rows)
    assert (result == np.array(visible_rows) * 4).all()


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns, num_workers",
    product([True, False], [True, False], [0, 2]),
)
def test_map_2(use_visible_rows, use_visible_columns, num_workers):
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
    result = dp.map(func, batch_size=4, batched=True, num_workers=num_workers)
    assert isinstance(result, DataPanel)
    assert len(result["x"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.ones(len(visible_rows)) * 2).all()


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_update_1(use_visible_rows, use_visible_columns, batched):
    """`update`, mixed datapanel, return single, new columns."""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows,
        use_visible_columns=use_visible_columns,
    )

    # mixed datapanel (i.e. has multiple colummn types)
    def func(x):
        out = {"x": (x["a"] + np.array(x["b"])) * 2}
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)

    result = dp.update(func, batch_size=4, batched=batched, num_workers=0)
    assert isinstance(result, DataPanel)
    assert set(result.column_names) == set(["a", "b", "c", "x", "index"])
    assert len(result["x"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()


@pytest.mark.parametrize(
    "use_visible_rows,use_visible_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_update_2(use_visible_rows, use_visible_columns, batched):
    """`update`, mixed datapanel, return multiple, new columns,
    `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=use_visible_rows, use_visible_columns=use_visible_columns
    )

    def func(x):
        out = {
            "x": (x["a"] + np.array(x["b"])) * 2,
            "y": (x["a"] * 6),
        }
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)

    result = dp.update(func, batch_size=4, batched=batched, num_workers=0)
    assert isinstance(result, DataPanel)
    assert set(result.column_names) == set(["a", "b", "c", "x", "y", "index"])
    assert len(result["x"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.array(visible_rows) * 6).all()


@pytest.mark.parametrize(
    "use_visible_rows, use_visible_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_update_3(use_visible_rows, use_visible_columns, batched):
    """`update`, mixed datapanel, return multiple, replace existing column,
    `batched=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_rows=False, use_visible_columns=False
    )

    def func(x):
        out = {
            "a": (x["a"] + np.array(x["b"])) * 2,
            "y": (x["a"] * 6),
        }
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)

    result = dp.update(func, batch_size=4, batched=batched, num_workers=0)
    assert isinstance(result, DataPanel)
    assert set(result.column_names) == set(["a", "b", "c", "y", "index"])
    assert len(result["a"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["a"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.array(visible_rows) * 6).all()


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

    result = dp.filter(func, batch_size=4, batched=batched, num_workers=0)
    if visible_rows is None:
        visible_rows = np.arange(16)

    assert isinstance(result, DataPanel)
    new_len = (np.array(visible_rows) % 2 == 0).sum()
    assert len(result) == new_len
    for col in result._data.values():
        assert len(col) == new_len

    # old datapane unchanged
    old_len = len(visible_rows)
    assert len(dp) == old_len
    for col in dp._data.values():
        # important to check that the column lengths are correct as well
        assert len(col) == old_len

    assert result.visible_columns == dp.visible_columns
    assert result.all_columns == dp.all_columns


@pytest.mark.parametrize(
    "use_visible_rows",
    product([True, False]),
)
def test_lz_map(tmpdir, use_visible_rows):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        use_visible_rows=use_visible_rows,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    visible_rows = (
        np.arange(length)
        if test_bed.visible_rows is None
        else np.array(test_bed.visible_rows)
    )

    def func(x):
        assert isinstance(x["img"], ImageColumn)
        return [str(img.filepath) for img in x["img"].lz]

    result = dp.map(func, materialize=False, num_workers=0, batched=True)

    assert isinstance(result, ListColumn)
    assert result.data == [test_bed.img_col.image_paths[i] for i in visible_rows]


@pytest.mark.parametrize(
    "use_visible_rows",
    product([True, False]),
)
def test_lz_filter(tmpdir, use_visible_rows):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        use_visible_rows=use_visible_rows,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    visible_rows = (
        np.arange(length)
        if test_bed.visible_rows is None
        else np.array(test_bed.visible_rows)
    )

    def func(x):
        # see `MockImageColumn` for filepath naming logic
        return (int(str(x["img"].filepath.name).split(".")[0]) % 2) == 0

    result = dp.filter(func, batched=False, num_workers=0, materialize=False)

    assert isinstance(result, DataPanel)
    new_len = (np.array(visible_rows) % 2 == 0).sum()
    assert len(result) == new_len
    for col in result._data.values():
        assert len(col) == new_len

    # old datapane unchanged
    old_len = len(visible_rows)
    assert len(dp) == old_len
    for col in dp._data.values():
        # important to check that the column lengths are correct as well
        assert len(col) == old_len

    assert result.visible_columns == dp.visible_columns
    assert result.all_columns == dp.all_columns


@pytest.mark.parametrize(
    "use_visible_rows",
    product([True, False]),
)
def test_lz_update(tmpdir, use_visible_rows: bool):
    """`update`, mixed datapanel, return single, new columns, `batched=True`"""
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        use_visible_rows=use_visible_rows,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    visible_rows = (
        np.arange(length)
        if test_bed.visible_rows is None
        else np.array(test_bed.visible_rows)
    )

    def func(x):
        out = {"x": str(x["img"].filepath)}
        return out

    result = dp.update(
        func, batch_size=4, batched=False, num_workers=0, materialize=False
    )
    assert set(result.column_names) == set(["a", "b", "c", "x", "img", "index"])
    assert len(result["x"]) == len(visible_rows)
    assert result["x"].data == [test_bed.img_col.image_paths[i] for i in visible_rows]


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


def test_copy():
    pass
