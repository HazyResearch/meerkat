"""Unittests for Datasets."""
import os
import tempfile
from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch
import ujson as json
from PIL.Image import Image

from meerkat import NumpyArrayColumn
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.image_column import ImageColumn
from meerkat.columns.lambda_column import LambdaCell
from meerkat.columns.list_column import ListColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel

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


def test_from_jsonl():
    # Build jsonl file
    temp_f = tempfile.NamedTemporaryFile()
    data = {
        "a": [3.4, 2.3, 1.2],
        "b": [[7, 9], [4], [1, 2]],
        "c": ["the walk", "the talk", "blah"],
    }
    with open(temp_f.name, "w") as out_f:
        for idx in range(3):
            to_write = {k: data[k][idx] for k in list(data.keys())}
            out_f.write(json.dumps(to_write) + "\n")

    dp_new = DataPanel.from_jsonl(temp_f.name)
    assert dp_new.column_names == ["a", "b", "c", "index"]
    # Skip index column
    for k in data:
        if isinstance(dp_new[k], NumpyArrayColumn):
            data_to_compare = dp_new[k]._data.tolist()
        else:
            data_to_compare = dp_new[k]._data
        assert data_to_compare == data[k]
    temp_f.close()


def test_from_csv():
    temp_f = tempfile.NamedTemporaryFile()
    data = {
        "a": [3.4, 2.3, 1.2],
        "b": ["alpha", "beta", "gamma"],
        "c": ["the walk", "the talk", "blah"],
    }
    pd.DataFrame(data).to_csv(temp_f.name)

    dp_new = DataPanel.from_csv(temp_f.name)
    assert dp_new.column_names == ["Unnamed: 0", "a", "b", "c", "index"]
    # Skip index column
    for k in data:
        if isinstance(dp_new[k], PandasSeriesColumn):
            data_to_compare = dp_new[k]._data.tolist()
        else:
            data_to_compare = dp_new[k]._data
        assert data_to_compare == data[k]


def test_col_index_single(tmpdir):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    # str index => single column ()
    index = "a"
    col = dp[index]
    assert isinstance(col, AbstractColumn)
    # enforce that a single column index returns a coreference
    assert col is dp._data["a"]


def test_col_index_multiple(tmpdir):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    # str index => single column ()
    index = ["a", "b"]
    new_dp = dp[index]
    assert isinstance(new_dp, DataPanel)

    # enforce that a column index multiple returns a view of the old datapanel
    for col_name in index:
        assert new_dp._data[col_name] is not dp._data[col_name]
        assert new_dp._data[col_name].data is dp._data[col_name].data


def test_row_index_single(tmpdir):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    # int index => single row (dict)
    index = 2
    row = dp[index]
    assert isinstance(row["img"], Image)
    assert (np.array(row["img"]) == test_bed.img_col.image_arrays[index]).all()
    assert row["a"] == index
    assert row["b"] == index


@pytest.mark.parametrize(
    "index_type,consolidate",
    product([NumpyArrayColumn, PandasSeriesColumn, TensorColumn], [True, False]),
)
def test_row_index_multiple(tmpdir, index_type, consolidate):
    length = 16
    rows = np.arange(length)
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    if consolidate:
        dp.consolidate()
    # slice index => multiple row selection (DataPanel)
    # tuple or list index => multiple row selection (DataPanel)
    # np.array indeex => multiple row selection (DataPanel)
    for rows, indices in (
        (dp[1:3], rows[1:3]),
        (dp[[0, 2]], rows[[0, 2]]),
        (dp[index_type(np.array((0,)))], rows[np.array((0,))]),
        (dp[index_type(np.array((1, 1)))], rows[np.array((1, 1))]),
        (
            dp[index_type(np.array((True, False) * (length // 2)))],
            rows[np.array((True, False) * (length // 2))],
        ),
        # (
        #     dp[index_type(dp["a"].data % 2 == 0)],
        #     rows[rows % 2 == 0],
        # ),
    ):
        assert isinstance(rows["img"], ListColumn)
        assert (rows["a"].data == indices).all()
        assert (rows["b"].data == indices).all()
        assert (rows["d"].data == torch.tensor(indices)).all()
        assert (rows["e"].data.values == indices).all()
        assert (rows["f"].data == 1).all()
        assert len(rows["f"].shape) == 2
        assert (rows["g"].data == 1).all()
        assert len(rows["g"].shape) == 2


def test_row_lz_index_single(tmpdir):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    # int index => single row (dict)
    index = 2
    row = dp.lz[index]
    assert isinstance(row["img"], LambdaCell)
    assert str(row["img"].data) == test_bed.img_col.image_paths[index]
    assert row["a"] == index
    assert row["b"] == index


@pytest.mark.parametrize(
    "index_type,consolidate",
    product([NumpyArrayColumn, PandasSeriesColumn, TensorColumn], [True, False]),
)
def test_row_lz_index_multiple(tmpdir, index_type, consolidate):
    length = 16
    rows = np.arange(length)
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    if consolidate:
        dp.consolidate()
    # slice index => multiple row selection (DataPanel)
    # tuple or list index => multiple row selection (DataPanel)
    # np.array indeex => multiple row selection (DataPanel)
    for rows, indices in (
        (dp.lz[1:3], rows[1:3]),
        (dp.lz[[0, 2]], rows[[0, 2]]),
        (dp.lz[index_type(np.array((0,)))], rows[np.array((0,))]),
        (dp.lz[index_type(np.array((1, 1)))], rows[np.array((1, 1))]),
        (
            dp.lz[index_type(np.array((True, False) * (length // 2)))],
            rows[np.array((True, False) * (length // 2))],
        ),
        (
            dp.lz[index_type(dp["a"].data % 2 == 0)],
            rows[rows % 2 == 0],
        ),
    ):
        assert isinstance(rows["img"], ImageColumn)
        assert list(map(lambda x: x, rows["img"].data)) == [
            test_bed.img_col.image_paths[i] for i in indices
        ]
        assert (rows["a"].data == indices).all()
        assert (rows["b"].data == indices).all()


def test_col_indexing_view_copy_semantics(tmpdir):
    testbed = MockDatapanel(length=16, include_image_column=True, tmpdir=tmpdir)
    dp = testbed.dp

    # Columns (1): Indexing a single column (i.e. with a str) returns the underlying
    # AbstractColumn object directly. In the example below col1 and col2 are
    # coreferences of the same column.
    for name in dp.columns:
        dp[name] is dp[name]

    # Columns (2): Indexing multiple columns (i.e. with Sequence[str]) returns a view of
    # the DataPanel holding views to the columns in the original DataPanel. This means
    # the AbstractColumn objects held in the new DataPanel are the same AbstractColumn
    # objects held in the original DataPanel.
    view_dp = dp[["a", "b"]]
    for name in view_dp.columns:
        dp[name] is not view_dp[name]
        dp[name].data is dp[name].data


def test_row_indexing_view_copy_semantics(tmpdir):
    testbed = MockDatapanel(length=16, include_image_column=True, tmpdir=tmpdir)
    dp = testbed.dp

    # slice index
    dp2 = dp[:8]
    col = "a"
    assert isinstance(dp2[col], NumpyArrayColumn)
    assert dp[col] is not dp2[col]
    assert dp[col].data is not dp2[col].data
    assert dp[col].data is dp2[col].data.base

    col = "d"
    assert isinstance(dp2[col], TensorColumn)
    assert dp[col] is not dp2[col]
    assert dp[col].data is not dp2[col].data
    # note `data_ptr` checks whether the tensors have the same memory address of the
    # first element, so this would not work if the slice didn't start at 0
    assert dp[col].data.data_ptr() == dp2[col].data.data_ptr()

    col = "e"
    assert isinstance(dp2[col], PandasSeriesColumn)
    assert dp[col] is not dp2[col]
    assert dp[col].data is not dp2[col].data
    # TODO (sabri): Figure out pandas copying behavior, it's not clear how it works and
    # this deserves a deeper investigation.
    # assert dp[col].data.values.base is dp2[col].data.values.base

    # slice index
    dp2 = dp[np.array([0, 1, 2, 5])]
    col = "a"
    assert isinstance(dp2[col], NumpyArrayColumn)
    assert dp[col] is not dp2[col]
    assert dp[col].data is not dp2[col].data
    assert dp[col].data is not dp2[col].data.base

    col = "d"
    assert isinstance(dp2[col], TensorColumn)
    assert dp[col] is not dp2[col]
    assert dp[col].data is not dp2[col].data
    # note `data_ptr` checks whether the tensors have the same memory address of the
    # first element, so this would not work if the slice didn't start at 0
    assert dp[col].data.data_ptr() != dp2[col].data.data_ptr()

    col = "e"
    assert isinstance(dp2[col], PandasSeriesColumn)
    assert dp[col] is not dp2[col]
    assert dp[col].data is not dp2[col].data
    assert dp[col].data.values is not dp2[col].data.values.base


@pytest.mark.parametrize(
    "use_visible_columns, use_input_columns, num_workers",
    product([True, False], [True, False], [0, 2]),
)
def test_map_1(use_visible_columns, use_input_columns, num_workers):
    """`map`, mixed datapanel, single return, `is_batched_fn=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns
    )
    input_columns = ["a", "b"] if use_input_columns else None

    def func(x):
        if use_input_columns:
            assert x.visible_columns == ["a", "b", "index"]
        out = (x["a"] + np.array(x["b"])) * 2
        return out

    rows = np.arange(16)
    result = dp.map(
        func,
        batch_size=4,
        is_batched_fn=True,
        num_workers=num_workers,
        input_columns=input_columns,
    )
    assert isinstance(result, NumpyArrayColumn)
    assert len(result) == len(rows)
    assert (result == np.array(rows) * 4).all()


@pytest.mark.parametrize("num_workers", [0, 2])
def test_map_2(num_workers):
    """`map`, mixed datapanel, return multiple, `is_batched_fn=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(use_visible_columns=False)

    def func(x):
        out = {
            "x": (x["a"] + np.array(x["b"])) * 2,
            "y": np.array([x["c"][i]["a"] for i in range(len(x["c"]))]),
        }
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)
    result = dp.map(
        func,
        batch_size=4,
        is_batched_fn=True,
        num_workers=num_workers,
    )
    assert isinstance(result, DataPanel)
    assert len(result["x"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.ones(len(visible_rows)) * 2).all()


@pytest.mark.parametrize(
    "use_visible_columns,use_input_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_update_1(use_visible_columns, use_input_columns, batched):
    """`update`, mixed datapanel, return single, new columns."""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns,
    )

    # mixed datapanel (i.e. has multiple colummn types)
    def func(x):
        out = {"x": (x["a"] + np.array(x["b"])) * 2}
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)

    input_columns = ["a", "b"] if use_input_columns else None
    result = dp.update(
        func,
        batch_size=4,
        is_batched_fn=batched,
        num_workers=0,
        input_columns=input_columns,
    )
    assert isinstance(result, DataPanel)
    assert set(result.visible_columns) == set(visible_columns + ["x"])
    assert len(result["x"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()


@pytest.mark.parametrize(
    "use_visible_columns,use_input_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_update_2(use_visible_columns, use_input_columns, batched):
    """`update`, mixed datapanel, return multiple, new columns,
    `is_batched_fn=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns
    )

    def func(x):
        out = {
            "x": (x["a"] + np.array(x["b"])) * 2,
            "y": (x["a"] * 6),
        }
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)

    input_columns = ["a", "b"] if use_input_columns else None
    result = dp.update(
        func,
        batch_size=4,
        is_batched_fn=batched,
        num_workers=0,
        input_columns=input_columns,
    )
    assert isinstance(result, DataPanel)
    assert set(result.visible_columns) == set(visible_columns + ["x", "y"])
    assert len(result["x"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["x"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.array(visible_rows) * 6).all()


@pytest.mark.parametrize(
    "use_visible_columns,use_input_columns,batched",
    product([True, False], [True, False], [True, False]),
)
def test_update_3(use_visible_columns, use_input_columns, batched):
    """`update`, mixed datapanel, return multiple, replace existing column,
    `is_batched_fn=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns
    )

    def func(x):
        out = {
            "a": (x["a"] + np.array(x["b"])) * 2,
            "y": (x["a"] * 6),
        }
        return out

    if visible_rows is None:
        visible_rows = np.arange(16)

    input_columns = ["a", "b"] if use_input_columns else None
    result = dp.update(
        func,
        batch_size=4,
        is_batched_fn=batched,
        num_workers=0,
        input_columns=input_columns,
    )
    assert isinstance(result, DataPanel)
    assert set(result.visible_columns) == set(visible_columns + ["y"])
    assert len(result["a"]) == len(visible_rows)
    assert len(result["y"]) == len(visible_rows)
    assert (result["a"] == np.array(visible_rows) * 4).all()
    assert (result["y"] == np.array(visible_rows) * 6).all()


@pytest.mark.parametrize(
    "use_visible_columns,batched",
    product([True, False], [True, False]),
)
def test_filter_1(use_visible_columns, batched):
    """`filter`, mixed datapanel."""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns
    )

    def func(x):
        return (x["a"] % 2) == 0

    result = dp.filter(func, batch_size=4, is_batched_fn=batched, num_workers=0)
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


def test_lz_map(tmpdir):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
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
        return [str(filepath) for filepath in x["img"].data]

    result = dp.map(func, materialize=False, num_workers=0, is_batched_fn=True)

    assert isinstance(result, ListColumn)
    assert result.data == [test_bed.img_col.image_paths[i] for i in visible_rows]


def test_lz_filter(tmpdir):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
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
        return (int(str(x["img"].data).split("/")[-1].split(".")[0]) % 2) == 0

    result = dp.filter(func, is_batched_fn=False, num_workers=0, materialize=False)

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


def test_lz_update(
    tmpdir,
):
    """`update`, mixed datapanel, return single, new columns,
    `is_batched_fn=True`"""
    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp
    visible_rows = (
        np.arange(length)
        if test_bed.visible_rows is None
        else np.array(test_bed.visible_rows)
    )

    def func(x):
        out = {"x": str(x["img"].data)}
        return out

    result = dp.update(
        func, batch_size=4, is_batched_fn=False, num_workers=0, materialize=False
    )
    assert set(result.column_names) == set(
        ["a", "b", "c", "d", "e", "f", "g", "x", "img", "index"]
    )
    assert len(result["x"]) == len(visible_rows)
    assert result["x"].data == [test_bed.img_col.image_paths[i] for i in visible_rows]


@pytest.mark.parametrize(
    "use_visible_columns,batched",
    product([True, False], [True, False]),
)
def test_filter_2(use_visible_columns, batched):
    """`filter`, mixed datapanel."""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns
    )

    def func(x):
        return (x["a"] % 2) == 0

    result = dp.filter(func, batch_size=4, is_batched_fn=batched)
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


def test_remove_column():
    a = np.arange(16)
    b = np.arange(16) * 2
    dp = DataPanel.from_batch({"a": a, "b": b})
    assert "a" in dp
    dp.remove_column("a")
    assert "a" not in dp


def test_overwrite_column():
    # make sure we remove the column when overwriting it
    a = np.arange(16)
    b = np.arange(16) * 2
    dp = DataPanel.from_batch({"a": a, "b": b})
    assert "a" in dp
    assert dp[["a", "b"]]["a"]._data is a
    # testing removal from block manager, so important to use non-blockable type here
    dp["a"] = ListColumn(range(16))
    assert dp[["a", "b"]]["a"]._data is not a


@pytest.mark.parametrize("use_visible_columns", [True, False])
def test_io(tmp_path, use_visible_columns):
    """`map`, mixed datapanel, return multiple, `is_batched_fn=True`"""
    dp, visible_rows, visible_columns = _get_datapanel(
        use_visible_columns=use_visible_columns
    )
    path = os.path.join(tmp_path, "test")
    dp.write(path)

    new_dp = DataPanel.read(path)

    assert isinstance(new_dp, DataPanel)
    assert len(new_dp) == len(dp)
    assert len(new_dp["a"]) == len(dp["a"])
    assert len(new_dp["b"]) == len(dp["b"])
    if not use_visible_columns:
        assert len(new_dp["c"]) == len(dp["c"])

    assert (dp["a"] == new_dp["a"]).all()
    assert dp["b"].data == new_dp["b"].data

    assert dp.visible_columns == new_dp.visible_columns


def test_repr_html_():
    dp, visible_rows, visible_columns = _get_datapanel(use_visible_columns=False)
    dp._repr_html_()


@pytest.mark.parametrize(
    "use_visible_columns",
    product([True, False]),
)
def test_to_pandas(tmpdir, use_visible_columns):
    import pandas as pd

    length = 16
    test_bed = MockDatapanel(
        length=length,
        include_image_column=True,
        use_visible_columns=use_visible_columns,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    df = dp.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == dp.visible_columns
    assert len(df) == len(dp)

    assert (df["a"].values == dp["a"].data).all()
    assert list(df["b"]) == list(dp["b"].data)

    if not use_visible_columns:
        assert isinstance(df["c"][0], dict)
        assert isinstance(df["img"][0], LambdaCell)

        assert isinstance(df["d"].values == dp["d"].numpy()).all()
        assert isinstance(df["e"].values == dp["e"].numpy()).all()


@pytest.mark.parametrize(
    "use_visible_columns",
    product([True, False]),
)
def test_head(tmpdir, use_visible_columns):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        use_visible_columns=use_visible_columns,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    new_dp = dp.head(n=2)

    assert isinstance(new_dp, DataPanel)
    assert new_dp.visible_columns == dp.visible_columns
    assert len(new_dp) == 2
    assert (new_dp["a"] == dp["a"][:2]).all()


@pytest.mark.parametrize(
    "use_visible_columns",
    product([True, False]),
)
def test_tail(tmpdir, use_visible_columns):
    length = 16
    test_bed = MockDatapanel(
        length=length,
        use_visible_columns=use_visible_columns,
        tmpdir=tmpdir,
    )
    dp = test_bed.dp

    new_dp = dp.tail(n=2)

    assert isinstance(new_dp, DataPanel)
    assert new_dp.visible_columns == dp.visible_columns
    assert len(new_dp) == 2
    assert (new_dp["a"] == dp["a"][-2:]).all()


@pytest.mark.parametrize(
    "use_visible_columns",
    product([True, False], [True, False]),
)
def test_append_columns(use_visible_columns):
    mock = MockDatapanel(
        length=16,
        use_visible_columns=use_visible_columns,
    )

    out = mock.dp.append(mock.dp, axis="rows")

    assert len(out) == len(mock.visible_rows) * 2
    assert isinstance(out, DataPanel)
    assert set(out.visible_columns) == set(mock.visible_columns)
    assert (out["a"].data == np.concatenate([mock.visible_rows] * 2)).all()
    assert out["b"].data == list(np.concatenate([mock.visible_rows] * 2))


class DataPanelSubclass(DataPanel):
    """Mock class to test that ops on subclass returns subclass."""

    pass


def test_subclass():
    dp1 = DataPanelSubclass.from_dict({"a": np.arange(3), "b": ["may", "jun", "jul"]})
    dp2 = DataPanelSubclass.from_dict(
        {"c": np.arange(3), "d": ["2021", "2022", "2023"]}
    )

    assert isinstance(dp1.lz[np.asarray([0, 1])], DataPanelSubclass)
    assert isinstance(dp1.lz[:2], DataPanelSubclass)
    assert isinstance(dp1[:2], DataPanelSubclass)

    assert isinstance(dp1.merge(dp2, left_on="a", right_on="c"), DataPanelSubclass)
    assert isinstance(dp1.append(dp1), DataPanelSubclass)
