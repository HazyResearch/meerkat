"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest
import torch

from mosaic.columns.image_column import ImageColumn
from mosaic.columns.list_column import ListColumn
from mosaic.columns.numpy_column import NumpyArrayColumn
from mosaic.columns.tensor_column import TensorColumn
from mosaic.datapanel import DataPanel
from mosaic.errors import MergeError

from ...testbeds import MockImageColumn


def get_dps(
    length1: int,
    length2: int,
    use_visible_rows: bool = False,
    use_visible_columns: bool = False,
    include_image_column: bool = False,
    tmpdir: str = None,
):
    shuffle1 = np.arange(length1)
    batch1 = {
        "a": np.arange(length1)[shuffle1],
        "b": list(np.arange(length1)[shuffle1]),
        "c": [[i] for i in np.arange(length1)[shuffle1]],
        "d": (torch.arange(length1) % 3)[shuffle1],
        "e": [f"1_{i}" for i in np.arange(length1)[shuffle1]],
    }

    np.random.seed(1)
    shuffle2 = np.random.permutation(np.arange(length2))
    batch2 = {
        "a": np.arange(length2)[shuffle2],
        "b": list(np.arange(length2)[shuffle2]),
        "e": [f"1_{i}" for i in np.arange(length1)[shuffle2]],
        "f": (np.arange(length2) % 2)[shuffle2],
    }

    if include_image_column:
        img_col = MockImageColumn(length=length1, tmpdir=tmpdir).col
        batch1["img"] = img_col
        img_col = MockImageColumn(length=length2, tmpdir=tmpdir).col
        batch2["img"] = img_col.lz[shuffle2]

    visible_rows = [0, 4, 6, 11] if use_visible_rows else None
    visible_columns = ["a", "b", "c"] if use_visible_columns else None

    dps = []
    for batch, shuffle in [(batch1, shuffle1), (batch2, shuffle2)]:
        dp = DataPanel.from_batch(batch)
        if use_visible_rows:
            for column in dp.values():
                column.visible_rows = visible_rows

        if use_visible_columns:
            dp.visible_columns = visible_columns

        dps.append(dp)
    return dps[0], dps[1], visible_rows, visible_columns, shuffle1, shuffle2


@pytest.mark.parametrize(
    "use_visible_rows,use_visible_columns,diff_length,sort",
    product([True, False], [True, False], [True, False], [True, False]),
)
def test_merge_inner(use_visible_rows, use_visible_columns, diff_length, sort):
    length1 = 16
    length2 = 12 if diff_length else 16
    dp1, dp2, visible_rows, visible_columns, shuffle1, shuffle2 = get_dps(
        length1=length1,
        length2=length2,
        use_visible_rows=use_visible_rows,
        use_visible_columns=use_visible_columns,
    )
    out = dp1.merge(
        dp2, on="a", how="inner", keep_indexes=False, suffixes=("_1", "_2"), sort=sort
    )

    assert isinstance(out, DataPanel)
    if use_visible_rows:
        # need to compute how many shared rows there are between the two dps
        expected_length = sum(
            [row_idx in visible_rows for row_idx in shuffle2[visible_rows]]
        )
    else:
        expected_length = min(length1, length2)
    assert len(out) == expected_length

    # check columns
    if use_visible_columns:
        expected_columns = ["a", "index", "b_1", "b_2", "c"]
    else:
        expected_columns = ["a", "index", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]

    # check sorted
    if sort:
        assert np.all(np.diff(out["a"]) >= 0)

    assert set(out.columns) == set(expected_columns)

    assert (out["b_1"] == out["b_2"]).all()
    if not use_visible_columns:
        assert list(out["e_1"]) == list(out["e_2"])


@pytest.mark.parametrize(
    "use_visible_rows,use_visible_columns,sort",
    product([True, False], [True, False], [True, False]),
)
def test_merge_outer(use_visible_rows, use_visible_columns, sort):
    dp1, dp2, visible_rows, visible_columns, shuffle1, shuffle2 = get_dps(
        length1=16,
        length2=12,
        use_visible_rows=use_visible_rows,
        use_visible_columns=use_visible_columns,
    )
    out = dp1.merge(
        dp2, on="a", how="outer", keep_indexes=False, suffixes=("_1", "_2"), sort=sort
    )

    a1 = set(shuffle1[visible_rows]) if use_visible_rows else set(shuffle1)
    a2 = set(shuffle2[visible_rows]) if use_visible_rows else set(shuffle2)

    assert isinstance(out, DataPanel)
    assert len(out) == len(a1 | a2)

    # check columns
    if use_visible_columns:
        expected_columns = ["a", "index", "b_1", "b_2", "c"]
    else:
        expected_columns = ["a", "index", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]
    assert set(out.columns) == set(expected_columns)

    # check sorted
    if sort:
        assert np.all(np.diff(out["a"]) >= 0)

    # check for `None` at unmatched rows
    mask_both = np.where([val in (a1 & a2) for val in out["a"]])[0]
    mask_1 = np.where([val in (a1 - a2) for val in out["a"]])[0]
    mask_2 = np.where([val in (a2 - a1) for val in out["a"]])[0]
    # check for equality at matched rows
    assert list(out.lz[mask_both]["b_1"]) == list(out.lz[mask_both]["b_2"])
    # check for `values` at unmatched rows
    assert set(out.lz[mask_1]["b_1"]) == a1 - a2
    assert set(out.lz[mask_2]["b_2"]) == a2 - a1
    # check for `None` at unmatched rows
    assert list(out.lz[mask_1]["b_2"]) == [None] * len(mask_1)
    assert list(out.lz[mask_2]["b_1"]) == [None] * len(mask_2)

    if not use_visible_columns:
        # check for `values` at unmatched rows
        assert set(out.lz[mask_1]["e_1"]) == set([f"1_{i}" for i in a1 - a2])
        assert set(out.lz[mask_2]["e_2"]) == set([f"1_{i}" for i in a2 - a1])
        # check for equality at matched rows
        assert list(out.lz[mask_1]["e_2"]) == [None] * len(mask_1)
        assert list(out.lz[mask_2]["e_1"]) == [None] * len(mask_2)


@pytest.mark.parametrize(
    "use_visible_rows,use_visible_columns,sort",
    product([True, False], [True, False], [True, False]),
)
def test_merge_left(use_visible_rows, use_visible_columns, sort):
    dp1, dp2, visible_rows, visible_columns, shuffle1, shuffle2 = get_dps(
        length1=16,
        length2=12,
        use_visible_rows=use_visible_rows,
        use_visible_columns=use_visible_columns,
    )
    out = dp1.merge(
        dp2, on="a", how="left", keep_indexes=False, suffixes=("_1", "_2"), sort=sort
    )

    a1 = set(shuffle1[visible_rows]) if use_visible_rows else set(shuffle1)
    a2 = set(shuffle2[visible_rows]) if use_visible_rows else set(shuffle2)

    assert isinstance(out, DataPanel)
    assert len(out) == len(a1)

    # check columns
    if use_visible_columns:
        expected_columns = ["a", "index", "b_1", "b_2", "c"]
    else:
        expected_columns = ["a", "index", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]
    assert set(out.columns) == set(expected_columns)

    # check sorted
    if sort:
        assert np.all(np.diff(out["a"]) >= 0)

    # check for `None` at unmatched rows
    mask_both = np.where([val in (a1 & a2) for val in out["a"]])[0]
    mask_1 = np.where([val in (a1 - a2) for val in out["a"]])[0]

    # check for equality at matched rows
    assert list(out.lz[mask_both]["b_1"]) == list(out.lz[mask_both]["b_2"])
    # check for `values` at unmatched rows
    assert set(out.lz[mask_1]["b_1"]) == a1 - a2
    # check for `None` at unmatched rows
    assert list(out.lz[mask_1]["b_2"]) == [None] * len(mask_1)

    if not use_visible_columns:
        # check for `values` at unmatched rows
        assert set(out.lz[mask_1]["e_1"]) == set([f"1_{i}" for i in a1 - a2])
        # check for equality at matched rows
        assert list(out.lz[mask_1]["e_2"]) == [None] * len(mask_1)


@pytest.mark.parametrize(
    "use_visible_rows,use_visible_columns,sort",
    product([True, False], [True, False], [True, False]),
)
def test_merge_right(use_visible_rows, use_visible_columns, sort):
    dp1, dp2, visible_rows, visible_columns, shuffle1, shuffle2 = get_dps(
        length1=16,
        length2=12,
        use_visible_rows=use_visible_rows,
        use_visible_columns=use_visible_columns,
    )
    out = dp1.merge(
        dp2, on="a", how="right", keep_indexes=False, suffixes=("_1", "_2"), sort=sort
    )

    a1 = set(shuffle1[visible_rows]) if use_visible_rows else set(shuffle1)
    a2 = set(shuffle2[visible_rows]) if use_visible_rows else set(shuffle2)

    assert isinstance(out, DataPanel)
    assert len(out) == len(a2)

    # check columns
    if use_visible_columns:
        expected_columns = ["a", "index", "b_1", "b_2", "c"]
    else:
        expected_columns = ["a", "index", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]
    assert set(out.columns) == set(expected_columns)

    # check sorted
    if sort:
        assert np.all(np.diff(out["a"]) >= 0)

    # check for `None` at unmatched rows
    mask_both = np.where([val in (a1 & a2) for val in out["a"]])[0]
    mask_2 = np.where([val in (a2 - a1) for val in out["a"]])[0]
    # check for equality at matched rows
    assert list(out.lz[mask_both]["b_1"]) == list(out.lz[mask_both]["b_2"])
    # check for `values` at unmatched rows
    assert set(out.lz[mask_2]["b_2"]) == a2 - a1
    # check for `None` at unmatched rows
    assert list(out.lz[mask_2]["b_1"]) == [None] * len(mask_2)

    if not use_visible_columns:
        # check for `values` at unmatched rows
        assert set(out.lz[mask_2]["e_2"]) == set([f"1_{i}" for i in a2 - a1])
        # check for equality at matched rows
        assert list(out.lz[mask_2]["e_1"]) == [None] * len(mask_2)


def test_merge_output_column_types():
    dp1 = DataPanel.from_batch({"a": np.arange(3), "b": ListColumn(["1", "2", "3"])})
    dp2 = dp1.copy()

    out = dp1.merge(dp2, on="b", how="inner")
    assert isinstance(out["b"], ListColumn)


def test_cell_merge(tmpdir):
    length = 16
    img_col_test_bed = MockImageColumn(length=length, tmpdir=tmpdir)
    dp1 = DataPanel.from_batch(
        {
            "a": np.arange(length),
            "img": img_col_test_bed.col,
        }
    )
    rows = np.arange(4, 8)
    dp2 = DataPanel.from_batch(
        {
            "a": rows,
        }
    )

    out = dp1.merge(dp2, on="a", how="inner")
    assert isinstance(out["img"], ImageColumn)
    assert [str(cell.filepath) for cell in out["img"].data] == [
        img_col_test_bed.image_paths[row] for row in rows
    ]


def test_check_merge_columns():
    length = 16
    # check dictionary not hashable
    dp1 = DataPanel.from_batch(
        {
            "a": ListColumn([{"a": 1}] * length),
            "b": list(np.arange(length)),
        }
    )
    dp2 = dp1.copy()
    with pytest.raises(MergeError):
        dp1.merge(dp2, on=["a"])

    # check multi-on
    with pytest.raises(MergeError):
        dp1.merge(dp2, on=["a", "b"])

    # check multi-dimensional numpy array
    dp1 = DataPanel.from_batch(
        {
            "a": NumpyArrayColumn(np.stack([np.arange(5)] * length)),
            "b": list(np.arange(length)),
        }
    )
    dp2 = dp1.copy()
    with pytest.raises(MergeError):
        dp1.merge(dp2, on="a")

    # check multi-dimensional numpy array
    dp1 = DataPanel.from_batch(
        {
            "a": TensorColumn(torch.stack([torch.arange(5)] * length)),
            "b": list(np.arange(length)),
        }
    )
    dp2 = dp1.copy()
    with pytest.raises(MergeError):
        dp1.merge(dp2, on="a")

    # checks that **all** cells are hashable (not just the first)
    dp1 = DataPanel.from_batch(
        {
            "a": ListColumn(["hello"] + [{"a": 1}] * (length - 1)),
            "b": list(np.arange(length)),
        }
    )
    dp2 = dp1.copy()
    with pytest.raises(MergeError):
        dp1.merge(dp2, on="a")
