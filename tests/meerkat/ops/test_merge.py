"""Unittests for Datasets."""
import os
from typing import Dict

import numpy as np
import pytest
import torch

from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.image_column import ImageColumn
from meerkat.columns.list_column import ListColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.errors import MergeError

from ...testbeds import MockImageColumn
from ..test_datapanel import DataPanelTestBed


class MergeTestBed(DataPanelTestBed):
    DEFAULT_CONFIG = {
        "lengths": [
            {"left": 12, "right": 16},
            {"left": 16, "right": 16},
            {"left": 16, "right": 12},
        ],
        "consolidated": [True, False],
    }

    def __init__(
        self,
        column_configs: Dict[str, AbstractColumn],
        simple: bool = False,
        lengths: int = 16,
        consolidated: int = 16,
        tmpdir: str = None,
    ):
        self.side_to_dp = {}
        if simple:
            # TODO (Sabri): do away with the simple testbed, and replace with the full
            # one after updating support for missing values
            # https://github.com/robustness-gym/meerkat/issues/123
            np.random.seed(1)
            self.side_to_dp["left"] = DataPanel.from_batch(
                {
                    "key": np.arange(lengths["left"]),
                    "b": list(np.arange(lengths["left"])),
                    "c": [[i] for i in np.arange(lengths["left"])],
                    "d": (torch.arange(lengths["left"]) % 3),
                    "e": [f"1_{i}" for i in np.arange(lengths["left"])],
                }
            ).lz[np.random.permutation(np.arange(lengths["left"]))]

            self.side_to_dp["right"] = DataPanel.from_batch(
                {
                    "key": np.arange(lengths["right"]),
                    "b": list(np.arange(lengths["right"])),
                    "e": [f"1_{i}" for i in np.arange(lengths["right"])],
                    "f": (np.arange(lengths["right"]) % 2),
                }
            )
        else:
            for side in ["left", "right"]:
                side_tmpdir = os.path.join(tmpdir, side)
                os.makedirs(side_tmpdir)
                column_testbeds = self._build_column_testbeds(
                    column_configs, length=lengths[side], tmpdir=side_tmpdir
                )
                columns = {
                    name: testbed.col for name, testbed in column_testbeds.items()
                }
                dp = DataPanel.from_batch(columns)

                dp["key"] = np.arange(len(dp))

                if consolidated:
                    dp.consolidate()

                if side == "left":
                    np.random.seed(1)
                    dp = dp.lz[np.random.permutation(np.arange(len(dp)))]
                self.side_to_dp[side] = dp


@pytest.fixture
def testbed(request, tmpdir):
    config = request.param
    return MergeTestBed(**config, tmpdir=tmpdir)


class TestMerge:
    @MergeTestBed.parametrize(params={"sort": [True, False]})
    def test_merge_inner(self, testbed: MergeTestBed, sort):
        dp1, dp2 = (
            testbed.side_to_dp["left"],
            testbed.side_to_dp["right"],
        )

        out = dp1.merge(
            dp2,
            on="key",
            how="inner",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        assert isinstance(out, DataPanel)
        assert len(out) == min(len(dp1), len(dp2))

        # # check sorted
        if sort:
            assert np.all(np.diff(out["key"]) >= 0)

        # assert set(out.columns) == set(expected_columns)
        for name in dp1.columns:
            if name in ["key"]:
                continue

            if isinstance(out[f"{name}_1"], ImageColumn):
                assert out[f"{name}_1"].__class__ == out[f"{name}_2"].__class__
                assert out[f"{name}_1"].data.is_equal(
                    out[f"{name}_2"].data.str.replace("right", "left")
                )
            else:
                assert out[f"{name}_1"].is_equal(out[f"{name}_2"])

    @MergeTestBed.parametrize(config={"simple": [True]}, params={"sort": [True, False]})
    def test_merge_outer(self, testbed, sort):
        dp1, dp2 = (
            testbed.side_to_dp["left"],
            testbed.side_to_dp["right"],
        )
        out = dp1.merge(
            dp2,
            on="key",
            how="outer",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        a1 = set(dp1["key"])
        a2 = set(dp2["key"])

        assert isinstance(out, DataPanel)
        assert len(out) == len(a1 | a2)

        # check columns
        expected_columns = ["key", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]
        assert set(out.columns) == set(expected_columns)

        # check sorted
        if sort:
            assert np.all(np.diff(out["key"]) >= 0)

        # check for `None` at unmatched rows
        mask_both = np.where([val in (a1 & a2) for val in out["key"]])[0]
        mask_1 = np.where([val in (a1 - a2) for val in out["key"]])[0]
        mask_2 = np.where([val in (a2 - a1) for val in out["key"]])[0]
        # check for equality at matched rows
        assert list(out.lz[mask_both]["b_1"]) == list(out.lz[mask_both]["b_2"])
        # check for `values` at unmatched rows
        assert set(out.lz[mask_1]["b_1"]) == a1 - a2
        assert set(out.lz[mask_2]["b_2"]) == a2 - a1
        # check for `None` at unmatched rows
        assert list(out.lz[mask_1]["b_2"]) == [None] * len(mask_1)
        assert list(out.lz[mask_2]["b_1"]) == [None] * len(mask_2)

        # check for `values` at unmatched rows
        assert set(out.lz[mask_1]["e_1"]) == set([f"1_{i}" for i in a1 - a2])
        assert set(out.lz[mask_2]["e_2"]) == set([f"1_{i}" for i in a2 - a1])
        # check for equality at matched rows
        assert list(out.lz[mask_1]["e_2"]) == [None] * len(mask_1)
        assert list(out.lz[mask_2]["e_1"]) == [None] * len(mask_2)

    @MergeTestBed.parametrize(config={"simple": [True]}, params={"sort": [True, False]})
    def test_merge_left(self, testbed, sort):
        dp1, dp2 = (
            testbed.side_to_dp["left"],
            testbed.side_to_dp["right"],
        )
        out = dp1.merge(
            dp2,
            on="key",
            how="left",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        a1 = set(dp1["key"])
        a2 = set(dp2["key"])

        assert isinstance(out, DataPanel)
        assert len(out) == len(a1)

        # check columns
        expected_columns = ["key", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]
        assert set(out.columns) == set(expected_columns)

        # check sorted
        if sort:
            assert np.all(np.diff(out["key"]) >= 0)

        # check for `None` at unmatched rows
        mask_both = np.where([val in (a1 & a2) for val in out["key"]])[0]
        mask_1 = np.where([val in (a1 - a2) for val in out["key"]])[0]

        # check for equality at matched rows
        assert list(out.lz[mask_both]["b_1"]) == list(out.lz[mask_both]["b_2"])
        # check for `values` at unmatched rows
        assert set(out.lz[mask_1]["b_1"]) == a1 - a2
        # check for `None` at unmatched rows
        assert list(out.lz[mask_1]["b_2"]) == [None] * len(mask_1)

        # check for `values` at unmatched rows
        assert set(out.lz[mask_1]["e_1"]) == set([f"1_{i}" for i in a1 - a2])
        # check for equality at matched rows
        assert list(out.lz[mask_1]["e_2"]) == [None] * len(mask_1)

    @MergeTestBed.parametrize(config={"simple": [True]}, params={"sort": [True, False]})
    def test_merge_right(self, testbed, sort):
        dp1, dp2 = (
            testbed.side_to_dp["left"],
            testbed.side_to_dp["right"],
        )
        out = dp1.merge(
            dp2,
            on="key",
            how="right",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        a1 = set(dp1["key"])
        a2 = set(dp2["key"])

        assert isinstance(out, DataPanel)
        assert len(out) == len(a2)

        # check columns
        expected_columns = ["key", "b_1", "b_2", "c", "d", "e_1", "e_2", "f"]
        assert set(out.columns) == set(expected_columns)

        # check sorted
        if sort:
            assert np.all(np.diff(out["key"]) >= 0)

        # check for `None` at unmatched rows
        mask_both = np.where([val in (a1 & a2) for val in out["key"]])[0]
        mask_2 = np.where([val in (a2 - a1) for val in out["key"]])[0]
        # check for equality at matched rows
        assert list(out.lz[mask_both]["b_1"]) == list(out.lz[mask_both]["b_2"])
        # check for `values` at unmatched rows
        assert set(out.lz[mask_2]["b_2"]) == a2 - a1
        # check for `None` at unmatched rows
        assert list(out.lz[mask_2]["b_1"]) == [None] * len(mask_2)

        # check for `values` at unmatched rows
        assert set(out.lz[mask_2]["e_2"]) == set([f"1_{i}" for i in a2 - a1])
        # check for equality at matched rows
        assert list(out.lz[mask_2]["e_1"]) == [None] * len(mask_2)

    def test_merge_output_column_types(self):
        dp1 = DataPanel.from_batch(
            {"a": np.arange(3), "b": ListColumn(["1", "2", "3"])}
        )
        dp2 = dp1.copy()

        out = dp1.merge(dp2, on="b", how="inner")
        assert isinstance(out["b"], ListColumn)

    def test_image_merge(self, tmpdir):
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
        assert [str(fp) for fp in out["img"].data] == [
            img_col_test_bed.image_paths[row] for row in rows
        ]

    def test_no_columns(tmpdir):
        length = 16
        dp1 = DataPanel.from_batch(
            {
                "a": np.arange(length),
            }
        )
        rows = np.arange(4, 8)
        dp2 = DataPanel.from_batch(
            {
                "a": rows,
            }
        )
        out = dp1.merge(dp2, on="a", how="inner")

        assert "a" in out.columns

    def test_no_columns_in_left(tmpdir):
        length = 16
        dp1 = DataPanel.from_batch(
            {
                "a": np.arange(length),
            }
        )
        rows = np.arange(4, 8)
        dp2 = DataPanel.from_batch({"a": rows, "b": rows})
        out = dp1.merge(dp2, on="a", how="inner")

        assert "a" in out.columns
        assert "b" in out.columns

    def test_no_columns_in_right(tmpdir):
        length = 16
        dp1 = DataPanel.from_batch(
            {
                "a": np.arange(length),
                "b": np.arange(length),
            }
        )
        rows = np.arange(4, 8)
        dp2 = DataPanel.from_batch(
            {
                "a": rows,
            }
        )
        out = dp1.merge(dp2, on="a", how="inner")

        assert "a" in out.columns
        assert "b" in out.columns

    def test_no_on(self):
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
            dp1.merge(dp2)

    def test_check_merge_columns(self):
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

        # checks if Cells in cell columns are NOT hashable
        dp1 = DataPanel.from_batch(
            {
                "a": ImageColumn.from_filepaths(["a"] * length),
                "b": list(np.arange(length)),
            }
        )
        dp2 = dp1.copy()
        with pytest.raises(MergeError):
            dp1.merge(dp2, on="a")

        # checks that having a column called __right_indices__ raises a merge error
        dp1 = DataPanel.from_batch(
            {
                "a": ListColumn(["hello"] + [{"a": 1}] * (length - 1)),
                "b": list(np.arange(length)),
                "__right_indices__": list(np.arange(length)),
            }
        )
        dp2 = dp1.copy()
        with pytest.raises(MergeError):
            dp1.merge(dp2, on="__right_indices__")
