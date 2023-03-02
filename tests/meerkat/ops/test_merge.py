"""Unittests for Datasets."""
import os
from typing import Dict

import numpy as np
import pytest
import torch

from meerkat.columns.abstract import Column
from meerkat.columns.deferred.file import FileColumn
from meerkat.columns.deferred.image import ImageColumn
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.errors import MergeError

from ...testbeds import MockImageColumn
from ..test_dataframe import DataFrameTestBed


class MergeTestBed(DataFrameTestBed):
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
        column_configs: Dict[str, Column],
        simple: bool = False,
        lengths: int = 16,
        consolidated: int = 16,
        tmpdir: str = None,
    ):
        self.side_to_df = {}
        if simple:
            # TODO (Sabri): do away with the simple testbed, and replace with the full
            # one after updating support for missing values
            # https://github.com/robustness-gym/meerkat/issues/123
            np.random.seed(1)
            self.side_to_df["left"] = DataFrame.from_batch(
                {
                    "key": np.arange(lengths["left"]),
                    "b": list(np.arange(lengths["left"])),
                    "c": [[i] for i in np.arange(lengths["left"])],
                    "d": (torch.arange(lengths["left"]) % 3),
                    "e": [f"1_{i}" for i in np.arange(lengths["left"])],
                }
            )[np.random.permutation(np.arange(lengths["left"]))]

            self.side_to_df["right"] = DataFrame.from_batch(
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
                df = DataFrame.from_batch(columns)

                df["key"] = np.arange(len(df))

                if consolidated:
                    df.consolidate()

                if side == "left":
                    np.random.seed(1)
                    df = df[np.random.permutation(np.arange(len(df)))]
                self.side_to_df[side] = df


@pytest.fixture
def testbed(request, tmpdir):
    config = request.param
    return MergeTestBed(**config, tmpdir=tmpdir)


class TestMerge:
    @MergeTestBed.parametrize(params={"sort": [True, False]})
    def test_merge_inner(self, testbed: MergeTestBed, sort):
        df1, df2 = (
            testbed.side_to_df["left"],
            testbed.side_to_df["right"],
        )

        out = df1.merge(
            df2,
            on="key",
            how="inner",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        assert isinstance(out, DataFrame)
        assert len(out) == min(len(df1), len(df2))

        # # check sorted
        if sort:
            assert np.all(np.diff(out["key"]) >= 0)

        # assert set(out.columns) == set(expected_columns)
        for name in df1.columns:
            if name in ["key"]:
                continue

            if isinstance(out[f"{name}_1"], ImageColumn):
                assert out[f"{name}_1"].__class__ == out[f"{name}_2"].__class__
                assert (
                    out[f"{name}_1"]
                    .data.args[0]
                    .is_equal(
                        out[f"{name}_2"].data.args[0].str.replace("right", "left")
                    )
                )
            else:
                assert out[f"{name}_1"].is_equal(out[f"{name}_2"])

    @pytest.mark.skip
    @MergeTestBed.parametrize(config={"simple": [True]}, params={"sort": [True, False]})
    def test_merge_outer(self, testbed, sort):
        df1, df2 = (
            testbed.side_to_df["left"],
            testbed.side_to_df["right"],
        )
        out = df1.merge(
            df2,
            on="key",
            how="outer",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        a1 = set(df1["key"])
        a2 = set(df2["key"])

        assert isinstance(out, DataFrame)
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
        assert list(out[mask_both]["b_1"]) == list(out[mask_both]["b_2"])
        # check for `values` at unmatched rows
        assert set(out[mask_1]["b_1"]) == a1 - a2
        assert set(out[mask_2]["b_2"]) == a2 - a1
        # check for `None` at unmatched rows
        assert np.isnan(out[mask_1]["b_2"]).all()
        assert np.isnan(out[mask_2]["b_1"]).all()

        # check for `values` at unmatched rows
        assert set(out[mask_1]["e_1"]) == set([f"1_{i}" for i in a1 - a2])
        assert set(out[mask_2]["e_2"]) == set([f"1_{i}" for i in a2 - a1])
        # check for equality at matched rows
        assert out[mask_1]["e_2"].isna().all()
        assert out[mask_2]["e_1"].isna().all()

    @pytest.mark.skip
    @MergeTestBed.parametrize(config={"simple": [True]}, params={"sort": [True, False]})
    def test_merge_left(self, testbed, sort):
        df1, df2 = (
            testbed.side_to_df["left"],
            testbed.side_to_df["right"],
        )
        out = df1.merge(
            df2,
            on="key",
            how="left",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        a1 = set(df1["key"])
        a2 = set(df2["key"])

        assert isinstance(out, DataFrame)
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
        assert list(out[mask_both]["b_1"]) == list(out[mask_both]["b_2"])
        # check for `values` at unmatched rows
        assert set(out[mask_1]["b_1"]) == a1 - a2
        # check for `None` at unmatched rows
        assert out[mask_1]["b_2"].isna().all()

        # check for `values` at unmatched rows
        assert set(out[mask_1]["e_1"]) == set([f"1_{i}" for i in a1 - a2])
        # check for equality at matched rows
        assert out[mask_1]["e_2"].isna().all()

    @pytest.mark.skip
    @MergeTestBed.parametrize(config={"simple": [True]}, params={"sort": [True, False]})
    def test_merge_right(self, testbed, sort):
        df1, df2 = (
            testbed.side_to_df["left"],
            testbed.side_to_df["right"],
        )
        out = df1.merge(
            df2,
            on="key",
            how="right",
            suffixes=("_1", "_2"),
            sort=sort,
        )

        a1 = set(df1["key"])
        a2 = set(df2["key"])

        assert isinstance(out, DataFrame)
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
        assert list(out[mask_both]["b_1"]) == list(out[mask_both]["b_2"])
        # check for `values` at unmatched rows
        assert set(out[mask_2]["b_2"]) == a2 - a1
        # check for `None` at unmatched rows
        assert (out[mask_2]["b_1"]).isna().all()

        # check for `values` at unmatched rows
        assert set(out[mask_2]["e_2"]) == set([f"1_{i}" for i in a2 - a1])
        # check for equality at matched rows
        assert (out[mask_2]["e_1"]).isna().all()

    def test_merge_output_column_types(self):
        df1 = DataFrame.from_batch(
            {"a": np.arange(3), "b": ObjectColumn(["1", "2", "3"])}
        )
        df2 = df1.copy()

        out = df1.merge(df2, on="b", how="inner")
        assert isinstance(out["b"], ObjectColumn)

    def test_image_merge(self, tmpdir):
        length = 16
        img_col_test_bed = MockImageColumn(length=length, tmpdir=tmpdir)
        df1 = DataFrame.from_batch(
            {
                "a": np.arange(length),
                "img": img_col_test_bed.col,
            }
        )
        rows = np.arange(4, 8)
        df2 = DataFrame.from_batch(
            {
                "a": rows,
            }
        )

        out = df1.merge(df2, on="a", how="inner")
        assert isinstance(out["img"], FileColumn)
        assert [str(fp) for fp in out["img"].data.args[0]] == [
            os.path.basename(img_col_test_bed.image_paths[row]) for row in rows
        ]

    def test_no_columns(tmpdir):
        length = 16
        df1 = DataFrame.from_batch(
            {
                "a": np.arange(length),
            }
        )
        rows = np.arange(4, 8)
        df2 = DataFrame.from_batch(
            {
                "a": rows,
            }
        )
        out = df1.merge(df2, on="a", how="inner")

        assert "a" in out.columns

    def test_no_columns_in_left(tmpdir):
        length = 16
        df1 = DataFrame.from_batch(
            {
                "a": np.arange(length),
            }
        )
        rows = np.arange(4, 8)
        df2 = DataFrame.from_batch({"a": rows, "b": rows})
        out = df1.merge(df2, on="a", how="inner")

        assert "a" in out.columns
        assert "b" in out.columns

    def test_no_columns_in_right(tmpdir):
        length = 16
        df1 = DataFrame.from_batch(
            {
                "a": np.arange(length),
                "b": np.arange(length),
            }
        )
        rows = np.arange(4, 8)
        df2 = DataFrame.from_batch(
            {
                "a": rows,
            }
        )
        out = df1.merge(df2, on="a", how="inner")

        assert "a" in out.columns
        assert "b" in out.columns

    def test_no_on(self):
        length = 16
        # check dictionary not hashable
        df1 = DataFrame.from_batch(
            {
                "a": ObjectColumn([{"a": 1}] * length),
                "b": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2)

    def test_check_merge_columns(self):
        import meerkat as mk

        length = 16
        # check dictionary not hashable
        df1 = DataFrame.from_batch(
            {
                "a": ObjectColumn([{"a": 1}] * length),
                "b": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2, on=["a"])

        # check multi-on
        with pytest.raises(MergeError):
            df1.merge(df2, on=["a", "b"])

        # check multi-dimensional numpy array
        df1 = DataFrame.from_batch(
            {
                "a": TorchTensorColumn(np.stack([np.arange(5)] * length)),
                "b": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2, on="a")

        # check multi-dimensional numpy array
        df1 = DataFrame.from_batch(
            {
                "a": TorchTensorColumn(torch.stack([torch.arange(5)] * length)),
                "b": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2, on="a")

        # checks that **all** cells are hashable (not just the first)
        df1 = DataFrame.from_batch(
            {
                "a": ObjectColumn(["hello"] + [{"a": 1}] * (length - 1)),
                "b": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2, on="a")

        # checks if Cells in cell columns are NOT hashable
        df1 = DataFrame.from_batch(
            {
                "a": mk.column(["a"] * length).defer(lambda x: x + "b"),
                "b": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2, on="a")

        # checks that having a column called __right_indices__ raises a merge error
        df1 = DataFrame.from_batch(
            {
                "a": ObjectColumn(["hello"] + [{"a": 1}] * (length - 1)),
                "b": list(np.arange(length)),
                "__right_indices__": list(np.arange(length)),
            }
        )
        df2 = df1.copy()
        with pytest.raises(MergeError):
            df1.merge(df2, on="__right_indices__")
