"""Unittests for Datasets."""
import os
import tempfile
from functools import wraps
from itertools import product
from typing import Dict, Sequence, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import ujson as json

import meerkat
from meerkat import NumpyArrayColumn
from meerkat.block.manager import BlockManager
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.arrow_column import ArrowArrayColumn
from meerkat.columns.list_column import ListColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel

from .columns.test_arrow_column import ArrowArrayColumnTestBed
from .columns.test_cell_column import CellColumnTestBed
from .columns.test_image_column import ImageColumnTestBed
from .columns.test_numpy_column import NumpyArrayColumnTestBed
from .columns.test_pandas_column import PandasSeriesColumnTestBed
from .columns.test_tensor_column import TensorColumnTestBed


class DataPanelTestBed:

    DEFAULT_CONFIG = {
        "consolidated": [True, False],
    }

    DEFAULT_COLUMN_CONFIGS = {
        "np": {"testbed_class": NumpyArrayColumnTestBed, "n": 2},
        "pd": {"testbed_class": PandasSeriesColumnTestBed, "n": 2},
        "torch": {"testbed_class": TensorColumnTestBed, "n": 2},
        "img": {"testbed_class": ImageColumnTestBed, "n": 2},
        "cell": {"testbed_class": CellColumnTestBed, "n": 2},
        "arrow": {"testbed_class": ArrowArrayColumnTestBed, "n": 2},
    }

    def __init__(
        self,
        column_configs: Dict[str, AbstractColumn],
        consolidated: bool = True,
        length: int = 4,
        tmpdir: str = None,
    ):
        self.column_testbeds = self._build_column_testbeds(
            column_configs, length=length, tmpdir=tmpdir
        )

        self.columns = {
            name: testbed.col for name, testbed in self.column_testbeds.items()
        }
        self.dp = DataPanel.from_batch(self.columns)

        if consolidated:
            self.dp.consolidate()

    def _build_column_testbeds(
        self, column_configs: Dict[str, AbstractColumn], length: int, tmpdir: str
    ):
        def _get_tmpdir(name):
            path = os.path.join(tmpdir, name)
            os.makedirs(path)
            return path

        column_testbeds = {}
        for name, config in column_configs.items():
            params = config["testbed_class"].get_params(**config.get("kwargs", {}))
            column_testbeds.update(
                {
                    f"{name}_{col_id}_{idx}": config["testbed_class"](
                        **col_config[1],
                        seed=idx,
                        length=length,
                        tmpdir=_get_tmpdir(f"{name}_{col_id}_{idx}"),
                    )
                    for idx in range(config["n"])
                    for col_config, col_id in zip(params["argvalues"], params["ids"])
                }
            )
        return column_testbeds

    @classmethod
    def get_params(
        cls,
        config: dict = None,
        column_configs: Sequence[Dict] = None,
        params: dict = None,
    ):
        # produce all combinations of the config
        updated_config = cls.DEFAULT_CONFIG.copy()
        if config is not None:
            updated_config.update(config)
        configs = list(
            map(
                dict,
                product(*[[(k, v) for v in vs] for k, vs in updated_config.items()]),
            )
        )

        # add the column_configs to every
        if column_configs is None:
            column_configs = cls.DEFAULT_COLUMN_CONFIGS.copy()
        for config in configs:
            config["column_configs"] = column_configs

        if params is None:
            return {
                "argnames": "testbed",
                "argvalues": configs,
                "ids": [str(config) for config in configs],
            }
        else:
            argvalues = list(product(configs, *params.values()))
            return {
                "argnames": "testbed," + ",".join(params.keys()),
                "argvalues": argvalues,
                "ids": [",".join(map(str, values)) for values in argvalues],
            }

    @classmethod
    @wraps(pytest.mark.parametrize)
    def parametrize(
        cls,
        config: dict = None,
        column_configs: Sequence[Dict] = None,
        params: dict = None,
    ):
        return pytest.mark.parametrize(
            **cls.get_params(
                config=config, params=params, column_configs=column_configs
            ),
            indirect=["testbed"],
        )


@pytest.fixture
def testbed(request, tmpdir):
    config = request.param
    return DataPanelTestBed(**config, tmpdir=tmpdir)


class TestDataPanel:

    testbed_class: type = DataPanelTestBed
    dp_class: type = DataPanel

    @DataPanelTestBed.parametrize()
    def test_col_index_single(self, testbed):
        dp = testbed.dp

        # str index => single column ()
        for name in testbed.columns:
            index = name
            col = dp[index]
            assert isinstance(col, AbstractColumn)
            # enforce that a single column index returns a coreference
            assert col is dp._data[index]

    @DataPanelTestBed.parametrize()
    def test_col_index_multiple(self, testbed):
        dp = testbed.dp

        # str index => single column ()
        columns = list(testbed.columns)
        for excluded_column in columns:
            index = [c for c in columns if c != excluded_column]
            new_dp = dp[index]
            assert isinstance(new_dp, DataPanel)

            # enforce that a column index multiple returns a view of the old datapanel
            for col_name in index:
                assert new_dp._data[col_name] is not dp._data[col_name]

    #                assert new_dp._data[col_name].data is dp._data[col_name].data

    @DataPanelTestBed.parametrize()
    def test_row_index_single(self, testbed):
        dp = testbed.dp

        # int index => single row (dict)
        index = 2
        row = dp[index]
        assert isinstance(row, dict)

        for key, value in row.items():
            col_testbed = testbed.column_testbeds[key]
            col_testbed.assert_data_equal(value, col_testbed.get_data(index))

    @DataPanelTestBed.parametrize(
        params={
            "index_type": [
                np.array,
                pd.Series,
                torch.Tensor,
                NumpyArrayColumn,
                PandasSeriesColumn,
                TensorColumn,
                list,
            ]
        }
    )
    def test_row_index_multiple(self, testbed, index_type):
        dp = testbed.dp
        rows = np.arange(len(dp))

        def convert_to_index_type(index, dtype):
            index = index_type(index)
            if index_type == torch.Tensor:
                return index.to(dtype)
            return index

        # slice index => multiple row selection (DataPanel)
        # tuple or list index => multiple row selection (DataPanel)
        # np.array indeex => multiple row selection (DataPanel)
        for rows, indices in (
            (dp[1:3], rows[1:3]),
            (dp[[0, 2]], rows[[0, 2]]),
            (
                dp[convert_to_index_type(np.array((0,)), dtype=int)],
                rows[np.array((0,))],
            ),
            (
                dp[convert_to_index_type(np.array((1, 1)), dtype=int)],
                rows[np.array((1, 1))],
            ),
            (
                dp[
                    convert_to_index_type(
                        np.array((True, False) * (len(dp) // 2)), dtype=bool
                    )
                ],
                rows[np.array((True, False) * (len(dp) // 2))],
            ),
        ):
            assert isinstance(rows, DataPanel)
            for key, value in rows.items():
                col_testbed = testbed.column_testbeds[key]
                data = col_testbed.get_data(indices)
                col_testbed.assert_data_equal(value.data, data)

                if value.__class__ == dp[key].__class__:
                    # if the getitem returns a column of the same type, enforce that all
                    # attributes were cloned over appropriately. We don't want to check
                    # for columns that return columns of different type from getitem
                    # (e.g. LambdaColumn)
                    assert dp[key]._clone(data=data).is_equal(value)

    @DataPanelTestBed.parametrize()
    def test_row_lz_index_single(self, testbed):
        dp = testbed.dp

        # int index => single row (dict)
        index = 2
        row = dp.lz[index]
        assert isinstance(row, dict)

        for key, value in row.items():
            col_testbed = testbed.column_testbeds[key]
            col_testbed.assert_data_equal(
                value, col_testbed.get_data(index, materialize=False)
            )

    @DataPanelTestBed.parametrize(
        params={
            "index_type": [
                np.array,
                pd.Series,
                torch.Tensor,
                NumpyArrayColumn,
                PandasSeriesColumn,
                TensorColumn,
            ]
        }
    )
    def test_row_lz_index_multiple(self, testbed, index_type):
        dp = testbed.dp
        rows = np.arange(len(dp))

        def convert_to_index_type(index, dtype):
            index = index_type(index)
            if index_type == torch.Tensor:
                return index.to(dtype)
            return index

        # slice index => multiple row selection (DataPanel)
        # tuple or list index => multiple row selection (DataPanel)
        # np.array indeex => multiple row selection (DataPanel)
        for rows, indices in (
            (dp.lz[1:3], rows[1:3]),
            (dp.lz[[0, 2]], rows[[0, 2]]),
            (
                dp.lz[convert_to_index_type(np.array((0,)), dtype=int)],
                rows[np.array((0,))],
            ),
            (
                dp.lz[convert_to_index_type(np.array((1, 1)), dtype=int)],
                rows[np.array((1, 1))],
            ),
            (
                dp.lz[
                    convert_to_index_type(
                        np.array((True, False) * (len(dp) // 2)), dtype=bool
                    )
                ],
                rows[np.array((True, False) * (len(dp) // 2))],
            ),
        ):
            assert isinstance(rows, DataPanel)
            for key, value in rows.items():
                col_testbed = testbed.column_testbeds[key]
                data = col_testbed.get_data(indices, materialize=False)
                col_testbed.assert_data_equal(value.data, data)

                # if the getitem returns a column of the same type, enforce that all the
                # attributes were cloned over appropriately. We don't want to check this
                # for columns that return columns of different type from getitem
                # (e.g. LambdaColumn)
                if value.__class__ == dp[key].__class__:
                    assert dp[key]._clone(data=data).is_equal(value)

    @DataPanelTestBed.parametrize()
    def test_invalid_indices(self, testbed):
        dp = testbed.dp
        index = ["nonexistent_column"]
        missing_cols = set(index) - set(dp.columns)
        with pytest.raises(
            KeyError, match=f"DataPanel does not have columns {missing_cols}"
        ):
            dp[index]

        dp = testbed.dp
        index = "nonexistent_column"
        with pytest.raises(KeyError, match=f"Column `{index}` does not exist."):
            dp[index]

        dp = testbed.dp
        index = np.zeros((len(dp), 10))
        with pytest.raises(
            ValueError, match="Index must have 1 axis, not {}".format(len(index.shape))
        ):
            dp[index]

        dp = testbed.dp
        index = torch.zeros((len(dp), 10))
        with pytest.raises(
            ValueError, match="Index must have 1 axis, not {}".format(len(index.shape))
        ):
            dp[index]

        dp = testbed.dp
        index = {"a": 1}
        with pytest.raises(
            TypeError, match="Invalid index type: {}".format(type(index))
        ):
            dp[index]

    @DataPanelTestBed.parametrize()
    def test_col_indexing_view_copy_semantics(self, testbed):
        dp = testbed.dp

        # Columns (1): Indexing a single column (i.e. with a str) returns the underlying
        # AbstractColumn object directly. In the example below col1 and col2 are
        # coreferences of the same column.
        for name in dp.columns:
            dp[name] is dp[name]

        # Columns (2): Indexing multiple columns (i.e. with Sequence[str]) returns a
        # view of the DataPanel holding views to the columns in the original DataPanel.
        # This means the AbstractColumn objects held in the new DataPanel are the same
        # AbstractColumn objects held in the original DataPanel.
        columns = list(testbed.columns)
        for excluded_column in columns:
            index = [c for c in columns if c != excluded_column]
            view_dp = dp[index]
            for name in view_dp.columns:
                dp[name] is not view_dp[name]
                dp[name].data is dp[name].data

    def test_row_indexing_view_copy_semantics(self):
        length = 16
        batch = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length)),
            "c": [{"a": 2}] * length,
            "d": torch.arange(length),
            # offset the index to test robustness to nonstandard indices
            "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
            # test multidimensional
            "f": np.ones((length, 5)).astype(int),
            "g": torch.ones(length, 5).to(int),
        }
        dp = DataPanel.from_batch(batch)

        # slice index
        dp2 = dp[:8]
        col = "a"
        assert isinstance(dp2[col], NumpyArrayColumn)
        assert dp[col] is not dp2[col]
        assert dp[col].data is not dp2[col].data
        assert dp[col].data.base is dp2[col].data.base

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
        # TODO (sabri): Figure out pandas copying behavior, it's not clear how it works
        # and this deserves a deeper investigation.
        # assert dp[col].data.values.base is dp2[col].data.values.base

        # slice index
        dp2 = dp[np.array([0, 1, 2, 5])]
        col = "a"
        assert isinstance(dp2[col], NumpyArrayColumn)
        assert dp[col] is not dp2[col]
        assert dp[col].data is not dp2[col].data
        assert dp[col].data.base is not dp2[col].data.base

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
        assert dp[col].data.values.base is not dp2[col].data.values.base

    @DataPanelTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_map_return_multiple(
        self, testbed: DataPanelTestBed, batched: bool, materialize: bool
    ):
        dp = testbed.dp
        map_specs = {
            name: col_testbed.get_map_spec(
                batched=batched, materialize=materialize, salt=1
            )
            for name, col_testbed in testbed.column_testbeds.items()
        }

        def func(x):
            out = {key: map_spec["fn"](x[key]) for key, map_spec in map_specs.items()}
            return out

        result = dp.map(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            output_type={
                key: map_spec["output_type"]
                for key, map_spec in map_specs.items()
                if "output_type" in map_spec
            },
        )
        assert isinstance(result, DataPanel)
        for key, map_spec in map_specs.items():
            assert result[key].is_equal(map_spec["expected_result"])

    @DataPanelTestBed.parametrize(
        column_configs={"img": {"testbed_class": ImageColumnTestBed, "n": 2}},
        params={"batched": [True, False], "materialize": [True, False]},
    )
    def test_map_return_multiple_img_only(
        self, testbed: DataPanelTestBed, batched: bool, materialize: bool
    ):
        self.test_map_return_multiple(
            testbed=testbed, batched=batched, materialize=materialize
        )

    @DataPanelTestBed.parametrize(
        params={
            "batched": [True, False],
            "materialize": [True, False],
            "num_workers": [0],
            "use_kwargs": [True, False],
        }
    )
    def test_map_return_single(
        self,
        testbed: DataPanelTestBed,
        batched: bool,
        materialize: bool,
        num_workers: int,
        use_kwargs: bool,
    ):
        dp = testbed.dp
        kwargs = {"kwarg": 2} if use_kwargs else {}
        name = list(testbed.column_testbeds.keys())[0]
        map_spec = testbed.column_testbeds[name].get_map_spec(
            batched=batched, materialize=materialize, salt=1, **kwargs
        )

        def func(x, kwarg=0):
            out = map_spec["fn"](x[name], k=kwarg)
            return out

        result = dp.map(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            num_workers=num_workers,
            **kwargs,
        )
        assert isinstance(result, AbstractColumn)
        assert result.is_equal(map_spec["expected_result"])

    @DataPanelTestBed.parametrize(config={"consolidated": [True]})
    def test_map_return_single_multi_worker(
        self,
        testbed: DataPanelTestBed,
    ):
        self.test_map_return_single(
            testbed, batched=True, materialize=True, num_workers=2, use_kwargs=False
        )

    @DataPanelTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_map_update_new(
        self, testbed: DataPanelTestBed, batched: bool, materialize: bool
    ):
        dp = testbed.dp
        map_specs = {
            name: col_testbed.get_map_spec(
                batched=batched, materialize=materialize, salt=1
            )
            for name, col_testbed in testbed.column_testbeds.items()
        }

        def func(x):
            out = {
                f"{key}_new": map_spec["fn"](x[key])
                for key, map_spec in map_specs.items()
            }
            return out

        result = dp.update(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            output_type={
                f"{key}_new": map_spec["output_type"]
                for key, map_spec in map_specs.items()
                if "output_type" in map_spec
            },
        )
        assert set(result.columns) == set(dp.columns) | {
            f"{key}_new" for key in dp.columns
        }
        assert isinstance(result, DataPanel)
        for key, map_spec in map_specs.items():
            assert result[f"{key}_new"].is_equal(map_spec["expected_result"])

    @DataPanelTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_map_update_existing(
        self, testbed: DataPanelTestBed, batched: bool, materialize: bool
    ):
        dp = testbed.dp
        map_specs = {
            name: col_testbed.get_map_spec(
                batched=batched, materialize=materialize, salt=1
            )
            for name, col_testbed in testbed.column_testbeds.items()
        }

        def func(x):
            out = {
                f"{key}": map_spec["fn"](x[key]) for key, map_spec in map_specs.items()
            }
            return out

        result = dp.update(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            output_type={
                key: map_spec["output_type"]
                for key, map_spec in map_specs.items()
                if "output_type" in map_spec
            },
        )
        assert set(result.columns) == set(dp.columns)
        assert result.data is not dp.data
        assert isinstance(result, DataPanel)
        for key, map_spec in map_specs.items():
            assert result[key].is_equal(map_spec["expected_result"])

    @DataPanelTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_filter(self, testbed: DataPanelTestBed, batched: bool, materialize: bool):
        dp = testbed.dp
        name = list(testbed.column_testbeds.keys())[0]
        filter_spec = testbed.column_testbeds[name].get_filter_spec(
            batched=batched, materialize=materialize, salt=1
        )

        def func(x):
            out = filter_spec["fn"](x[name])
            return out

        result = dp.filter(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
        )
        assert isinstance(result, DataPanel)
        result[name].is_equal(filter_spec["expected_result"])

    def test_remove_column(self):
        a = np.arange(16)
        b = np.arange(16) * 2
        dp = DataPanel.from_batch({"a": a, "b": b})
        assert "a" in dp
        dp.remove_column("a")
        assert "a" not in dp

    def test_overwrite_column(self):
        # make sure we remove the column when overwriting it
        a = np.arange(16)
        b = np.arange(16) * 2
        dp = DataPanel.from_batch({"a": a, "b": b})
        assert "a" in dp
        assert dp[["a", "b"]]["a"]._data.base is a
        # testing removal from block manager, so important to use non-blockable type
        dp["a"] = ListColumn(range(16))
        assert dp[["a", "b"]]["a"]._data is not a
        # check that there are no duplicate columns
        assert set(dp.columns) == set(["a", "b"])

    @DataPanelTestBed.parametrize(params={"move": [True, False]})
    def test_io(self, testbed, tmp_path, move):
        """`map`, mixed datapanel, return multiple, `is_batched_fn=True`"""
        dp = testbed.dp
        path = os.path.join(tmp_path, "test")
        dp.write(path)
        if move:
            new_path = os.path.join(tmp_path, "new_test")
            os.rename(path, new_path)
            path = new_path
        new_dp = DataPanel.read(path)

        assert isinstance(new_dp, DataPanel)
        assert dp.columns == new_dp.columns
        assert len(new_dp) == len(dp)
        for name in dp.columns:
            # check that the mmap status is preserved across dp loads
            assert isinstance(new_dp[name], np.memmap) == isinstance(
                dp[name], np.memmap
            )
            assert new_dp[name].is_equal(dp[name])

    @DataPanelTestBed.parametrize()
    def test_repr_html_(self, testbed):
        testbed.dp._repr_html_()

    def test_append_columns(self):
        length = 16
        batch = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length)),
            "c": [{"a": 2}] * length,
            "d": torch.arange(length),
            # offset the index to test robustness to nonstandard indices
            "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
            # test multidimensional
            "f": np.ones((length, 5)).astype(int),
            "g": torch.ones(length, 5).to(int),
        }
        dp = DataPanel.from_batch(batch)

        out = dp.append(dp, axis="rows")

        assert len(out) == len(dp) * 2
        assert isinstance(out, DataPanel)
        assert set(out.columns) == set(dp.columns)
        assert (out["a"].data == np.concatenate([np.arange(length)] * 2)).all()
        assert out["b"].data == list(np.concatenate([np.arange(length)] * 2))

    @DataPanelTestBed.parametrize(
        params={
            "shuffle": [True, False],
            "batch_size": [1, 4],
            "materialize": [True, False],
        }
    )
    def test_batch(self, testbed, shuffle: bool, batch_size: int, materialize: bool):
        dp = testbed.dp
        dp["idx"] = np.arange(len(dp))
        order = []
        for batch in dp.batch(batch_size=batch_size, shuffle=shuffle):
            order.append(batch["idx"].data)
            for name, col in batch.items():
                if materialize:
                    col.is_equal(dp[batch["idx"]][name])
                else:
                    col.is_equal(dp.lz[batch["idx"]][name])
        order = np.array(order).flatten()

        if shuffle:
            assert (order != np.arange(len(dp))).any()
        else:
            assert (order == np.arange(len(dp))).all()

    @DataPanelTestBed.parametrize()
    def test_tail(self, testbed):
        dp = testbed.dp

        new_dp = dp.tail(n=2)

        assert isinstance(new_dp, DataPanel)
        assert new_dp.columns == dp.columns
        assert len(new_dp) == 2

    @DataPanelTestBed.parametrize()
    def test_head(self, testbed):
        dp = testbed.dp

        new_dp = dp.head(n=2)

        assert isinstance(new_dp, DataPanel)
        assert new_dp.columns == dp.columns
        assert len(new_dp) == 2

    class DataPanelSubclass(DataPanel):
        """Mock class to test that ops on subclass returns subclass."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.name = "subclass"

        def _state_keys(cls) -> Set[str]:
            return super()._state_keys().union({"name"})

    def test_subclass(self):
        dp1 = self.DataPanelSubclass.from_dict(
            {"a": np.arange(3), "b": ["may", "jun", "jul"]}
        )
        dp2 = self.DataPanelSubclass.from_dict(
            {"c": np.arange(3), "d": ["2021", "2022", "2023"]}
        )

        assert isinstance(dp1.lz[np.asarray([0, 1])], self.DataPanelSubclass)
        assert isinstance(dp1.lz[:2], self.DataPanelSubclass)
        assert isinstance(dp1[:2], self.DataPanelSubclass)

        assert isinstance(
            dp1.merge(dp2, left_on="a", right_on="c"), self.DataPanelSubclass
        )
        assert isinstance(dp1.append(dp1), self.DataPanelSubclass)

        assert dp1._state_keys() == set(["name"])
        assert dp1._get_state() == {"name": "subclass"}

    def test_from_csv(self):
        temp_f = tempfile.NamedTemporaryFile()
        data = {
            "a": [3.4, 2.3, 1.2],
            "b": ["alpha", "beta", "gamma"],
            "c": ["the walk", "the talk", "blah"],
        }
        pd.DataFrame(data).to_csv(temp_f.name)

        dp_new = DataPanel.from_csv(temp_f.name)
        assert dp_new.columns == ["Unnamed: 0", "a", "b", "c"]
        # Skip index column
        for k in data:
            if isinstance(dp_new[k], PandasSeriesColumn):
                data_to_compare = dp_new[k]._data.tolist()
            else:
                data_to_compare = dp_new[k]._data
            assert data_to_compare == data[k]

    def test_from_huggingface(self, tmpdir: str):
        # Returns a dataset dict
        dp = DataPanel.from_huggingface(
            "hf-internal-testing/fixtures_ade20k",
            cache_dir=tmpdir,
        )["test"]
        assert len(dp) == 4
        assert len(dp.columns) == 2

        # Returns a dataset
        dp = DataPanel.from_huggingface(
            "hf-internal-testing/fixtures_ade20k",
            cache_dir=tmpdir,
            split="test",
        )
        assert len(dp) == 4
        assert len(dp.columns) == 2

    def test_from_jsonl(self):
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
        assert dp_new.columns == ["a", "b", "c"]
        # Skip index column
        for k in data:
            if isinstance(dp_new[k], NumpyArrayColumn):
                data_to_compare = dp_new[k]._data.tolist()
            else:
                data_to_compare = dp_new[k]._data
            if k == "d":
                assert data_to_compare == data[k]
            else:
                assert (data_to_compare == np.array(data[k])).all()
        temp_f.close()

    def test_from_batch(self):
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
        assert set(datapanel.columns) == {"a", "b", "c", "d", "e", "f"}
        assert len(datapanel) == 3

    def test_from_arrow(self):
        table = pa.Table.from_arrays(
            [
                pa.array(np.arange(0, 100)),
                pa.array(np.arange(0, 100).astype(float)),
                pa.array(map(str, np.arange(0, 100))),
            ],
            names=["a", "b", "c"],
        )
        dp = DataPanel.from_arrow(table)

        # check that the underlying block is the same object as the pyarrow table
        dp["a"]._block is table
        dp["a"]._block is dp["b"]._block
        dp["a"]._block is dp["c"]._block

        for col in ["a", "b", "c"]:
            assert isinstance(dp[col], ArrowArrayColumn)
            assert pa.compute.equal(dp[col].data, table[col])

    def test_to_pandas(self):
        import pandas as pd

        length = 16
        batch = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length)),
            "c": [{"a": 2}] * length,
            "d": torch.arange(length),
            # offset the index to test robustness to nonstandard indices
            "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
            # test multidimensional
            "f": np.ones((length, 5)).astype(int),
            "g": torch.ones(length, 5).to(int),
        }
        dp = DataPanel.from_batch(batch)

        df = dp.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == dp.columns
        assert len(df) == len(dp)

        assert (df["a"].values == dp["a"].data).all()
        assert list(df["b"]) == list(dp["b"].data)

        assert isinstance(df["c"][0], dict)

        assert (df["d"].values == dp["d"].numpy()).all()
        assert (df["e"].values == dp["e"].values).all()

    def test_to_jsonl(self, tmpdir: str):
        length = 16
        batch = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length)),
            "d": torch.arange(length),
            # offset the index to test robustness to nonstandard indices
            "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
            "f": ArrowArrayColumn(np.arange(length)),
        }
        dp = DataPanel.from_batch(batch)

        dp.to_jsonl(os.path.join(tmpdir, "test.jsonl"))
        df = pd.read_json(
            os.path.join(tmpdir, "test.jsonl"), lines=True, orient="records"
        )

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == dp.columns
        assert len(df) == len(dp)

        assert (df["a"].values == dp["a"].data).all()
        assert list(df["b"]) == list(dp["b"].data)
        assert (df["d"].values == dp["d"].numpy()).all()
        assert (df["e"].values == dp["e"].values).all()
        assert (df["f"] == dp["f"].to_pandas()).all()

    def test_constructor(self):
        length = 16

        # from dictionary
        data = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length)),
        }
        dp = DataPanel(data=data)
        assert len(dp) == length
        assert dp["a"].is_equal(NumpyArrayColumn(np.arange(length)))

        # from BlockManager
        mgr = BlockManager.from_dict(data)
        dp = DataPanel(data=mgr)
        assert len(dp) == length
        assert dp["a"].is_equal(NumpyArrayColumn(np.arange(length)))
        assert dp.columns == ["a", "b"]

        # from list of dictionaries
        data = [{"a": idx, "b": str(idx), "c": {"test": idx}} for idx in range(length)]
        dp = DataPanel(data=data)
        assert len(dp) == length
        assert dp["a"].is_equal(NumpyArrayColumn(np.arange(length)))
        assert isinstance(dp["c"], ListColumn)
        assert dp.columns == ["a", "b", "c"]

        # from list of dictionaries, missing values
        data = [
            {"a": idx, "b": str(idx)}
            if (idx % 2 == 0)
            else {"a": idx, "b": str(idx), "c": idx}
            for idx in range(length)
        ]
        dp = DataPanel(data=data)
        assert len(dp) == length
        assert dp["a"].is_equal(NumpyArrayColumn(np.arange(length)))
        assert dp["c"].is_equal(
            NumpyArrayColumn([np.nan if idx % 2 == 0 else idx for idx in range(length)])
        )
        assert dp.columns == ["a", "b", "c"]

        # from nothing
        dp = DataPanel()
        assert len(dp) == 0

    def test_constructor_w_invalid_data(self):
        with pytest.raises(
            ValueError,
            match=f"Cannot set DataPanel `data` to object of type {type(5)}.",
        ):
            DataPanel(data=5)

    def test_constructor_w_invalid_sequence(self):
        data = list(range(4))
        with pytest.raises(
            ValueError,
            match="Cannot set DataPanel `data` to a Sequence containing object of "
            f" type {type(data[0])}. Must be a Sequence of Mapping.",
        ):
            DataPanel(data=data)

    def test_constructor_w_unequal_lengths(self):
        length = 16
        data = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length - 1)),
        }
        with pytest.raises(
            ValueError,
            match=(
                f"Cannot add column 'b' with length {length - 1} to `BlockManager` "
                f" with length {length} columns."
            ),
        ):
            DataPanel(data=data)

    def test_shape(self):
        length = 16
        data = {
            "a": np.arange(length),
            "b": ListColumn(np.arange(length)),
        }
        dp = DataPanel(data)
        assert dp.shape == (16, 2)

    @DataPanelTestBed.parametrize()
    def test_streamlit(self, testbed):
        testbed.dp.streamlit()

    @DataPanelTestBed.parametrize()
    def test_str(self, testbed):
        result = str(testbed.dp)
        assert isinstance(result, str)

    @DataPanelTestBed.parametrize()
    def test_repr(self, testbed):
        result = repr(testbed.dp)
        assert isinstance(result, str)

    @DataPanelTestBed.parametrize(params={"max_rows": [6, 16, 20]})
    def test_repr_pandas(self, testbed, max_rows: int):
        meerkat.config.display.max_rows = max_rows
        df, _ = testbed.dp._repr_pandas_()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == min(len(df), max_rows + 1)
