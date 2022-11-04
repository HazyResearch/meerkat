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
from meerkat.columns.lambda_column import LambdaColumn
from meerkat.columns.list_column import ListColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.dataframe import DataFrame

from ..utils import product_parametrize
from .columns.test_arrow_column import ArrowArrayColumnTestBed
from .columns.test_cell_column import CellColumnTestBed
from .columns.test_image_column import ImageColumnTestBed
from .columns.test_numpy_column import NumpyArrayColumnTestBed
from .columns.test_pandas_column import PandasSeriesColumnTestBed
from .columns.test_tensor_column import TensorColumnTestBed


class DataFrameTestBed:

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
        self.df = DataFrame.from_batch(self.columns)

        if consolidated:
            self.df.consolidate()

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
                        **col_config.values[0][
                            1
                        ],  # may need to change this for non parameter set
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

    @classmethod
    @wraps(pytest.fixture)
    def fixture(
        cls, config: dict = None, column_configs: Sequence[Dict] = None, *args, **kwargs
    ):
        params = cls.get_params(
            config=config, column_configs=column_configs, *args, **kwargs
        )
        return pytest.fixture(
            params=params["argvalues"], ids=params["ids"], *args, **kwargs
        )


@DataFrameTestBed.fixture()
def testbed(request, tmpdir):
    config = request.param
    return DataFrameTestBed(**config, tmpdir=tmpdir)


def test_col_index_single(testbed):
    df = testbed.df

    # str index => single column ()
    for name in testbed.columns:
        index = name
        col = df[index]
        assert isinstance(col, AbstractColumn)
        # enforce that a single column index returns a coreference
        assert col is df._data[index]


def test_col_index_multiple(testbed):
    df = testbed.df

    # str index => single column ()
    columns = list(testbed.columns)
    for excluded_column in columns:
        index = [c for c in columns if c != excluded_column]
        new_df = df[index]
        assert isinstance(new_df, DataFrame)

        # enforce that a column index multiple returns a view of the old dataframe
        for col_name in index:
            assert new_df._data[col_name] is df._data[col_name]


#                assert new_df._data[col_name].data is df._data[col_name].data


def test_row_index_single(testbed):
    df = testbed.df

    # int index => single row (dict)
    index = 2
    row = df[index]
    assert isinstance(row, dict)

    for key, value in row.items():
        col_testbed = testbed.column_testbeds[key]
        col_testbed.assert_data_equal(value, col_testbed.get_data(index))


@product_parametrize(
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
def test_row_index_multiple(testbed, index_type):
    df = testbed.df
    rows = np.arange(len(df))

    def convert_to_index_type(index, dtype):
        index = index_type(index)
        if index_type == torch.Tensor:
            return index.to(dtype)
        return index

    # slice index => multiple row selection (DataFrame)
    # tuple or list index => multiple row selection (DataFrame)
    # np.array indeex => multiple row selection (DataFrame)
    for rows, indices in (
        (df[1:3], rows[1:3]),
        (df[[0, 2]], rows[[0, 2]]),
        (
            df[convert_to_index_type(np.array((0,)), dtype=int)],
            rows[np.array((0,))],
        ),
        (
            df[convert_to_index_type(np.array((1, 1)), dtype=int)],
            rows[np.array((1, 1))],
        ),
        (
            df[
                convert_to_index_type(
                    np.array((True, False) * (len(df) // 2)), dtype=bool
                )
            ],
            rows[np.array((True, False) * (len(df) // 2))],
        ),
    ):
        assert isinstance(rows, DataFrame)
        for key, value in rows.items():
            col_testbed = testbed.column_testbeds[key]
            data = col_testbed.get_data(indices)
            col_testbed.assert_data_equal(value.data, data)

            if value.__class__ == df[key].__class__:
                # if the getitem returns a column of the same type, enforce that all
                # attributes were cloned over appropriately. We don't want to check
                # for columns that return columns of different type from getitem
                # (e.g. LambdaColumn)
                assert df[key]._clone(data=data).is_equal(value)


def test_row_lz_index_single(testbed):
    df = testbed.df

    # int index => single row (dict)
    index = 2
    row = df.lz[index]
    assert isinstance(row, dict)

    for key, value in row.items():
        col_testbed = testbed.column_testbeds[key]
        col_testbed.assert_data_equal(
            value, col_testbed.get_data(index, materialize=False)
        )


@product_parametrize(
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
def test_row_lz_index_multiple(testbed, index_type):
    df = testbed.df
    rows = np.arange(len(df))

    def convert_to_index_type(index, dtype):
        index = index_type(index)
        if index_type == torch.Tensor:
            return index.to(dtype)
        return index

    # slice index => multiple row selection (DataFrame)
    # tuple or list index => multiple row selection (DataFrame)
    # np.array indeex => multiple row selection (DataFrame)
    for rows, indices in (
        (df.lz[1:3], rows[1:3]),
        (df.lz[[0, 2]], rows[[0, 2]]),
        (
            df.lz[convert_to_index_type(np.array((0,)), dtype=int)],
            rows[np.array((0,))],
        ),
        (
            df.lz[convert_to_index_type(np.array((1, 1)), dtype=int)],
            rows[np.array((1, 1))],
        ),
        (
            df.lz[
                convert_to_index_type(
                    np.array((True, False) * (len(df) // 2)), dtype=bool
                )
            ],
            rows[np.array((True, False) * (len(df) // 2))],
        ),
    ):
        assert isinstance(rows, DataFrame)
        for key, value in rows.items():
            col_testbed = testbed.column_testbeds[key]
            data = col_testbed.get_data(indices, materialize=False)
            col_testbed.assert_data_equal(value.data, data)

            # if the getitem returns a column of the same type, enforce that all the
            # attributes were cloned over appropriately. We don't want to check this
            # for columns that return columns of different type from getitem
            # (e.g. LambdaColumn)
            if value.__class__ == df[key].__class__:
                assert df[key]._clone(data=data).is_equal(value)


def test_invalid_indices(testbed):
    df = testbed.df
    index = ["nonexistent_column"]
    missing_cols = set(index) - set(df.columns)
    with pytest.raises(
        KeyError, match=f"DataFrame does not have columns {missing_cols}"
    ):
        df[index]

    df = testbed.df
    index = "nonexistent_column"
    with pytest.raises(KeyError, match=f"Column `{index}` does not exist."):
        df[index]

    df = testbed.df
    index = np.zeros((len(df), 10))
    with pytest.raises(
        ValueError, match="Index must have 1 axis, not {}".format(len(index.shape))
    ):
        df[index]

    df = testbed.df
    index = torch.zeros((len(df), 10))
    with pytest.raises(
        ValueError, match="Index must have 1 axis, not {}".format(len(index.shape))
    ):
        df[index]

    df = testbed.df
    index = {"a": 1}
    with pytest.raises(TypeError, match="Invalid index type: {}".format(type(index))):
        df[index]


def test_col_indexing_view_copy_semantics(testbed):
    df = testbed.df

    # Columns (1): Indexing a single column (i.e. with a str) returns the underlying
    # AbstractColumn object directly. In the example below col1 and col2 are
    # coreferences of the same column.
    for name in df.columns:
        df[name] is df[name]

    # Columns (2): Indexing multiple columns (i.e. with Sequence[str]) returns a
    # view of the DataFrame holding views to the columns in the original DataFrame.
    # This means the AbstractColumn objects held in the new DataFrame are the same
    # AbstractColumn objects held in the original DataFrame.
    columns = list(testbed.columns)
    for excluded_column in columns:
        index = [c for c in columns if c != excluded_column]
        view_df = df[index]
        for name in view_df.columns:
            df[name] is view_df[name]
            df[name].data is df[name].data


def test_row_indexing_view_copy_semantics():
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
    df = DataFrame.from_batch(batch)

    # slice index
    df2 = df[:8]
    col = "a"
    assert isinstance(df2[col], NumpyArrayColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    assert df[col].data is df2[col].data.base

    col = "d"
    assert isinstance(df2[col], TensorColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    # note `data_ptr` checks whether the tensors have the same memory address of the
    # first element, so this would not work if the slice didn't start at 0
    assert df[col].data.data_ptr() == df2[col].data.data_ptr()

    col = "e"
    assert isinstance(df2[col], PandasSeriesColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    # TODO (sabri): Figure out pandas copying behavior, it's not clear how it works
    # and this deserves a deeper investigation.
    # assert df[col].data.values.base is df2[col].data.values.base

    # slice index
    df2 = df[np.array([0, 1, 2, 5])]
    col = "a"
    assert isinstance(df2[col], NumpyArrayColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    assert df[col].data.base is not df2[col].data.base

    col = "d"
    assert isinstance(df2[col], TensorColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    # note `data_ptr` checks whether the tensors have the same memory address of the
    # first element, so this would not work if the slice didn't start at 0
    assert df[col].data.data_ptr() != df2[col].data.data_ptr()

    col = "e"
    assert isinstance(df2[col], PandasSeriesColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    assert df[col].data.values.base is not df2[col].data.values.base


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_multiple(
    testbed: DataFrameTestBed, batched: bool, materialize: bool
):
    df = testbed.df
    map_specs = {
        name: col_testbed.get_map_spec(batched=batched, materialize=materialize, salt=1)
        for name, col_testbed in testbed.column_testbeds.items()
    }

    def func(x):
        out = {key: map_spec["fn"](x[key]) for key, map_spec in map_specs.items()}
        return out

    result = df.map(
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
    assert isinstance(result, DataFrame)
    for key, map_spec in map_specs.items():
        assert result[key].is_equal(map_spec["expected_result"])


@DataFrameTestBed.parametrize(
    column_configs={"img": {"testbed_class": ImageColumnTestBed, "n": 2}},
)
@product_parametrize(
    params={"batched": [True, False], "materialize": [True, False]},
)
def test_map_return_multiple_img_only(
    testbed: DataFrameTestBed, batched: bool, materialize: bool
):
    test_map_return_multiple(testbed=testbed, batched=batched, materialize=materialize)


@product_parametrize(
    params={
        "batched": [True, False],
        "materialize": [True, False],
        "num_workers": [0],
        "use_kwargs": [True, False],
    }
)
def test_map_return_single(
    testbed: DataFrameTestBed,
    batched: bool,
    materialize: bool,
    num_workers: int,
    use_kwargs: bool,
):
    df = testbed.df
    kwargs = {"kwarg": 2} if use_kwargs else {}
    name = list(testbed.column_testbeds.keys())[0]
    map_spec = testbed.column_testbeds[name].get_map_spec(
        batched=batched, materialize=materialize, salt=1, **kwargs
    )

    def func(x, kwarg=0):
        out = map_spec["fn"](x[name], k=kwarg)
        return out

    result = df.map(
        func,
        batch_size=4,
        is_batched_fn=batched,
        materialize=materialize,
        num_workers=num_workers,
        **kwargs,
    )
    assert isinstance(result, AbstractColumn)
    assert result.is_equal(map_spec["expected_result"])


@DataFrameTestBed.parametrize(config={"consolidated": [True]})
def test_map_return_single_multi_worker(
    testbed: DataFrameTestBed,
):
    test_map_return_single(
        testbed, batched=True, materialize=True, num_workers=2, use_kwargs=False
    )


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_update_new(testbed: DataFrameTestBed, batched: bool, materialize: bool):
    df = testbed.df
    map_specs = {
        name: col_testbed.get_map_spec(batched=batched, materialize=materialize, salt=1)
        for name, col_testbed in testbed.column_testbeds.items()
    }

    def func(x):
        out = {
            f"{key}_new": map_spec["fn"](x[key]) for key, map_spec in map_specs.items()
        }
        return out

    result = df.update(
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
    assert set(result.columns) == set(df.columns) | {f"{key}_new" for key in df.columns}
    assert isinstance(result, DataFrame)
    for key, map_spec in map_specs.items():
        assert result[f"{key}_new"].is_equal(map_spec["expected_result"])


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_update_existing(
    testbed: DataFrameTestBed, batched: bool, materialize: bool
):
    df = testbed.df
    map_specs = {
        name: col_testbed.get_map_spec(batched=batched, materialize=materialize, salt=1)
        for name, col_testbed in testbed.column_testbeds.items()
    }

    def func(x):
        out = {f"{key}": map_spec["fn"](x[key]) for key, map_spec in map_specs.items()}
        return out

    result = df.update(
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
    assert set(result.columns) == set(df.columns)
    assert result.data is not df.data
    assert isinstance(result, DataFrame)
    for key, map_spec in map_specs.items():
        assert result[key].is_equal(map_spec["expected_result"])


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_filter(testbed: DataFrameTestBed, batched: bool, materialize: bool):
    df = testbed.df
    name = list(testbed.column_testbeds.keys())[0]
    filter_spec = testbed.column_testbeds[name].get_filter_spec(
        batched=batched, materialize=materialize, salt=1
    )

    def func(x):
        out = filter_spec["fn"](x[name])
        return out

    result = df.filter(
        func,
        batch_size=4,
        is_batched_fn=batched,
        materialize=materialize,
    )
    assert isinstance(result, DataFrame)
    result[name].is_equal(filter_spec["expected_result"])


def test_remove_column():
    a = np.arange(16)
    b = np.arange(16) * 2
    df = DataFrame.from_batch({"a": a, "b": b})
    assert "a" in df
    df.remove_column("a")
    assert "a" not in df


def test_overwrite_column():
    # make sure we remove the column when overwriting it
    a = np.arange(16)
    b = np.arange(16) * 2
    df = DataFrame.from_batch({"a": a, "b": b})
    assert "a" in df
    assert df[["a", "b"]]["a"]._data is a
    # testing removal from block manager, so important to use non-blockable type
    df["a"] = ListColumn(range(16))
    assert df[["a", "b"]]["a"]._data is not a
    # check that there are no duplicate columns
    assert set(df.columns) == set(["a", "b"])


@product_parametrize(params={"move": [True, False]})
def test_io(testbed, tmp_path, move):
    """`map`, mixed dataframe, return multiple, `is_batched_fn=True`"""
    df = testbed.df
    path = os.path.join(tmp_path, "test")
    df.write(path)
    if move:
        new_path = os.path.join(tmp_path, "new_test")
        os.rename(path, new_path)
        path = new_path
    new_df = DataFrame.read(path)

    assert isinstance(new_df, DataFrame)
    assert df.columns == new_df.columns
    assert len(new_df) == len(df)
    for name in df.columns:
        # check that the mmap status is preserved across df loads
        assert isinstance(new_df[name], np.memmap) == isinstance(df[name], np.memmap)

        if isinstance(new_df[name], LambdaColumn):
            # the lambda function isn't exactly the same after reading
            new_df[name].data.fn = df[name].data.fn
        if not new_df[name].is_equal(df[name]):
            assert False


def test_repr_html_(testbed):
    testbed.df._repr_html_()


def test_append_columns():
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
    df = DataFrame.from_batch(batch)

    out = df.append(df, axis="rows")

    assert len(out) == len(df) * 2
    assert isinstance(out, DataFrame)
    assert set(out.columns) == set(df.columns)
    assert (out["a"].data == np.concatenate([np.arange(length)] * 2)).all()
    assert out["b"].data == list(np.concatenate([np.arange(length)] * 2))


@product_parametrize(
    params={
        "shuffle": [True, False],
        "batch_size": [1, 4],
        "materialize": [True, False],
    }
)
def test_batch(testbed, shuffle: bool, batch_size: int, materialize: bool):
    df = testbed.df
    df["idx"] = np.arange(len(df))
    order = []
    for batch in df.batch(batch_size=batch_size, shuffle=shuffle):
        order.append(batch["idx"].data)
        for name, col in batch.items():
            if materialize:
                col.is_equal(df[batch["idx"]][name])
            else:
                col.is_equal(df.lz[batch["idx"]][name])
    order = np.array(order).flatten()

    if shuffle:
        assert (order != np.arange(len(df))).any()
    else:
        assert (order == np.arange(len(df))).all()


def test_tail(testbed):
    df = testbed.df

    new_df = df.tail(n=2)

    assert isinstance(new_df, DataFrame)
    assert new_df.columns == df.columns
    assert len(new_df) == 2


def test_head(testbed):
    df = testbed.df

    new_df = df.head(n=2)

    assert isinstance(new_df, DataFrame)
    assert new_df.columns == df.columns
    assert len(new_df) == 2


class DataFrameSubclass(DataFrame):
    """Mock class to test that ops on subclass returns subclass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "subclass"

    def _state_keys(cls) -> Set[str]:
        return super()._state_keys().union({"name"})


def test_subclass():
    df1 = DataFrameSubclass.from_dict({"a": np.arange(3), "b": ["may", "jun", "jul"]})
    df2 = DataFrameSubclass.from_dict(
        {"c": np.arange(3), "d": ["2021", "2022", "2023"]}
    )

    assert isinstance(df1.lz[np.asarray([0, 1])], DataFrameSubclass)
    assert isinstance(df1.lz[:2], DataFrameSubclass)
    assert isinstance(df1[:2], DataFrameSubclass)

    assert isinstance(df1.merge(df2, left_on="a", right_on="c"), DataFrameSubclass)
    assert isinstance(df1.append(df1), DataFrameSubclass)

    assert df1._state_keys() == set(["name"])
    assert df1._get_state() == {"name": "subclass"}


def test_from_csv():
    temp_f = tempfile.NamedTemporaryFile()
    data = {
        "a": [3.4, 2.3, 1.2],
        "b": ["alpha", "beta", "gamma"],
        "c": ["the walk", "the talk", "blah"],
    }
    pd.DataFrame(data).to_csv(temp_f.name)

    df_new = DataFrame.from_csv(temp_f.name)
    assert df_new.columns == ["Unnamed: 0", "a", "b", "c"]
    # Skip index column
    for k in data:
        if isinstance(df_new[k], PandasSeriesColumn):
            data_to_compare = df_new[k]._data.tolist()
        else:
            data_to_compare = df_new[k]._data
        assert data_to_compare == data[k]


def test_from_huggingface(tmpdir: str):
    # Returns a dataset dict
    df = DataFrame.from_huggingface(
        "hf-internal-testing/fixtures_ade20k",
        cache_dir=tmpdir,
    )["test"]
    assert len(df) == 4
    assert len(df.columns) == 2

    # Returns a dataset
    df = DataFrame.from_huggingface(
        "hf-internal-testing/fixtures_ade20k",
        cache_dir=tmpdir,
        split="test",
    )
    assert len(df) == 4
    assert len(df.columns) == 2


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

    df_new = DataFrame.from_jsonl(temp_f.name)
    assert df_new.columns == ["a", "b", "c"]
    # Skip index column
    for k in data:
        if isinstance(df_new[k], NumpyArrayColumn):
            data_to_compare = df_new[k]._data.tolist()
        else:
            data_to_compare = df_new[k]._data
        if k == "d":
            assert data_to_compare == data[k]
        else:
            assert (data_to_compare == np.array(data[k])).all()
    temp_f.close()


def test_from_batch():
    # Build a dataset from a batch
    dataframe = DataFrame.from_batch(
        {
            "a": [1, 2, 3],
            "b": [True, False, True],
            "c": ["x", "y", "z"],
            "d": [{"e": 2}, {"e": 3}, {"e": 4}],
            "e": torch.ones(3),
            "f": np.ones(3),
        },
    )
    assert set(dataframe.columns) == {"a", "b", "c", "d", "e", "f"}
    assert len(dataframe) == 3


def test_from_arrow():
    table = pa.Table.from_arrays(
        [
            pa.array(np.arange(0, 100)),
            pa.array(np.arange(0, 100).astype(float)),
            pa.array(map(str, np.arange(0, 100))),
        ],
        names=["a", "b", "c"],
    )
    df = DataFrame.from_arrow(table)

    # check that the underlying block is the same object as the pyarrow table
    df["a"]._block is table
    df["a"]._block is df["b"]._block
    df["a"]._block is df["c"]._block

    for col in ["a", "b", "c"]:
        assert isinstance(df[col], ArrowArrayColumn)
        assert pa.compute.equal(df[col].data, table[col])


def test_to_pandas():
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
    df = DataFrame.from_batch(batch)

    df_pd = df.to_pandas()
    assert isinstance(df_pd, pd.DataFrame)
    assert all(list(df.columns) == list(df_pd.columns))
    assert len(df) == len(df_pd)

    assert (df_pd["a"].values == df["a"].data).all()
    assert list(df["b"]) == list(df["b"].data)

    assert isinstance(df_pd["c"][0], dict)

    assert (df_pd["d"].values == df["d"].numpy()).all()
    assert (df_pd["e"].values == df["e"].values).all()


def test_to_jsonl(tmpdir: str):
    length = 16
    batch = {
        "a": np.arange(length),
        "b": ListColumn(np.arange(length)),
        "d": torch.arange(length),
        # offset the index to test robustness to nonstandard indices
        "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
        "f": ArrowArrayColumn(np.arange(length)),
    }
    df = DataFrame.from_batch(batch)

    df.to_jsonl(os.path.join(tmpdir, "test.jsonl"))
    df_pd = pd.read_json(
        os.path.join(tmpdir, "test.jsonl"), lines=True, orient="records"
    )

    assert isinstance(df_pd, pd.DataFrame)
    assert list(df_pd.columns) == list(df.columns)
    assert len(df) == len(df_pd)

    assert (df_pd["a"].values == df["a"].data).all()
    assert list(df_pd["b"]) == list(df["b"].data)
    assert (df_pd["d"].values == df["d"].numpy()).all()
    assert (df_pd["e"].values == df["e"].values).all()
    assert (df_pd["f"] == df["f"].to_pandas()).all()


def test_constructor():
    length = 16

    # from dictionary
    data = {
        "a": np.arange(length),
        "b": ListColumn(np.arange(length)),
    }
    df = DataFrame(data=data)
    assert len(df) == length
    assert df["a"].is_equal(NumpyArrayColumn(np.arange(length)))

    # from BlockManager
    mgr = BlockManager.from_dict(data)
    df = DataFrame(data=mgr)
    assert len(df) == length
    assert df["a"].is_equal(NumpyArrayColumn(np.arange(length)))
    assert df.columns == ["a", "b"]

    # from list of dictionaries
    data = [{"a": idx, "b": str(idx), "c": {"test": idx}} for idx in range(length)]
    df = DataFrame(data=data)
    assert len(df) == length
    assert df["a"].is_equal(NumpyArrayColumn(np.arange(length)))
    assert isinstance(df["c"], ListColumn)
    assert df.columns == ["a", "b", "c"]

    # from list of dictionaries, missing values
    data = [
        {"a": idx, "b": str(idx)}
        if (idx % 2 == 0)
        else {"a": idx, "b": str(idx), "c": idx}
        for idx in range(length)
    ]
    df = DataFrame(data=data)
    assert len(df) == length
    assert df["a"].is_equal(NumpyArrayColumn(np.arange(length)))
    assert df["c"].is_equal(
        NumpyArrayColumn([np.nan if idx % 2 == 0 else idx for idx in range(length)])
    )
    assert df.columns == ["a", "b", "c"]

    # from nothing
    df = DataFrame()
    assert len(df) == 0


def test_constructor_w_invalid_data():
    with pytest.raises(
        ValueError,
        match=f"Cannot set DataFrame `data` to object of type {type(5)}.",
    ):
        DataFrame(data=5)


def test_constructor_w_invalid_sequence():
    data = list(range(4))
    with pytest.raises(
        ValueError,
        match="Cannot set DataFrame `data` to a Sequence containing object of "
        f" type {type(data[0])}. Must be a Sequence of Mapping.",
    ):
        DataFrame(data=data)


def test_constructor_w_unequal_lengths():
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
        DataFrame(data=data)


def test_shape():
    length = 16
    data = {
        "a": np.arange(length),
        "b": ListColumn(np.arange(length)),
    }
    df = DataFrame(data)
    assert df.shape == (16, 2)


def test_streamlit(testbed):
    testbed.df.streamlit()


def test_str(testbed):
    result = str(testbed.df)
    assert isinstance(result, str)


def test_repr(testbed):
    result = repr(testbed.df)
    assert isinstance(result, str)


@product_parametrize(params={"max_rows": [6, 16, 20]})
def test_repr_pandas(testbed, max_rows: int):
    meerkat.config.display.max_rows = max_rows
    df, _ = testbed.df._repr_pandas_()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(df), max_rows + 1)
