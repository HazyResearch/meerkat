"""Unittests for Datasets."""
import os
import tempfile
import warnings
from functools import wraps
from itertools import product
from typing import Dict, Sequence, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import ujson as json

import meerkat as mk
from meerkat.block.manager import BlockManager
from meerkat.columns.abstract import Column
from meerkat.columns.deferred.base import DeferredColumn
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.scalar import ScalarColumn
from meerkat.columns.scalar.arrow import ArrowScalarColumn
from meerkat.columns.tensor.abstract import TensorColumn
from meerkat.columns.tensor.numpy import NumPyTensorColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.dataframe import DataFrame
from meerkat.interactive.graph.operation import Operation
from meerkat.interactive.node import NodeMixin
from meerkat.row import Row

from ..utils import product_parametrize
from .columns.deferred.test_image import ImageColumnTestBed
from .columns.scalar.test_arrow import ArrowScalarColumnTestBed
from .columns.scalar.test_pandas import PandasScalarColumnTestBed
from .columns.tensor.test_numpy import NumPyTensorColumnTestBed
from .columns.tensor.test_torch import TorchTensorColumnTestBed


class DataFrameTestBed:
    DEFAULT_CONFIG = {
        "consolidated": [True, False],
    }

    DEFAULT_COLUMN_CONFIGS = {
        "np": {"testbed_class": NumPyTensorColumnTestBed, "n": 2},
        "pd": {"testbed_class": PandasScalarColumnTestBed, "n": 2},
        "torch": {"testbed_class": TorchTensorColumnTestBed, "n": 2},
        "img": {"testbed_class": ImageColumnTestBed, "n": 2},
        "arrow": {"testbed_class": ArrowScalarColumnTestBed, "n": 2},
    }

    def __init__(
        self,
        column_configs: Dict[str, Column],
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
        self, column_configs: Dict[str, Column], length: int, tmpdir: str
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

            def _repr_value(value):
                if isinstance(value, type):
                    return value.__name__
                return str(value)

            argvalues = list(product(configs, *params.values()))
            return {
                "argnames": "testbed," + ",".join(params.keys()),
                "argvalues": argvalues,
                "ids": [",".join(map(_repr_value, values)) for values in argvalues],
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
        assert isinstance(col, Column)
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


def test_row_index_single(testbed):
    df = testbed.df

    # int index => single row (dict)
    index = 2
    row = df[index]
    assert isinstance(row, Row)

    for key, value in row().items():
        col_testbed = testbed.column_testbeds[key]
        col_testbed.assert_data_equal(value, col_testbed.get_data(index))


@product_parametrize(
    params={
        "index_type": [
            np.array,
            pd.Series,
            torch.Tensor,
            NumPyTensorColumn,
            ScalarColumn,
            TorchTensorColumn,
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
        rows = rows()
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
    row = df[index]
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
            TorchTensorColumn,
            ScalarColumn,
            TorchTensorColumn,
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
        "a": NumPyTensorColumn(np.arange(length)),
        "b": ObjectColumn(np.arange(length)),
        "c": [{"a": 2}] * length,
        "d": TorchTensorColumn(torch.arange(length)),
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
    assert isinstance(df2[col], NumPyTensorColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    assert df[col].data.base is df2[col].data.base

    col = "d"
    assert isinstance(df2[col], TorchTensorColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    # note `data_ptr` checks whether the tensors have the same memory address of the
    # first element, so this would not work if the slice didn't start at 0
    assert df[col].data.data_ptr() == df2[col].data.data_ptr()

    col = "e"
    assert isinstance(df2[col], ScalarColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    # TODO (sabri): Figure out pandas copying behavior, it's not clear how it works
    # and this deserves a deeper investigation.
    # assert df[col].data.values.base is df2[col].data.values.base

    # slice index
    df2 = df[np.array([0, 1, 2, 5])]
    col = "a"
    assert isinstance(df2[col], NumPyTensorColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    assert df[col].data.base is not df2[col].data.base

    col = "d"
    assert isinstance(df2[col], TorchTensorColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    # note `data_ptr` checks whether the tensors have the same memory address of the
    # first element, so this would not work if the slice didn't start at 0
    assert df[col].data.data_ptr() != df2[col].data.data_ptr()

    col = "e"
    assert isinstance(df2[col], ScalarColumn)
    assert df[col] is not df2[col]
    assert df[col].data is not df2[col].data
    assert df[col].data.values.base is not df2[col].data.values.base


# @product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
# def test_map_return_multiple(
#     testbed: DataFrameTestBed, batched: bool, materialize: bool
# ):
#     df = testbed.df
#     map_specs = {
#         name: col_testbed.get_map_spec(batched=batched,
#           materialize=materialize, salt=1)
#         for name, col_testbed in testbed.column_testbeds.items()
#     }

#     def func(x):
#         out = {key: map_spec["fn"](x[key]) for key, map_spec in map_specs.items()}
#         return out

#     result = df.map(
#         func,
#         batch_size=4,
#         is_batched_fn=batched,
#         materialize=materialize,
#         output_type={
#             key: map_spec["output_type"]
#             for key, map_spec in map_specs.items()
#             if "output_type" in map_spec
#         },
#     )
#     assert isinstance(result, DataFrame)
#     for key, map_spec in map_specs.items():
#         assert result[key].is_equal(map_spec["expected_result"])


# @DataFrameTestBed.parametrize(
#     column_configs={"img": {"testbed_class": ImageColumnTestBed, "n": 2}},
# )
# @product_parametrize(
#     params={"batched": [True, False], "materialize": [True, False]},
# )
# def test_map_return_multiple_img_only(
#     testbed: DataFrameTestBed, batched: bool, materialize: bool
# ):
#     test_map_return_multiple(testbed=testbed, batched=batched,
#       materialize=materialize)


# @product_parametrize(
#     params={
#         "batched": [True, False],
#         "materialize": [True, False],
#         "num_workers": [0],
#         "use_kwargs": [True, False],
#     }
# )
# def test_map_return_single(
#     testbed: DataFrameTestBed,
#     batched: bool,
#     materialize: bool,
#     num_workers: int,
#     use_kwargs: bool,
# ):
#     df = testbed.df
#     kwargs = {"kwarg": 2} if use_kwargs else {}
#     name = list(testbed.column_testbeds.keys())[0]
#     map_spec = testbed.column_testbeds[name].get_map_spec(
#         batched=batched, materialize=materialize, salt=1, **kwargs
#     )

#     def func(x, kwarg=0):
#         out = map_spec["fn"](x[name], k=kwarg)
#         return out

#     result = df.map(
#         func,
#         batch_size=4,
#         is_batched_fn=batched,
#         materialize=materialize,
#         num_workers=num_workers,
#         **kwargs,
#     )
#     assert isinstance(result, Column)
#     # FIXME(Sabri):  put this back after implementing map
#     # assert result.is_equal(map_spec["expected_result"])


# @DataFrameTestBed.parametrize(config={"consolidated": [True]})
# def test_map_return_single_multi_worker(
#     testbed: DataFrameTestBed,
# ):
#     test_map_return_single(
#         testbed, batched=True, materialize=True, num_workers=2, use_kwargs=False
#     )


# @product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
# def test_map_update_new(testbed: DataFrameTestBed, batched: bool, materialize: bool):
#     df = testbed.df
#     map_specs = {
#         name: col_testbed.get_map_spec(batched=batched,
#           materialize=materialize, salt=1)
#         for name, col_testbed in testbed.column_testbeds.items()
#     }

#     def func(x):
#         out = {
#             f"{key}_new": map_spec["fn"](x[key])
#               for key, map_spec in map_specs.items()
#         }
#         return out

#     result = df.update(
#         func,
#         batch_size=4,
#         is_batched_fn=batched,
#         materialize=materialize,
#         output_type={
#             f"{key}_new": map_spec["output_type"]
#             for key, map_spec in map_specs.items()
#             if "output_type" in map_spec
#         },
#     )
#     assert set(result.columns) == set(df.columns) |
#           {f"{key}_new" for key in df.columns}
#     assert isinstance(result, DataFrame)
#     for key, map_spec in map_specs.items():
#         assert result[f"{key}_new"].is_equal(map_spec["expected_result"])


# @product_parametrize(params={"batched": [True, False],
#               "materialize": [True, False]})
# def test_map_update_existing(
#     testbed: DataFrameTestBed, batched: bool, materialize: bool
# ):
#     df = testbed.df
#     map_specs = {
#         name: col_testbed.get_map_spec(batched=batched,
#           materialize=materialize, salt=1)
#         for name, col_testbed in testbed.column_testbeds.items()
#     }

#     def func(x):
#         out = {f"{key}": map_spec["fn"](x[key])
#           for key, map_spec in map_specs.items()}
#         return out

#     result = df.update(
#         func,
#         batch_size=4,
#         is_batched_fn=batched,
#         materialize=materialize,
#         output_type={
#             key: map_spec["output_type"]
#             for key, map_spec in map_specs.items()
#             if "output_type" in map_spec
#         },
#     )
#     assert set(result.columns) == set(df.columns)
#     assert result.data is not df.data
#     assert isinstance(result, DataFrame)
#     for key, map_spec in map_specs.items():
#         assert result[key].is_equal(map_spec["expected_result"])


# @product_parametrize(params={"batched": [True, False],
#               "materialize": [True, False]})
# def test_filter(testbed: DataFrameTestBed, batched: bool, materialize: bool):
#     df = testbed.df
#     name = list(testbed.column_testbeds.keys())[0]
#     filter_spec = testbed.column_testbeds[name].get_filter_spec(
#         batched=batched, materialize=materialize, salt=1
#     )

#     def func(x):
#         out = filter_spec["fn"](x[name])
#         return out

#     result = df.filter(
#         func,
#         batch_size=4,
#         is_batched_fn=batched,
#         materialize=materialize,
#     )
#     assert isinstance(result, DataFrame)
#     result[name].is_equal(filter_spec["expected_result"])


def test_remove_column():
    a = np.arange(16)
    b = np.arange(16) * 2
    df = DataFrame.from_batch({"a": a, "b": b})
    assert "a" in df
    df.remove_column("a")
    assert "a" not in df


def test_overwrite_column():
    # make sure we remove the column when overwriting it
    a = NumPyTensorColumn(np.arange(16))
    b = NumPyTensorColumn(np.arange(16) * 2)
    df = DataFrame.from_batch({"a": a, "b": b})
    assert "a" in df
    assert df[["a", "b"]]["a"]._data.base is a._data.base
    # testing removal from block manager, so important to use non-blockable type
    df["a"] = ObjectColumn(range(16))
    assert df[["a", "b"]]["a"]._data is not a
    # check that there are no duplicate columns
    assert set(df.columns) == set(["a", "b"])


def test_rename():
    a = NumPyTensorColumn(np.arange(16))
    b = NumPyTensorColumn(np.arange(16) * 2)

    df = DataFrame.from_batch({"a": a, "b": b})
    assert "a" in df

    new_df = df.rename({"a": "A"})

    # make sure "a" was renamed to "A"
    assert np.equal(new_df["A"], a)
    assert np.equal(new_df["b"], b)

    # check that there are no duplicate columns
    assert set(new_df.columns) == set(["A", "b"])

    # make sure rename happened out of place
    assert df["a"]._data is a._data
    assert df["b"]._data is b._data

    new_df = df.rename(str.upper)

    # make sure "a" was renamed to "A" and "b" was renamed to "B"
    assert np.equal(new_df["A"], a)
    assert np.equal(new_df["B"], b)

    # check that there are no duplicate columns
    assert set(new_df.columns) == set(["A", "B"])

    # make sure rename happened out of place
    assert df["a"]._data is a._data
    assert df["b"]._data is b._data


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

        if isinstance(new_df[name], DeferredColumn):
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
        "b": ObjectColumn(np.arange(length)),
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
                col.is_equal(df[batch["idx"]][name])
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
        self.name = "subclass"
        super().__init__(*args, **kwargs)

    def _state_keys(cls) -> Set[str]:
        return super()._state_keys().union({"name"})


def test_subclass():
    df1 = DataFrameSubclass({"a": np.arange(3), "b": ["may", "jun", "jul"]})
    df2 = DataFrameSubclass(
        {"c": np.arange(3), "d": ["2021", "2022", "2023"]}
    )

    assert isinstance(df1[np.asarray([0, 1])], DataFrameSubclass)
    assert isinstance(df1[:2], DataFrameSubclass)
    assert isinstance(df1[:2], DataFrameSubclass)

    assert isinstance(df1.merge(df2, left_on="a", right_on="c"), DataFrameSubclass)
    assert isinstance(df1.append(df1), DataFrameSubclass)

    assert df1._state_keys() == set(["name", "_primary_key"])
    assert df1._get_state() == {"name": "subclass", "_primary_key": "a"}


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
        if isinstance(df_new[k], ScalarColumn):
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
    assert len(df.columns) == 3

    # Returns a dataset
    df = DataFrame.from_huggingface(
        "hf-internal-testing/fixtures_ade20k",
        cache_dir=tmpdir,
        split="test",
    )
    assert len(df) == 4
    assert len(df.columns) == 3


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

    df_new = DataFrame.from_json(temp_f.name, lines=True)
    assert df_new.columns == ["a", "b", "c"]
    # Skip index column
    for k in data:
        if isinstance(df_new[k], TorchTensorColumn):
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
        assert isinstance(df[col], ArrowScalarColumn)
        assert pa.compute.equal(df[col].data, table[col])


def test_to_pandas_allow_objects():
    import pandas as pd

    length = 16
    batch = {
        "a": np.arange(length),
        "b": ObjectColumn(np.arange(length)),
        "c": [{"a": 2}] * length,
        "d": torch.arange(length),
        # offset the index to test robustness to nonstandard indices
        "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
        # test multidimensional
        "f": np.ones((length, 5)).astype(int),
        "g": torch.ones(length, 5).to(int),
    }
    df = DataFrame(batch)

    df_pd = df.to_pandas(allow_objects=True)
    assert isinstance(df_pd, pd.DataFrame)
    assert list(df.columns) == list(df_pd.columns)
    assert len(df) == len(df_pd)

    assert (df_pd["a"].values == df["a"].data).all()
    assert list(df["b"]) == list(df["b"].data)

    assert isinstance(df_pd["c"][0], dict)

    assert (df_pd["d"].values == df["d"].numpy()).all()
    assert (df_pd["e"].values == df["e"].values).all()


def test_to_pandas_disallow_objects(testbed):
    df = testbed.df

    pdf = df.to_pandas(allow_objects=False)

    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in pdf
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in pdf
        else:
            assert name in pdf


def test_to_arrow(testbed):
    df = testbed.df

    adf = df.to_arrow()
    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in adf.column_names
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in adf.column_names
        else:
            assert name in adf.column_names
            assert (adf[name].to_numpy() == col.to_numpy()).all()


@product_parametrize(params={"engine": ["arrow", "pandas"]})
def test_csv_io(testbed, tmpdir, engine):
    df = testbed.df
    filepath = os.path.join(tmpdir, "test.csv")

    with pytest.warns():
        df.to_csv(filepath, engine=engine)

    df2 = DataFrame.from_csv(filepath)

    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in df2
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in df2
        else:
            # note we do not check equality because writing to CSV can lead to
            # casting issues
            assert name in df2


@product_parametrize(params={"engine": ["arrow", "pandas"]})
def test_feather_io(testbed, tmpdir, engine):
    df = testbed.df
    filepath = os.path.join(tmpdir, "test.feather")

    with pytest.warns():
        df.to_feather(filepath, engine=engine)

    df2 = DataFrame.from_feather(filepath)

    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in df2
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in df2
        else:
            assert name in df2
            assert (df2[name].to_numpy() == col.to_numpy()).all()


@product_parametrize(params={"engine": ["arrow", "pandas"]})
def test_parquet_io(testbed, tmpdir, engine):
    df = testbed.df
    filepath = os.path.join(tmpdir, "test.parquet")

    with pytest.warns():
        df.to_parquet(filepath, engine=engine)

    df2 = DataFrame.from_parquet(filepath)

    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in df2
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in df2
        else:
            assert name in df2
            assert (df2[name].to_numpy() == col.to_numpy()).all()


def test_json_io(testbed, tmpdir):
    df = testbed.df
    filepath = os.path.join(tmpdir, "test.json")

    with pytest.warns():
        df.to_json(filepath)

    df2 = DataFrame.from_json(filepath, dtype=False)

    for name, col in df.items():
        if isinstance(col, ObjectColumn) or isinstance(col, DeferredColumn):
            assert name not in df2
        elif isinstance(col, TensorColumn) and len(col.shape) > 1:
            assert name not in df2
        else:
            assert name in df2
            if col.to_numpy().dtype == np.object:
                assert np.all(df2[name].to_numpy() == col.to_numpy())
            else:
                assert np.allclose(df2[name].to_numpy(), col.to_numpy())



def test_constructor():
    length = 16

    # from dictionary
    data = {
        "a": np.arange(length),
        "b": ObjectColumn(np.arange(length)),
    }
    df = DataFrame(data=data)
    assert len(df) == length
    assert df["a"].is_equal(ScalarColumn(np.arange(length)))

    # from BlockManager
    mgr = BlockManager.from_dict(data)
    df = DataFrame(data=mgr)
    assert len(df) == length
    assert df["a"].is_equal(ScalarColumn(np.arange(length)))
    assert df.columns == ["a", "b"]

    # from list of dictionaries
    data = [{"a": idx, "b": str(idx), "c": {"test": idx}} for idx in range(length)]
    df = DataFrame(data=data)
    assert len(df) == length
    assert df["a"].is_equal(ScalarColumn(np.arange(length)))
    assert isinstance(df["c"], ObjectColumn)
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
    assert df["a"].is_equal(ScalarColumn(np.arange(length)))
    # need to fillna because nan comparisons return false in pandas
    assert (
        df["c"]
        .fillna(0)
        .is_equal(ScalarColumn([0 if idx % 2 == 0 else idx for idx in range(length)]))
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
        "b": ObjectColumn(np.arange(length - 1)),
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
        "b": ObjectColumn(np.arange(length)),
    }
    df = DataFrame(data)
    assert df.shape == (16, 2)


def test_str(testbed):
    result = str(testbed.df)
    assert isinstance(result, str)


def test_repr(testbed):
    result = repr(testbed.df)
    assert isinstance(result, str)


@product_parametrize(params={"max_rows": [6, 16, 20]})
def test_repr_pandas(testbed, max_rows: int):
    mk.config.display.max_rows = max_rows
    df, _ = testbed.df._repr_pandas_()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == min(len(df), max_rows + 1)


@product_parametrize(params={"column_type": [ScalarColumn, NumPyTensorColumn]})
def test_loc_single(testbed, column_type: type):
    df = testbed.df
    # int index => single row (dict)
    index = 2
    df["pk"] = column_type(np.arange(len(df)) + 10).astype(str)
    df = df.set_primary_key("pk")

    row = df.loc[str(index + 10)]
    assert isinstance(row, dict)

    for key, value in row.items():
        if key == "pk":
            continue
        col_testbed = testbed.column_testbeds[key]
        col_testbed.assert_data_equal(
            value, col_testbed.get_data(index, materialize=False)
        )


@product_parametrize(params={"column_type": [ScalarColumn, NumPyTensorColumn]})
def test_loc_multiple(testbed, column_type):
    df = testbed.df
    # int index => single row (dict)
    indices = np.array([2, 3])
    df["pk"] = column_type(np.arange(len(df)) + 10).astype(str)
    df = df.set_primary_key("pk")

    loc_index = (indices + 10).astype(str)
    new_df = df.loc[loc_index]
    assert isinstance(new_df, DataFrame)

    for key, value in new_df.items():
        if key == "pk":
            continue
        col_testbed = testbed.column_testbeds[key]
        data = col_testbed.get_data(indices, materialize=False)
        col_testbed.assert_data_equal(value.data, data)


def test_loc_missing():
    df = DataFrame({"x": TorchTensorColumn([1, 2, 3]), "y": ScalarColumn([4, 5, 6])})
    df = df.set_primary_key("y")

    with pytest.raises(KeyError):
        df.loc[1, 2, 4]


def test_primary_key_persistence():
    df = DataFrame({"a": ScalarColumn(np.arange(16)), "b": ScalarColumn(np.arange(16))})
    df = df.set_primary_key("a")

    df = df[:4]
    df._primary_key == "a"
    assert (df.primary_key == ScalarColumn(np.arange(4))).all()


def test_invalid_primary_key():
    # multidimenmsional
    df = DataFrame({"a": TorchTensorColumn([[1, 2, 3]])})

    with pytest.raises(ValueError):
        df.set_primary_key("a")


def test_primary_key_reset():
    df = DataFrame({"a": ScalarColumn(np.arange(16)), "b": ScalarColumn(np.arange(16))})
    df = df.set_primary_key("a")

    df["a"] = ScalarColumn(np.arange(16))
    assert df._primary_key is None


def test_check_primary_key_reset():
    df = DataFrame({"a": ScalarColumn(np.arange(16)), "b": ScalarColumn(np.arange(16))})
    df = df.set_primary_key("a")

    assert df.append(df).primary_key is None


def test_check_primary_key_no_reset():
    df = DataFrame({"a": ScalarColumn(np.arange(16)), "b": ScalarColumn(np.arange(16))})
    df = df.set_primary_key("a")

    df2 = DataFrame(
        {"a": ScalarColumn(np.arange(16, 32)), "b": ScalarColumn(np.arange(16))}
    )

    assert df.append(df2).primary_key is not None


@pytest.mark.parametrize("x", [0, 0.0, "hello world", np.nan, np.inf])
def test_scalar_setitem(x):
    df = DataFrame({"a": ScalarColumn(np.arange(16))})
    df["extra_column"] = x

    assert len(df["extra_column"]) == len(df)
    if isinstance(x, str):
        assert isinstance(df["extra_column"], ScalarColumn)
    else:
        assert isinstance(df["extra_column"], ScalarColumn)
    if not isinstance(x, str) and (np.isnan(x) or np.isinf(x)):
        if np.isnan(x):
            assert np.all(np.isnan(df["extra_column"]))
        elif np.isinf(x):
            assert np.all(np.isinf(df["extra_column"]))
    else:
        assert all(df["extra_column"] == x)


@mk.gui.endpoint
def _set_store_or_df(store, value):
    store.set(value)


@pytest.mark.parametrize(
    "name",
    [
        # Instance variables.
        "_data",
        "_primary_key",
        # Properties.
        "gui",
        "data",
        "columns",
        "primary_key",
        "primary_key_name",
        "nrows",
        "ncols",
        "shape",
    ],
)
def test_reactivity_attributes_and_properties(name):
    """Test that attributes and properties of the dataframe are reactive."""

    class Foo:
        pass

    df = DataFrame({"a": np.arange(10), "b": torch.arange(10), "c": [Foo()] * 10})

    # These should return an object that can be attached to a node.
    # i.e. we should be able to put the output on the graph.
    with mk.gui._react():
        out = getattr(df, name)
    assert isinstance(out, NodeMixin)
    assert df.inode.has_trigger_children()
    assert len(df.inode.trigger_children) == 1
    # Check the operation name
    op = df.inode.trigger_children[0].obj
    assert isinstance(op, Operation)
    assert op.fn.__name__ == name

    # No reactivity.
    # We want to check that the output is not a Store.
    # it can be other NodeMixIns, because these are classes built into meerkat.
    out = getattr(df, name)
    assert not isinstance(out, mk.gui.Store)


def test_reactivity_len():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    with mk.gui._react():
        with pytest.warns(UserWarning):
            length = len(df)
    assert length == 10

    # Warnings should not be raised if we are not in a reactive context.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        length = len(df)
    assert length == 10


def test_reactivity_contains():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    store = mk.gui.Store("a")
    with mk.gui._react():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a_contains = df.contains(store)
    inode = a_contains.inode

    assert a_contains
    assert isinstance(a_contains, mk.gui.Store)
    assert isinstance(a_contains, bool)
    assert df.inode.has_trigger_children()
    assert len(df.inode.trigger_children) == 1
    op = df.inode.trigger_children[0].obj
    assert isinstance(op, Operation)
    assert op.fn.__name__ == "contains"
    assert len(op.inode.trigger_children) == 1
    assert id(op.inode.trigger_children[0]) == id(a_contains.inode)

    # Change the store
    _set_store_or_df(store, "c")
    assert not inode.obj

    # The `in` operator coerces the output of __contains__ to a bool.
    # Store.__bool__ cannot return a bool due to cpython limitations.
    df = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    with mk.gui._react():
        with pytest.warns(UserWarning):
            a_in = "a" in df
    assert not isinstance(a_in, mk.gui.Store)


def test_reactivity_size():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    with mk.gui._react():
        shape = df.size()
    inode = shape.inode

    assert shape == (10, 2)
    assert isinstance(shape, mk.gui.Store)
    assert isinstance(shape, tuple)
    assert df.inode.has_trigger_children()
    assert len(df.inode.trigger_children) == 1
    op = df.inode.trigger_children[0].obj
    assert isinstance(op, Operation)
    assert op.fn.__name__ == "size"
    assert len(op.inode.trigger_children) == 1
    assert id(op.inode.trigger_children[0]) == id(shape.inode)

    # Change the dataframe
    _set_store_or_df(df, DataFrame({"a": np.arange(5)}))
    assert inode.obj == (5, 1)


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_reactivity_append(axis: str):
    df = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    if axis == "rows":
        df2 = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    else:
        df2 = DataFrame({"c": np.arange(10), "d": torch.arange(10)})

    with mk.gui._react():
        df_append = df.append(df2, axis=axis)
    inode = df_append.inode

    assert df.inode.has_trigger_children()
    assert len(df.inode.trigger_children) == 1
    op = df.inode.trigger_children[0].obj
    assert isinstance(op, Operation)
    assert op.fn.__name__ == "append"
    assert len(op.inode.trigger_children) == 1
    assert id(op.inode.trigger_children[0]) == id(df_append.inode)

    # Change the input dataframe
    if axis == "rows":
        _set_store_or_df(df, DataFrame({"a": np.arange(5), "b": torch.arange(5)}))
        assert inode.obj.shape == (15, 2)
    else:
        _set_store_or_df(
            df, DataFrame({"alpha": np.arange(10), "beta": torch.arange(10)})
        )
        assert inode.obj.columns == ["alpha", "beta", "c", "d"]


@pytest.mark.parametrize("op_name", ["head", "tail"])
def test_reactivity_head_tail(op_name: str):
    df = DataFrame({"a": np.arange(10), "b": torch.arange(10)})
    with mk.gui._react():
        df_slice = getattr(df, op_name)()
    assert df.inode.has_trigger_children()
    assert len(df.inode.trigger_children) == 1
    op = df.inode.trigger_children[0].obj
    assert isinstance(op, Operation)
    assert op.fn.__name__ == op_name
    assert len(op.inode.trigger_children) == 1
    assert id(op.inode.trigger_children[0]) == id(df_slice.inode)


def test_reactivity_getitem_multiple_columns():
    df = DataFrame(
        {"a": np.arange(10), "b": torch.arange(20, 30), "c": torch.arange(40, 50)}
    )
    store = mk.gui.Store(["a", "b"])
    with mk.gui._react():
        df_col = df[store]
    inode = df_col.inode

    assert isinstance(df_col, DataFrame)
    assert df.inode.has_trigger_children()
    assert len(df.inode.trigger_children) == 1
    op = df.inode.trigger_children[0].obj
    assert isinstance(op, Operation)
    assert op.fn.__name__ == "__getitem__"
    assert len(op.inode.trigger_children) == 1
    assert id(op.inode.trigger_children[0]) == id(df_col.inode)
    assert len(store.inode.trigger_children) == 1
    assert id(store.inode.trigger_children[0]) == id(op.inode)

    # Change the store
    _set_store_or_df(store, ["c"])
    assert np.all(inode.obj["c"].to_numpy() == df["c"].to_numpy())

    # Change the dataframe
    _set_store_or_df(df, DataFrame({"c": np.arange(5)}))
    assert np.all(inode.obj["c"].to_numpy() == np.arange(5))

# TODO: Add these tests back in 
# def test_reactivity_getitem_single_column():
#     # TODO: We need to add support for column modifications in _update_result
#     # in operation.
#     df = DataFrame(
#         {"a": np.arange(10), "b": torch.arange(20, 30), "c": torch.arange(40, 50)}
#     )
#     store = mk.gui.Store("b")
#     with mk.gui.react():
#         df_col = df[store]
#     inode = df_col.inode

#     _set_store_or_df(df, DataFrame({"c": np.arange(5)}))
#     store.set("c")
#     assert np.all(inode.obj["a"] == np.arange(5))


# def test_reactivity_getitem_slicing():
#     df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
#     store = mk.gui.Store(slice(0, 5))
#     with mk.gui.react():
#         df_col = df[store]
#     inode = df_col.inode
def test_reactivity_getitem_single_column():
    # TODO: We need to add support for column modifications in _update_result
    # in operation.
    df = DataFrame(
        {"a": np.arange(10), "b": torch.arange(20, 30), "c": torch.arange(40, 50)}
    )
    store = mk.gui.Store("b")
    with mk.gui._react():
        df_col = df[store]
    inode = df_col.inode

#     assert isinstance(df_col, DataFrame)
#     assert df.inode.has_trigger_children()
#     assert len(df.inode.trigger_children) == 1
#     op = df.inode.trigger_children[0].obj
#     assert isinstance(op, Operation)
#     assert op.fn.__name__ == "__getitem__"
#     assert len(op.inode.trigger_children) == 1
#     assert id(op.inode.trigger_children[0]) == id(df_col.inode)
#     assert len(store.inode.trigger_children) == 1
#     assert id(store.inode.trigger_children[0]) == id(op.inode)

#     # Change the store
#     _set_store_or_df(store, slice(5, 10))
#     assert np.all(inode.obj["a"] == np.arange(5, 10))
#     assert np.all(inode.obj["b"] == np.arange(25, 30))

#     # Change the dataframe
#     _set_store_or_df(df, DataFrame({"a": np.arange(5)}))
#     assert len(inode.obj) == 0
#     _set_store_or_df(store, slice(0, 5))
#     assert np.all(inode.obj["a"] == np.arange(5))


# def test_reactivity_merge():
#     df1 = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
#     df2 = DataFrame({"a": np.arange(10), "d": torch.arange(20, 30)})
#     on = mk.gui.Store("a")
#     with mk.gui.react():
#         df_merge = df1.merge(df2, on=on)
#     inode = df_merge.inode
def test_reactivity_getitem_slicing():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
    store = mk.gui.Store(slice(0, 5))
    with mk.gui._react():
        df_col = df[store]
    inode = df_col.inode

#     assert np.all(df_merge.to_pandas() == df1.merge(df2, on="a").to_pandas())
#     assert len(df1.inode.trigger_children) == 1
#     assert len(df2.inode.trigger_children) == 1
#     assert df1.inode.trigger_children[0].obj.fn.__name__ == "merge"
#     assert df2.inode.trigger_children[0].obj.fn.__name__ == "merge"

#     new_df = df1.copy()
#     new_df["a"][-1] = 20
#     _set_store_or_df(df1, new_df)
#     assert len(inode.obj) == 9


# def test_reactivity_sort():
#     a, b = np.arange(10), np.arange(20, 30)
#     np.random.shuffle(a)
#     np.random.shuffle(b)
def test_reactivity_merge():
    df1 = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
    df2 = DataFrame({"a": np.arange(10), "d": torch.arange(20, 30)})
    on = mk.gui.Store("a")
    with mk.gui._react():
        df_merge = df1.merge(df2, on=on)
    inode = df_merge.inode

#     df = DataFrame({"a": a, "b": b})
#     store = mk.gui.Store("a")
#     with mk.gui.react():
#         df_sort = df.sort(by=store)
#     inode = df_sort.inode

#     assert np.all(inode.obj["a"] == np.arange(10))

#     _set_store_or_df(store, "b")
#     assert np.all(inode.obj["b"] == np.arange(20, 30))


# def test_reactivity_sample():
#     df = DataFrame({"a": np.arange(100)})
#     frac = mk.gui.Store(0.1)
#     with mk.gui.react():
#         df_sample = df.sample(frac=frac)
#     inode = df_sample.inode

#     assert len(inode.obj) == 10
    # df = DataFrame({"a": a, "b": b})
    # store = mk.gui.Store("a")
    # with mk.gui._react():
    #     df_sort = df.sort(by=store)
    # inode = df_sort.inode

#     _set_store_or_df(frac, 0.2)
#     assert len(inode.obj) == 20


# def test_reactivity_rename():
#     df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
#     store = mk.gui.Store({"a": "c"})
#     with mk.gui.react():
#         df_rename = df.rename(mapper=store)
#     inode = df_rename.inode
def test_reactivity_sample():
    df = DataFrame({"a": np.arange(100)})
    frac = mk.gui.Store(0.1)
    with mk.gui._react():
        df_sample = df.sample(frac=frac)
    inode = df_sample.inode

#     assert list(inode.obj.keys()) == ["c", "b"]

#     # rename is an out-of-place method.
#     # renaming occurs on the source dataframe, which has columns "a" and "b".
#     # Calling `rename` with "b" -> "d" will operate on the source dataframe.
#     # Thus column "a" should still exist.
#     _set_store_or_df(store, {"b": "d"})
#     assert list(inode.obj.keys()) == ["a", "d"]


# def test_reactivity_drop():
#     df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
#     store = mk.gui.Store(["a"])
#     with mk.gui.react():
#         df_drop = df.drop(columns=store)
#     inode = df_drop.inode
def test_reactivity_rename():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
    store = mk.gui.Store({"a": "c"})
    with mk.gui._react():
        df_rename = df.rename(mapper=store)
    inode = df_rename.inode

#     assert list(inode.obj.keys()) == ["b"]

#     # drop is an out-of-place method.
#     # Thus, column "a" will still exist when `drop` is rerun with argument "b".
#     _set_store_or_df(store, ["b"])
#     assert list(inode.obj.keys()) == ["a"]


# def test_reactivity_keys():
#     df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
#     with mk.gui.react():
#         keys = df.keys()
#     inode = keys.inode
def test_reactivity_drop():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
    store = mk.gui.Store(["a"])
    with mk.gui._react():
        df_drop = df.drop(columns=store)
    inode = df_drop.inode

#     assert list(keys) == ["a", "b"]

#     _set_store_or_df(df, DataFrame({"c": np.arange(10)}))
#     assert list(inode.obj) == ["c"]
    # drop is an out-of-place method.
    # Thus, column "a" will still exist when `drop` is rerun with argument "b".
    _set_store_or_df(store, ["b"])
    assert list(inode.obj.keys()) == ["a"]


def test_reactivity_keys():
    df = DataFrame({"a": np.arange(10), "b": torch.arange(20, 30)})
    with mk.gui._react():
        keys = df.keys()
    inode = keys.inode

    assert list(keys) == ["a", "b"]

    _set_store_or_df(df, DataFrame({"c": np.arange(10)}))
    assert list(inode.obj) == ["c"]
