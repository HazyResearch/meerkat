"""Unittests for Datasets."""
import os
import tempfile
from functools import wraps
from itertools import product
from typing import Dict, Sequence

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
from .columns.test_numpy_column import NumpyArrayColumnTestBed
from .columns.test_pandas_column import PandasSeriesColumnTestBed
from .columns.test_image_column import ImageColumnTestBed


class DataPanelTestBed:

    DEFAULT_CONFIG = {
        "consolidated": [True, False],
    }

    DEFAULT_COLUMN_CONFIGS = {
        "np": {"testbed_class": NumpyArrayColumnTestBed, "n": 2},
        "pd": {"testbed_class": PandasSeriesColumnTestBed, "n": 2},
        "img": {"testbed_class": ImageColumnTestBed, "n": 2},
    }

    def __init__(
        self,
        column_configs: Dict[str, AbstractColumn],
        consolidated: bool = True,
        length: int = 16,
        tmpdir: str = None,
    ):
        self.column_testbeds = {}

        def _get_tmpdir(name):
            path = os.path.join(tmpdir, name)
            os.makedirs(path)
            return path

        for name, config in column_configs.items():
            params = config["testbed_class"].get_params(**config.get("kwargs", {}))
            self.column_testbeds.update(
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

        self.columns = {
            name: testbed.col for name, testbed in self.column_testbeds.items()
        }
        self.dp = DataPanel.from_batch(self.columns)

        if consolidated:
            self.dp.consolidate()

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
    def parametrize(cls, config: dict = None, params: dict = None):
        return pytest.mark.parametrize(
            **cls.get_params(config=config, params=params), indirect=["testbed"]
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
                assert new_dp._data[col_name].data is dp._data[col_name].data

    @DataPanelTestBed.parametrize()
    def test_row_index_single(self, testbed):
        dp = testbed.dp

        # int index => single row (dict)
        index = 2
        row = dp[index]
        assert isinstance(row, dict)

        for key, value in row.items():
            if key == "index":
                # TODO(Sabri): remove this when  we change the index functionality
                continue
            col_testbed = testbed.column_testbeds[key]
            col_testbed.assert_data_equal(value, col_testbed.get_data(index))

    @DataPanelTestBed.parametrize(
        params={
            "index_type": [
                np.array,
                # pd.Series,
                # torch.Tensor,
                NumpyArrayColumn,
                PandasSeriesColumn,
                TensorColumn,
            ]
        }
    )
    def test_row_index_multiple(self, testbed, index_type):
        dp = testbed.dp
        rows = np.arange(len(dp))

        # slice index => multiple row selection (DataPanel)
        # tuple or list index => multiple row selection (DataPanel)
        # np.array indeex => multiple row selection (DataPanel)
        for rows, indices in (
            (dp[1:3], rows[1:3]),
            (dp[[0, 2]], rows[[0, 2]]),
            (dp[index_type(np.array((0,)))], rows[np.array((0,))]),
            (dp[index_type(np.array((1, 1)))], rows[np.array((1, 1))]),
            (
                dp[index_type(np.array((True, False) * (len(dp) // 2)))],
                rows[np.array((True, False) * (len(dp) // 2))],
            ),
        ):
            assert isinstance(rows, DataPanel)
            for key, value in rows.items():
                if key == "index":
                    # TODO(Sabri): remove this when  we change the index functionality
                    continue
                col_testbed = testbed.column_testbeds[key]
                data = col_testbed.get_data(indices)
                col_testbed.assert_data_equal(value.data, data)

                if type(value) == type(dp[key]):
                    # if the getitem returns a column of the same type, enforce that all the
                    # attributes were cloned over appropriately. We don't want to check this
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
            if key == "index":
                # TODO(Sabri): remove this when  we change the index functionality
                continue
            col_testbed = testbed.column_testbeds[key]
            col_testbed.assert_data_equal(
                value, col_testbed.get_data(index, materialize=False)
            )

    @DataPanelTestBed.parametrize(
        params={
            "index_type": [
                np.array,
                # pd.Series,
                # torch.Tensor,
                NumpyArrayColumn,
                PandasSeriesColumn,
                TensorColumn,
            ]
        }
    )
    def test_row_index_multiple(self, testbed, index_type):
        dp = testbed.dp
        rows = np.arange(len(dp))

        # slice index => multiple row selection (DataPanel)
        # tuple or list index => multiple row selection (DataPanel)
        # np.array indeex => multiple row selection (DataPanel)
        for rows, indices in (
            (dp.lz[1:3], rows[1:3]),
            (dp.lz[[0, 2]], rows[[0, 2]]),
            (dp.lz[index_type(np.array((0,)))], rows[np.array((0,))]),
            (dp.lz[index_type(np.array((1, 1)))], rows[np.array((1, 1))]),
            (
                dp.lz[index_type(np.array((True, False) * (len(dp) // 2)))],
                rows[np.array((True, False) * (len(dp) // 2))],
            ),
        ):
            assert isinstance(rows, DataPanel)
            for key, value in rows.items():
                if key == "index":
                    # TODO(Sabri): remove this when  we change the index functionality
                    continue
                col_testbed = testbed.column_testbeds[key]
                data = col_testbed.get_data(indices, materialize=False)
                col_testbed.assert_data_equal(value.data, data)

                if type(value) == type(dp[key]):
                    # if the getitem returns a column of the same type, enforce that all the
                    # attributes were cloned over appropriately. We don't want to check this
                    # for columns that return columns of different type from getitem
                    # (e.g. LambdaColumn)
                    assert dp[key]._clone(data=data).is_equal(value)


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
