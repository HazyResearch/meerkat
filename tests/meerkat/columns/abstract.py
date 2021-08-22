import os
import pickle
from functools import wraps
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meerkat.datapanel import DataPanel
from meerkat.ops.concat import concat


@pytest.fixture
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


class AbstractColumnTestBed:

    DEFAULT_CONFIG = {}

    @classmethod
    def get_params(cls, config: dict = None, params: dict = None, single: bool = False):
        updated_config = cls.DEFAULT_CONFIG.copy()
        if config is not None:
            updated_config.update(config)
        configs = [
            (cls, config)
            for config in map(
                dict,
                product(*[[(k, v) for v in vs] for k, vs in updated_config.items()]),
            )
        ]
        if single:
            configs = configs[:1]
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
        cls, config: dict = None, params: dict = None, single: bool = False
    ):
        return pytest.mark.parametrize(
            **cls.get_params(config=config, params=params, single=single),
            indirect=["testbed"]
        )

    @classmethod
    def single(cls, tmpdir):
        return cls(**cls.get_params(single=True)["argvalues"][0][1], tmpdir=tmpdir)

    def get_map_spec(self, key: str = "default"):
        raise NotImplementedError()

    def get_data(self, index):
        raise NotImplementedError()

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        raise NotImplementedError()


class TestAbstractColumn:
    __test__ = False
    testbed_class: type = None
    column_class: type = None

    def test_getitem(self, testbed, index_type: type = np.array):
        col = testbed.col

        testbed.assert_data_equal(testbed.get_data(1), col[1])

        for index in [
            slice(2, 4, 1),
            (np.arange(len(col)) % 2).astype(bool),
            np.array([0, 3, 5, 6]),
        ]:
            col_index = index_type(index) if not isinstance(index, slice) else index
            data = testbed.get_data(index)
            result = col[col_index]
            testbed.assert_data_equal(data, result.data)

            if type(result) == type(col):
                # if the getitem returns a column of the same type, enforce that all the
                # attributes were cloned over appropriately. We don't want to check this
                # for columns that return columns of different type from getitem
                # (e.g. LambdaColumn)
                assert col._clone(data=data).is_equal(result)

    def _get_data_to_set(self, testbed, data_index):
        raise NotImplementedError

    def test_set_item(self, testbed, index_type: type = np.array):
        col = testbed.col

        for index in [
            1,
            slice(2, 4, 1),
            (np.arange(len(col)) % 2).astype(bool),
            np.array([0, 3, 5, 6]),
        ]:
            col_index = index_type(index) if isinstance(index, np.ndarray) else index
            data_to_set = self._get_data_to_set(testbed, index)
            col[col_index] = data_to_set
            if isinstance(index, int):
                testbed.assert_data_equal(data_to_set, col.lz[col_index])
            else:
                testbed.assert_data_equal(data_to_set, col.lz[col_index].data)

    def test_map_return_single(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        """`map`, single return,"""
        col = testbed.col
        map_spec = testbed.get_map_spec(batched=batched, materialize=materialize)

        def func(x):
            out = map_spec["fn"](x)
            return out

        result = col.map(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            output_type=map_spec.get("output_type", None),
        )
        assert result.is_equal(map_spec["expected_result"])

    def test_map_return_single_w_kwarg(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        """`map`, single return,"""
        col = testbed.col
        kwarg = 2
        map_spec = testbed.get_map_spec(
            batched=batched, materialize=materialize, kwarg=kwarg
        )

        def func(x, k=0):
            out = map_spec["fn"](x, k=k)
            return out

        result = col.map(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            output_type=map_spec.get("output_type", None),
            k=kwarg,
        )
        assert result.is_equal(map_spec["expected_result"])

    def test_map_return_multiple(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool = True
    ):
        """`map`, single return,"""
        col = testbed.col
        map_specs = {
            "map1": testbed.get_map_spec(
                batched=batched, materialize=materialize, salt=1
            ),
            "map2": testbed.get_map_spec(
                batched=batched, materialize=materialize, salt=2
            ),
        }

        def func(x):
            out = {key: map_spec["fn"](x) for key, map_spec in map_specs.items()}
            return out

        result = col.map(
            func,
            batch_size=4,
            is_batched_fn=batched,
            materialize=materialize,
            output_type=list(map_specs.values())[0].get("output_type", None),
        )
        assert isinstance(result, DataPanel)
        for key, map_spec in map_specs.items():
            assert result[key].is_equal(map_spec["expected_result"])

    def test_filter_1(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool = True
    ):
        """multiple_dim=False."""
        col = testbed.col
        filter_spec = testbed.get_filter_spec(batched=batched, materialize=materialize)

        def func(x):
            out = filter_spec["fn"](x)
            return out

        result = col.filter(
            func, batch_size=4, is_batched_fn=batched, materialize=materialize
        )

        assert result.is_equal(filter_spec["expected_result"])

    def test_concat(self, testbed: AbstractColumnTestBed, n: int = 2):
        col = testbed.col
        out = concat([col] * n)

        assert len(out) == len(col) * n
        assert isinstance(out, type(col))
        for i in range(n):
            assert out.lz[i * len(col) : (i + 1) * len(col)].is_equal(col)

    def test_copy(self, testbed: AbstractColumnTestBed):
        col, _ = testbed.col, testbed.data
        col_copy = col.copy()

        assert isinstance(col_copy, self.column_class)
        assert col.is_equal(col_copy)

    def test_pickle(self, testbed):
        # important for dataloader
        col = testbed.col
        buf = pickle.dumps(col)
        new_col = pickle.loads(buf)

        assert isinstance(new_col, self.column_class)
        assert col.is_equal(new_col)

    def test_io(self, tmp_path, testbed: AbstractColumnTestBed):
        # uses the tmp_path fixture which will provide a
        # temporary directory unique to the test invocation,
        # important for dataloader
        col, _ = testbed.col, testbed.data

        path = os.path.join(tmp_path, "test")
        col.write(path)

        new_col = self.column_class.read(path)

        assert isinstance(new_col, self.column_class)
        assert col.is_equal(new_col)

    def test_head(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        length = 10
        result = testbed.col.head(length)
        assert len(result) == length
        assert result.is_equal(testbed.col.lz[:length])

    def test_tail(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        length = 10
        result = testbed.col.tail(length)
        assert len(result) == length
        assert result.is_equal(testbed.col.lz[-length:])

    def test_repr_html(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        testbed.col._repr_html_()

    def test_str(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        result = str(testbed.col)
        assert isinstance(result, str)

    def test_repr(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        result = repr(testbed.col)
        assert isinstance(result, str)

    def test_streamlit(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        testbed.col.streamlit()

    def test_repr_pandas(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        series, _ = testbed.col._repr_pandas_()
        assert isinstance(series, pd.Series)

    def test_to_pandas(self, tmpdir):
        testbed = self.testbed_class.single(tmpdir=tmpdir)
        series = testbed.col.to_pandas()
        assert isinstance(series, pd.Series)
