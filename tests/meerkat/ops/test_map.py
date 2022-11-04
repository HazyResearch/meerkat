import pytest

from meerkat import LambdaColumn
from meerkat.dataframe import DataFrame

from ...utils import product_parametrize
from ..columns.abstract import AbstractColumnTestBed, column_parametrize
from ..columns.test_arrow_column import ArrowArrayColumnTestBed
from ..columns.test_cell_column import CellColumnTestBed
from ..columns.test_image_column import ImageColumnTestBed
from ..columns.test_lambda_column import LambdaColumnTestBed
from ..columns.test_numpy_column import NumpyArrayColumnTestBed
from ..columns.test_pandas_column import PandasSeriesColumnTestBed
from ..columns.test_tensor_column import TensorColumnTestBed


@pytest.fixture(
    **column_parametrize(
        [
            NumpyArrayColumnTestBed,
            PandasSeriesColumnTestBed,
            TensorColumnTestBed,
            LambdaColumnTestBed,
            ArrowArrayColumnTestBed,
            CellColumnTestBed,
            ImageColumnTestBed,
        ]
    )
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_single(
    column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, LambdaColumn) or materialize):
        # skip columns for which materialize has no effect
        return

    col = column_testbed.col

    map_spec = column_testbed.get_map_spec(batched=batched, materialize=materialize)

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


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_single_w_kwarg(
    column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, LambdaColumn) or materialize):
        # skip columns for which materialize has no effect
        return

    col = column_testbed.col
    kwarg = 2
    map_spec = column_testbed.get_map_spec(
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


@product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
def test_map_return_multiple(
    column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, LambdaColumn) or materialize):
        # skip columns for which materialize has no effect
        return

    col = column_testbed.col
    map_specs = {
        "map1": column_testbed.get_map_spec(
            batched=batched, materialize=materialize, salt=1
        ),
        "map2": column_testbed.get_map_spec(
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
    assert isinstance(result, DataFrame)
    for key, map_spec in map_specs.items():
        assert result[key].is_equal(map_spec["expected_result"])
