import pytest

from meerkat import DeferredColumn
from meerkat.dataframe import DataFrame

from ...utils import product_parametrize
from ..columns.abstract import AbstractColumnTestBed, column_parametrize

# from ..columns.deferred.test_deferred import DeferredColumnTestBed
# from ..columns.deferred.test_image import ImageColumnTestBed
# from ..columns.scalar.test_arrow import ArrowScalarColumnTestBed
# from ..columns.scalar.test_pandas import PandasScalarColumnTestBed
from ..columns.tensor.test_numpy import NumPyTensorColumnTestBed

# from ..columns.tensor.test_torch import TorchTensorColumnTestBed


@pytest.fixture(
    **column_parametrize(
        [
            NumPyTensorColumnTestBed,
            # PandasScalarColumnTestBed,
            # TorchTensorColumnTestBed,
            # DeferredColumnTestBed,
            # ArrowScalarColumnTestBed,
            # ImageColumnTestBed,
        ]
    )
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@product_parametrize(
    params={
        "batched": [True, False],
        "materialize": [True, False],
        "use_ray": [False], # TODO: Put the tests with ray back
    }
)
def test_map_return_single(
    column_testbed: AbstractColumnTestBed,
    batched: bool,
    materialize: bool,
    use_ray: bool,
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, DeferredColumn) or materialize):
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
        use_ray=use_ray,
    )
    assert result.is_equal(map_spec["expected_result"])


# @product_parametrize(params={"batched": [True, False], "materialize": [True, False]})
# def test_map_return_single_w_kwarg(
#     column_testbed: AbstractColumnTestBed, batched: bool, materialize: bool
# ):
#     """`map`, single return,"""
#     if not (isinstance(column_testbed.col, DeferredColumn) or materialize):
#         # skip columns for which materialize has no effect
#         return

#     col = column_testbed.col
#     kwarg = 2
#     map_spec = column_testbed.get_map_spec(
#         batched=batched, materialize=materialize, kwarg=kwarg
#     )

#     def func(x, k=0):
#         out = map_spec["fn"](x, k=k)
#         return out

#     result = col.map(
#         func,
#         batch_size=4,
#         is_batched_fn=batched,
#         materialize=materialize,
#         output_type=map_spec.get("output_type", None),
#         k=kwarg,
#     )
#     assert result.is_equal(map_spec["expected_result"])


@product_parametrize(
    params={
        "batched": [True, False],
        "materialize": [True, False],
        "use_ray": [False],  # TODO (dean): Multiple outputs not supported.
    }
)
def test_map_return_multiple(
    column_testbed: AbstractColumnTestBed,
    batched: bool,
    materialize: bool,
    use_ray: bool,
):
    """`map`, single return,"""
    if not (isinstance(column_testbed.col, DeferredColumn) or materialize):
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
        output_type={k: v.get("output_type", None) for k, v in map_specs.items()},
        use_ray=use_ray,
    )
    assert isinstance(result, DataFrame)
    for key, map_spec in map_specs.items():
        assert result[key].is_equal(map_spec["expected_result"])
