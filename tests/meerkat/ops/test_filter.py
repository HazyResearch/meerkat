import pytest

from ...utils import product_parametrize
from ..columns.abstract import AbstractColumnTestBed, column_parametrize

# from ..columns.scalar.test_arrow import ArrowScalarColumnTestBed
from ..columns.scalar.test_pandas import PandasScalarColumnTestBed

# from ..columns.tensor.test_numpy import NumPyTensorColumnTestBed
# from ..columns.tensor.test_torch import TorchTensorColumnTestBed


@pytest.fixture(
    **column_parametrize(
        [
            # NumPyTensorColumnTestBed.get_params(
            #     config={"num_dims": [1], "dim_length": [1]}
            # ),
            PandasScalarColumnTestBed,
            # TorchTensorColumnTestBed.get_params(
            #     config={"num_dims": [1], "dim_length": [1]}
            # ),
            # ArrowScalarColumnTestBed,
        ]
    )
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@product_parametrize(params={"batched": [True, False]})
def test_filter(column_testbed: AbstractColumnTestBed, batched: bool):
    """multiple_dim=False."""

    col = column_testbed.col
    filter_spec = column_testbed.get_filter_spec(
        batched=batched,
    )

    def func(x):
        out = filter_spec["fn"](x)
        return out

    result = col.filter(
        func,
        batch_size=4,
        is_batched_fn=batched,
    )

    assert result.is_equal(filter_spec["expected_result"])
