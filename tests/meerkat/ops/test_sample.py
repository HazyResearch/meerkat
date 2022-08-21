import numpy as np
import pytest

import meerkat as mk
from meerkat.ops.sample import sample


@pytest.fixture
def simple_dp():
    return mk.DataPanel(
        {
            "tensor": mk.TensorColumn([1, 2, 3, 4]),
            "pandas": mk.PandasSeriesColumn([8, 7, 9, 6]),
            "numpy": mk.NumpyArrayColumn([4, 6, 5, 7]),
        }
    )


@pytest.fixture
def simple_column():
    return mk.NumpyArrayColumn([4, 6, 5, 7])


def test_sample_dp_w_n(simple_dp):
    out = simple_dp.sample(
        n=2,
        random_state=42,
    )

    assert (out["tensor"] == mk.TensorColumn([2, 4])).all()
    assert (out["pandas"] == mk.PandasSeriesColumn([7, 6])).all()
    assert (out["numpy"] == mk.NumpyArrayColumn([6, 7])).all()


def test_sample_dp_w_frac(simple_dp):
    out = simple_dp.sample(
        frac=0.5,
        random_state=42,
    )

    assert (out["tensor"] == mk.TensorColumn([2, 4])).all()
    assert (out["pandas"] == mk.PandasSeriesColumn([7, 6])).all()
    assert (out["numpy"] == mk.NumpyArrayColumn([6, 7])).all()


def test_sample_dp_w_weights(simple_dp):
    out = simple_dp.sample(
        n=2,
        weights=np.array([0.5, 0.1, 0.2, 0.2]),
        random_state=42,
    )
    assert (out["tensor"] == mk.TensorColumn([1, 4])).all()
    assert (out["pandas"] == mk.PandasSeriesColumn([8, 6])).all()
    assert (out["numpy"] == mk.NumpyArrayColumn([4, 7])).all()


def test_sample_dp_w_weights_as_str(simple_dp):
    out = simple_dp.sample(
        n=2,
        weights="tensor",
        random_state=42,
    )

    assert (out["tensor"] == mk.TensorColumn([3, 4])).all()
    assert (out["pandas"] == mk.PandasSeriesColumn([9, 6])).all()
    assert (out["numpy"] == mk.NumpyArrayColumn([5, 7])).all()


def test_column(simple_column):
    out = simple_column.sample(
        n=2,
        random_state=42,
    )
    assert (out == mk.NumpyArrayColumn([6, 7])).all()


def test_column_w_weights_as_str(simple_column):
    with pytest.raises(ValueError):
        sample(
            simple_column,
            n=2,
            weights="tensor",
            random_state=42,
        )
