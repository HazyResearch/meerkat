import numpy as np
import pytest

import meerkat as mk
from meerkat.ops.sample import sample


@pytest.fixture
def simple_df():
    return mk.DataFrame(
        {
            "tensor": mk.TorchTensorColumn([1, 2, 3, 4]),
            "pandas": mk.ScalarColumn([8, 7, 9, 6]),
            "numpy": mk.TorchTensorColumn([4, 6, 5, 7]),
        }
    )


@pytest.fixture
def simple_column():
    return mk.TorchTensorColumn([4, 6, 5, 7])


def test_sample_df_w_n(simple_df):
    out = simple_df.sample(
        n=2,
        random_state=42,
    )

    assert (out["tensor"] == mk.TorchTensorColumn([2, 4])).all()
    assert (out["pandas"] == mk.ScalarColumn([7, 6])).all()
    assert (out["numpy"] == mk.TorchTensorColumn([6, 7])).all()


def test_sample_df_w_frac(simple_df):
    out = simple_df.sample(
        frac=0.5,
        random_state=42,
    )

    assert (out["tensor"] == mk.TorchTensorColumn([2, 4])).all()
    assert (out["pandas"] == mk.ScalarColumn([7, 6])).all()
    assert (out["numpy"] == mk.TorchTensorColumn([6, 7])).all()


def test_sample_df_w_weights(simple_df):
    out = simple_df.sample(
        n=2,
        weights=np.array([0.5, 0.1, 0.2, 0.2]),
        random_state=42,
    )
    assert (out["tensor"] == mk.TorchTensorColumn([1, 4])).all()
    assert (out["pandas"] == mk.ScalarColumn([8, 6])).all()
    assert (out["numpy"] == mk.TorchTensorColumn([4, 7])).all()


def test_sample_df_w_weights_as_str(simple_df):
    out = simple_df.sample(
        n=2,
        weights="tensor",
        random_state=42,
    )

    assert (out["tensor"] == mk.TorchTensorColumn([3, 4])).all()
    assert (out["pandas"] == mk.ScalarColumn([9, 6])).all()
    assert (out["numpy"] == mk.TorchTensorColumn([5, 7])).all()


def test_column(simple_column):
    out = simple_column.sample(
        n=2,
        random_state=42,
    )
    assert (out == mk.TorchTensorColumn([6, 7])).all()


def test_column_w_weights_as_str(simple_column):
    with pytest.raises(ValueError):
        sample(
            simple_column,
            n=2,
            weights="tensor",
            random_state=42,
        )
