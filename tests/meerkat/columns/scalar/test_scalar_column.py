import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from meerkat import ScalarColumn
from tests.utils import product_parametrize

BACKENDS = ["arrow", "pandas"]


@pytest.mark.parametrize(
    "data",
    [[1, 2, 3], np.asarray([1, 2, 3]), torch.tensor([1, 2, 3]), pd.Series([1, 2, 3])],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_backend(data, backend: str):
    col = ScalarColumn(data, backend=backend)

    expected_type = {"arrow": (pa.Array, pa.ChunkedArray), "pandas": pd.Series}[backend]
    assert isinstance(col.data, expected_type)

    col_data = col.data
    if isinstance(col_data, torch.Tensor):
        col_data = col_data.numpy()
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    col_data = np.asarray(col_data)
    data = np.asarray(data)
    assert (col_data == data).all()


DATAS_W_BOOL = [
    np.array([1, 4, 6, 8]),
    np.array([1, 4, 6, 8], dtype=float),
    np.array([True, False, True]),
]
DATAS_WO_BOOL = DATAS_W_BOOL[:-1]
BOOL_DATA = [
    np.array([True, True, True]),
    np.array([True, False, True]),
    np.array([False, False, False]),
]


@product_parametrize({"backend": BACKENDS, "data": DATAS_W_BOOL})
def test_mean(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.mean() == col.mean()


@product_parametrize({"backend": BACKENDS, "data": DATAS_W_BOOL})
def test_mode(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.all(pd.Series(data).mode().values == col.mode().to_numpy())


@product_parametrize({"backend": BACKENDS, "data": DATAS_WO_BOOL})
def test_median(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.median(data) == col.median()


@product_parametrize({"backend": BACKENDS, "data": DATAS_W_BOOL})
def test_min(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.min(data) == col.min()


@product_parametrize({"backend": BACKENDS, "data": DATAS_W_BOOL})
def test_max(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.max(data) == col.max()


@product_parametrize({"backend": BACKENDS, "data": DATAS_WO_BOOL})
def test_var(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.var(data, ddof=1) == col.var()


@product_parametrize({"backend": BACKENDS, "data": DATAS_WO_BOOL})
def test_std(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.std(data, ddof=1) == col.std()


@product_parametrize({"backend": BACKENDS, "data": DATAS_W_BOOL})
def test_sum(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.sum() == col.sum()


@product_parametrize({"backend": BACKENDS, "data": BOOL_DATA})
def test_any(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.prod() == col.product()


@product_parametrize({"backend": BACKENDS, "data": BOOL_DATA})
def test_all(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.prod() == col.product()
