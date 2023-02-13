from typing import Dict
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


NUMERIC_COLUMNS = [
    np.array([1, 4, 6, 8]),
    np.array([1, 4, 6, 8], dtype=float),
]
BOOL_COLUMNS = [
    np.array([True, True, True]),
    np.array([True, False, True]),
    np.array([False, False, False]),
]


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS + BOOL_COLUMNS})
def test_mean(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.mean() == col.mean()


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS + BOOL_COLUMNS})
def test_mode(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.all(pd.Series(data).mode().values == col.mode().to_numpy())


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS})
def test_median(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.median(data) == col.median()


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS + BOOL_COLUMNS})
def test_min(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.min(data) == col.min()


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS + BOOL_COLUMNS})
def test_max(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.max(data) == col.max()


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS})
def test_var(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.var(data, ddof=1) == col.var()


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS})
def test_std(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert np.std(data, ddof=1) == col.std()


@product_parametrize({"backend": BACKENDS, "data": NUMERIC_COLUMNS + BOOL_COLUMNS})
def test_sum(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.sum() == col.sum()


@product_parametrize({"backend": BACKENDS, "data": BOOL_COLUMNS})
def test_any(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.prod() == col.product()


@product_parametrize({"backend": BACKENDS, "data": BOOL_COLUMNS})
def test_all(data: np.ndarray, backend: str):
    col = ScalarColumn(data, backend=backend)
    assert data.prod() == col.product()


NUMERIC_COLUMN_OPERANDS = [{"a": col, "b": col + 1} for col in NUMERIC_COLUMNS]

NUMERIC_SCALAR_OPERANDS = [{"a": col, "b": col[0].item()} for col in NUMERIC_COLUMNS]


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_add_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a + col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] + operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_add_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] + col_a
    else:
        out = col_a + operands["b"]
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] + operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_sub_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a - col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] - operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_sub_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] - col_a
        correct = operands["b"] - operands["a"]
    else:
        out = col_a - operands["b"]
        correct = operands["a"] - operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_mul_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a * col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] * operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_mul_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] * col_a
        correct = operands["b"] * operands["a"]
    else:
        out = col_a * operands["b"]
        correct = operands["a"] * operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_truediv_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a / col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] / operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_truediv_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] / col_a
        correct = operands["b"] / operands["a"]
    else:
        out = col_a / operands["b"]
        correct = operands["a"] / operands["b"]

    assert isinstance(out, ScalarColumn)
    breakpoint()
    assert out.equals(ScalarColumn(correct, backend=backend))
