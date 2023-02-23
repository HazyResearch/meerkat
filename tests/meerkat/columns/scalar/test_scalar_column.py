import itertools
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from meerkat import ScalarColumn
from meerkat.dataframe import DataFrame
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
NAN_COLUMNS = [
    np.array([np.nan, 1, 2]),
    np.array([1, 4, 3], dtype=float),
    np.array([1, np.nan, 3], dtype=float),
    np.array([np.nan, np.nan, np.nan]),
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
    if backend == "arrow":
        with pytest.warns(UserWarning):
            assert np.median(data) == col.median()
    else:
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

BOOL_COLUMN_OPERANDS = [
    {"a": col_a, "b": col_b} for col_a, col_b in itertools.combinations(BOOL_COLUMNS, 2)
]

BOOL_SCALAR_OPERANDS = [
    {"a": col_a, "b": col_b[0].item()}
    for col_a, col_b in itertools.combinations(BOOL_COLUMNS, 2)
]


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
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_floordiv_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a // col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] // operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_floordiv_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] // col_a
        correct = operands["b"] // operands["a"]
    else:
        out = col_a // operands["b"]
        correct = operands["a"] // operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": ["pandas"], "operands": NUMERIC_COLUMN_OPERANDS})
def test_mod_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a % col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] % operands["b"], backend=backend))


@product_parametrize(
    {"backend": ["pandas"], "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_mod_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] % col_a
        correct = operands["b"] % operands["a"]
    else:
        out = col_a % operands["b"]
        correct = operands["a"] % operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_pow_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a**col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] ** operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_pow_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] ** col_a
        correct = operands["b"] ** operands["a"]
    else:
        out = col_a ** operands["b"]
        correct = operands["a"] ** operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_eq_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a == col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] == operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_eq_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] == col_a
        correct = operands["b"] == operands["a"]
    else:
        out = col_a == operands["b"]
        correct = operands["a"] == operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_gt_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a > col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] > operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_gt_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] > col_a
        correct = operands["b"] > operands["a"]
    else:
        out = col_a > operands["b"]
        correct = operands["a"] > operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_lt_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a < col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] < operands["b"], backend=backend))


@product_parametrize(
    {"backend": BACKENDS, "operands": NUMERIC_SCALAR_OPERANDS, "right": [True, False]}
)
def test_lt_scalar(backend: str, operands: Dict[str, np.array], right: bool):
    col_a = ScalarColumn(operands["a"], backend=backend)
    if right:
        out = operands["b"] < col_a
        correct = operands["b"] < operands["a"]
    else:
        out = col_a < operands["b"]
        correct = operands["a"] < operands["b"]

    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(correct, backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_and_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a & col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] & operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_and_scalar(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    out = col_a & operands["b"]
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] & operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_or_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a | col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] | operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_or_scalar(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    out = col_a | operands["b"]
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] | operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_xor_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    col_b = ScalarColumn(operands["b"], backend=backend)
    out = col_a ^ col_b
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] ^ operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_xor_scalar(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    out = col_a ^ operands["b"]
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(operands["a"] ^ operands["b"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": BOOL_COLUMN_OPERANDS})
def test_invert_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    out = ~col_a
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(~operands["a"], backend=backend))


@product_parametrize({"backend": BACKENDS, "operands": NUMERIC_COLUMN_OPERANDS})
def test_isin_column(backend: str, operands: Dict[str, np.array]):
    col_a = ScalarColumn(operands["a"], backend=backend)
    values = [operands["b"][0], operands["b"][1], operands["b"][2]]
    out = col_a.isin(values)
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(np.isin(operands["a"], values), backend=backend))


@product_parametrize({"backend": BACKENDS, "column": NAN_COLUMNS})
def test_isna(backend: str, column: np.array):
    col = ScalarColumn(column, backend=backend)
    out = col.isna()
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(np.isnan(column), backend=backend))


@product_parametrize({"backend": BACKENDS, "column": NAN_COLUMNS})
def test_isnull(backend: str, column: np.array):
    col = ScalarColumn(column, backend=backend)
    out = col.isnull()
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(np.isnan(column), backend=backend))


STRING_COLUMNS = [
    pd.Series(
        [
            "a asdsd ",
            "bfdidf.",
            "c asdasd dsd",
            "1290_dij",
            "d",
            " efdsdf ",
            "Fasdd asdasd",
            "pppqqqqq",
            "1290_dijaaa",
            "hl2dirf83WIW",
            "22222",
            "1290_disdj",
        ]
    ),
]


@product_parametrize(
    {
        "backend": BACKENDS,
        "column": STRING_COLUMNS,
        "compute_fn": [
            "capitalize",
            "isalnum",
            "isalpha",
            "isdecimal",
            "isdigit",
            "islower",
            "isnumeric",
            "isspace",
            "istitle",
            "isupper",
            "lower",
            "upper",
            "len",
            "lower",
            "swapcase",
            "title",
            "strip",
            "lstrip",
            "rstrip",
        ],
    }
)
def test_unary_str_methods(backend: str, column: pd.Series, compute_fn: str):
    col = ScalarColumn(column, backend=backend)
    out = getattr(col.str, compute_fn)()
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(getattr(column.str, compute_fn)(), backend=backend))


# @product_parameterize(
#     {
#         "backend": BACKENDS,
#         "column": STRING_COLUMNS,
#         "compute_fn": [
#         ]
#     }
# )
# def test_pattern_str_methods(backend: str, column: pd.Series, compute_fn: str):
#     col = ScalarColumn(column, backend=backend)
#     out = getattr(col.str, compute_fn)()
#     assert isinstance(out, ScalarColumn)
#     assert out.equals(
#         ScalarColumn(getattr(column.str, compute_fn)(), backend=backend)
#     )


@product_parametrize(
    {
        "backend": BACKENDS,
        "column": STRING_COLUMNS,
    }
)
def test_center(backend: str, column: pd.Series):
    col = ScalarColumn(column, backend=backend)
    out = col.str.center(20)
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(column.str.center(20), backend=backend))


@product_parametrize({"backend": BACKENDS, "column": STRING_COLUMNS, "n": [-1, 1, 2]})
def test_split(backend: str, column: pd.Series, n: int):
    col = ScalarColumn(column, backend=backend)
    out = col.str.split(" ", n=n)
    assert isinstance(out, DataFrame)
    correct_df = DataFrame(
        {
            str(name): ScalarColumn(col, backend=backend)
            for name, col in column.str.split(" ", n=n, expand=True).items()
        }
    )
    assert correct_df.columns == out.columns
    for name in correct_df.columns:
        assert correct_df[name].equals(out[name])


@product_parametrize({"backend": BACKENDS, "column": STRING_COLUMNS, "n": [-1, 1, 2]})
def test_rsplit(backend: str, column: pd.Series, n: int):
    col = ScalarColumn(column, backend=backend)
    out = col.str.rsplit(" ", n=n)
    assert isinstance(out, DataFrame)
    correct_df = DataFrame(
        {
            str(name): ScalarColumn(col, backend=backend)
            for name, col in column.str.rsplit(" ", n=n, expand=True).items()
        }
    )
    assert correct_df.columns == out.columns
    for name in correct_df.columns:
        assert correct_df[name].equals(out[name])


@product_parametrize(
    {
        "backend": BACKENDS,
        "column": STRING_COLUMNS,
    }
)
def test_center(backend: str, column: pd.Series):
    col = ScalarColumn(column, backend=backend)
    out = col.str.startswith("1290")
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(column.str.startswith("1290"), backend=backend))


@product_parametrize(
    {
        "backend": BACKENDS,
        "column": STRING_COLUMNS,
    }
)
def test_replace(backend: str, column: pd.Series):
    col = ScalarColumn(column, backend=backend)
    out = col.str.replace("di", "do")
    assert isinstance(out, ScalarColumn)
    assert out.equals(ScalarColumn(column.str.replace("di", "do"), backend=backend))
