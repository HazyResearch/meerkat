import os
from typing import Callable, Dict, Iterable, List, Set, Tuple, Union

import numpy as np
import pandas as pd

import meerkat as mk


def make_test_dp(
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
):
    """Helper function, returns test dp."""
    dp = mk.DataPanel(
        {
            "tensor": mk.TensorColumn([3, 1, 2]),
            "pandas": mk.PandasSeriesColumn([9, 8, 7]),
            "numpy": mk.NumpyArrayColumn([5, 4, 6]),
        }
    )
    test = dp.sort(by=by, ascending=ascending)
    return test


def make_tiebreaker_test_dp(
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
):
    dp = mk.DataPanel(
        {
            "tensor": mk.TensorColumn([3, 2, 1]),
            "pandas": mk.PandasSeriesColumn([9, 7, 9]),
            "numpy": mk.NumpyArrayColumn([4, 4, 6]),
        }
    )
    test = dp.sort(by=by, ascending=ascending)
    return test


# flake8: noqa
######## SINGLE COLUMN TESTS ########


def test_sort_by_ascending_tensor_column():
    """Testing all columns after sorting by an ascending tensor column."""

    test = make_test_dp(by=["tensor"])
    assert (
        (test["tensor"] == mk.TensorColumn([1, 2, 3])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([8, 7, 9])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([4, 6, 5])).all()
    )


def test_sort_by_ascending_pandas_on_pandas_column():
    """Testing all columns after sorting by an ascending pandas column."""

    test = make_test_dp(by=["pandas"])
    assert (
        (test["tensor"] == mk.TensorColumn([2, 1, 3])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([7, 8, 9])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([6, 4, 5])).all()
    )


def test_sort_single_numpy_column_ascending():
    """Testing all columns after sorting by an ascending numpy column."""

    test = make_test_dp(by=["numpy"])
    assert (
        (test["tensor"] == mk.TensorColumn([1, 3, 2])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([8, 9, 7])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([4, 5, 6])).all()
    )


# flake8: noqa
######## SINGLE COLUMN TESTS DESCENDING ########


def test_sort_single_tensor_column_descending():
    """Testing all columns after sorting by a descending tensor column."""

    test = make_test_dp(by=["tensor"], ascending=False)
    assert (
        (test["tensor"] == mk.TensorColumn([3, 2, 1])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([9, 7, 8])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([5, 6, 4])).all()
    )


def test_sort_single_pandas_column_descending():
    """Testing all columns after sorting by a descending pandas column."""
    test = make_test_dp(by=["pandas"], ascending=False)
    assert (
        (test["tensor"] == mk.TensorColumn([3, 1, 2])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([9, 8, 7])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([5, 4, 6])).all()
    )


def test_sort_single_numpy_column_descending():
    """Testing all columns after sorting by a descending numpy column."""
    test = make_test_dp(by=["numpy"], ascending=False)
    assert (
        (test["tensor"] == mk.TensorColumn([2, 3, 1])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([7, 9, 8])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([6, 5, 4])).all()
    )


######## MULTIPLE COLUMN TESTS ########


def test_sort_numpy_and_tensor_ascending():
    """# Testing all columns after sorting with multiple ascending columns
    (numpy and tensor)"""
    test = make_tiebreaker_test_dp(by=["numpy", "tensor"], ascending=True)
    assert (
        (test["tensor"] == mk.TensorColumn([2, 3, 1])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([7, 9, 9])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([4, 4, 6])).all()
    )


def test_sort_numpy_and_pandas_ascending():
    """Testing all columns after sorting with multiple ascending columns (numpy
    and tensor)"""
    test = make_tiebreaker_test_dp(by=["numpy", "pandas"], ascending=True)
    assert (
        (test["tensor"] == mk.TensorColumn([2, 3, 1])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([7, 9, 9])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([4, 4, 6])).all()
    )


def test_sort_numpy_and_pandas_and_tensor_ascending():
    """Testing all columns after sorting with multiple ascending columns (numpy
    and pandas and tensor)"""
    dp = mk.DataPanel(
        {
            "tensor": mk.TensorColumn([3, 2, 1]),
            "pandas": mk.PandasSeriesColumn([9, 7, 7]),
            "numpy": mk.NumpyArrayColumn([6, 4, 4]),
        }
    )
    test = dp.sort(by=["numpy", "pandas", "tensor"], ascending=True)
    assert (
        (test["tensor"] == mk.TensorColumn([1, 2, 3])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([7, 7, 9])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([4, 4, 6])).all()
    )


def test_sort_tensor_and_pandas_descending():
    """Testing all columns after sorting with multiple ascending columns
    (tensor and pandas)."""
    dp = mk.DataPanel(
        {
            "tensor": mk.TensorColumn([3, 2, 2]),
            "pandas": mk.PandasSeriesColumn([9, 8, 7]),
            "numpy": mk.NumpyArrayColumn([6, 4, 4]),
        }
    )
    test = dp.sort(by=["tensor", "pandas"], ascending=False)
    assert (
        (test["tensor"] == mk.TensorColumn([3, 2, 2])).all()
        and (test["pandas"] == mk.PandasSeriesColumn([9, 8, 7])).all()
        and (test["numpy"] == mk.NumpyArrayColumn([6, 4, 4])).all()
    )
