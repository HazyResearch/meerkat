import meerkat as mk
import os
import torch
import numpy as np
import pandas as pd
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Set,
    Tuple,
    Union,
)

# helper function, returns test dp
def make_test_dp(by: Union[str, List[str]],
                  ascending: Union[bool, List[bool]] = True,):
   dp = mk.DataPanel({
      'tensor': mk.TensorColumn([3,1,2]),
      'pandas':  mk.PandasSeriesColumn([9,8,7]),
      'numpy': mk.NumpyArrayColumn([5,4,6]), 
      }) 
   test = dp.sort(by=by, ascending=ascending)
   return test

######## SINGLE COLUMN TESTS ########

# Testing all columns after sorting by an ascending tensor column
def test_sort_by_ascending_tensor_column():
    test = make_test_dp(by=["tensor"])
    assert (test["tensor"] == mk.TensorColumn([1,2,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,7,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,6,5])).all()
  

# Testing all columns after sorting by an ascending pandas column
def test_sort_by_ascending_pandas_on_pandas_column():
    test = make_test_dp(by=["pandas"])
    assert (test["tensor"] == mk.TensorColumn([2,1,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([7,8,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([6,4,5])).all()

# Testing all columns after sorting by an ascending numpy column
def test_sort_single_numpy_column_ascending():
    test = make_test_dp(by=["numpy"])
    assert (test["tensor"] == mk.TensorColumn([1,3,2])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,9, 7])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,5, 6])).all()

######## SINGLE COLUMN TESTS DESCENDING ########
# Testing all columns after sorting by a descending tensor column
def test_sort_single_tensor_column_descending():
    test = make_test_dp(by=["tensor"], ascending=False)
    assert (test["tensor"] == mk.TensorColumn([3,2,1])).all() and (test["pandas"] == mk.PandasSeriesColumn([9,7,8])).all() and (test["numpy"] == mk.NumpyArrayColumn([5,6,4])).all()


# Testing all columns after sorting by a descending pandas column
def test_sort_single_pandas_column_descending():
    test = make_test_dp(by=["pandas"], ascending=False)
    assert (test["tensor"] == mk.TensorColumn([3,1,2])).all() and (test["pandas"] == mk.PandasSeriesColumn([9,8,7])).all() and (test["numpy"] == mk.NumpyArrayColumn([5,4,6])).all()


# Testing all columns after sorting by a descending numpy column
def test_sort_single_numpy_column_descending():
    test = make_test_dp(by=["numpy"], ascending=False)
    assert (test["tensor"] == mk.TensorColumn([2,3,1])).all() and (test["pandas"] == mk.PandasSeriesColumn([7,9,8])).all() and (test["numpy"] == mk.NumpyArrayColumn([6,5,4])).all()


######## MULTIPLE COLUMN TESTS ########

# Testing all columns after sorting with multiple ascending columns (numpy and tensor)
def test_sort_numpy_and_tensor_ascending():
    test = make_test_dp(by=["numpy", "tensor"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([1,3,2])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,9, 7])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,5, 6])).all()
    

# Testing all columns after sorting with multiple ascending columns (numpy and pandas)
def test_sort_numpy_and_pandas_ascending():
    test = make_test_dp(by=["numpy", "pandas"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([1,3,2])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,9, 7])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,5, 6])).all()


# Testing all columns after sorting with multiple ascending columns (numpy and pandas and tensor)
def test_sort_numpy_and_pandas_and_tensor_ascending():
    test = make_test_dp(by=["numpy", "pandas", "tensor"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([1,3,2])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,9, 7])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,5, 6])).all()
        

# Testing all columns after sorting with multiple ascending columns (tensor and numpy)
def test_sort_tensor_and_numpy_ascending():
    test = make_test_dp(by=["tensor", "numpy"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([1,2,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,7,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,6,5])).all()
        

# Testing all columns after sorting with multiple ascending columns (tensor and pandas)
def test_sort_tensor_and_pandas_ascending():
    test = make_test_dp(by=["tensor", "pandas"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([1,2,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,7,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,6,5])).all()
        
# Testing all columns after sorting with multiple ascending columns (tensor and pandas, numpy)
def test_sort_tensor_and_pandas_and_numpy_ascending():
    test = make_test_dp(by=["tensor", "pandas", "numpy"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([1,2,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([8,7,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([4,6,5])).all()
        

# Testing all columns after sorting with multiple ascending columns (pandas and numpy)
def test_sort_pandas_and_numpy_ascending():
    test = make_test_dp(by=["pandas", "numpy"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([2,1,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([7,8,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([6,4,5])).all()

# Testing all columns after sorting with multiple ascending columns (pandas and tensor)
def test_sort_pandas_and_tensor_ascending():
    test = make_test_dp(by=["pandas", "tensor"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([2,1,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([7,8,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([6,4,5])).all()
        
# Testing all columns after sorting with multiple ascending columns (pandas and tensor and numpy)
def test_sort_pandas_and_tensor_and_numpy_ascending():
    test = make_test_dp(by=["pandas", "tensor", "numpy"], ascending=True)
    assert (test["tensor"] == mk.TensorColumn([2,1,3])).all() and (test["pandas"] == mk.PandasSeriesColumn([7,8,9])).all() and (test["numpy"] == mk.NumpyArrayColumn([6,4,5])).all()
        
# Testing all columns after sorting with multiple ascending columns (pandas and tensor)
def test_sort_pandas_and_tensor_descending():
    test = make_test_dp(by=["pandas", "tensor"], ascending=False)
    assert (test["tensor"] == mk.TensorColumn([3,1,2])).all() and (test["pandas"] == mk.PandasSeriesColumn([9,8,7])).all() and (test["numpy"] == mk.NumpyArrayColumn([5,4,6])).all()
        
    
    