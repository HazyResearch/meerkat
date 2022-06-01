import meerkat as mk
import os
import torch
import numpy as np
import pandas as pd

def test_demo():
    dp = mk.DataPanel({
    'tensor': mk.TensorColumn([3,1,2]),
    'pandas':  mk.PandasSeriesColumn([9,8,7]),
    'numpy': mk.NumpyArrayColumn([5,4,6]), 
    }) 

    # test sort with single tensor column 
    test1 = dp.sort(by=["tensor"])
    assert test1["tensor"] == mk.TensorColumn([1,2,3])

    # test sort with single pandas column 
    test2 = dp.sort(by=["pandas"])
    test2["pandas"] = test2["pandas"].reset_index()
    assert test2["pandas"] == mk.PandasSeriesColumn([7,8,9])

    # test sort with single numpy column descending
    test3 = dp.sort(by=["numpy"], ascending=False)
    assert test3["numpy"] == mk.NumpyArrayColumn([1,5,4])


    # test sort with multiple columns (numpy and tensor)

    # test sort with multiple columns (pandas and tensor)

    # test sort with multiple columns (pandas and tensor descending)

    