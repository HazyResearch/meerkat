import os
import pickle
from functools import wraps
from itertools import product

import numpy as np
import pandas as pd
import pytest

from meerkat.datapanel import DataPanel
from meerkat.ops.concat import concat

from .abstract import column_parametrize
from .test_numpy_column import NumpyArrayColumnTestBed
from .test_pandas_column import PandasSeriesColumnTestBed


@pytest.fixture(
    **column_parametrize([NumpyArrayColumnTestBed, PandasSeriesColumnTestBed])
)
def column_testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


def test_1(column_testbed):
    assert True
