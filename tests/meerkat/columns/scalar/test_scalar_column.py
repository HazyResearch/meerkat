import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch

from meerkat import ScalarColumn


@pytest.mark.parametrize(
    "data",
    [[1, 2, 3], np.asarray([1, 2, 3]), torch.tensor([1, 2, 3]), pd.Series([1, 2, 3])],
)
@pytest.mark.parametrize("backend", ["arrow", "pandas"])
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
