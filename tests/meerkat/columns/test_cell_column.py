from typing import Collection, List, Union

import numpy as np
import pandas as pd
import pytest

from meerkat import NumpyArrayColumn
from meerkat.cells.abstract import AbstractCell
from meerkat.columns.cell_column import CellColumn

from .abstract import AbstractColumnTestBed, column_parametrize


class SimpleCell(AbstractCell):
    def __init__(
        self,
        data: str,
        transform: callable = None,
    ):
        super(AbstractCell, self).__init__()
        self.data = data
        self.transform = transform

    def get(self, *args, **kwargs):
        return self.transform(self.data)

    @classmethod
    def _state_keys(cls) -> Collection:
        return {"loader", "data"}

    def __eq__(self, other):
        return other.data == self.data and other.transform == self.transform


def add_one(x):
    return x + 1


class CellColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {}
    marks = pytest.mark.cell_col

    def __init__(
        self,
        length: int = 16,
        seed: int = 123,
        tmpdir: str = None,
    ):
        np.random.seed(seed)
        array = np.random.random(length) * 10
        self.data = array
        self.cells = [SimpleCell(data=data, transform=add_one) for data in array]
        self.col = CellColumn.from_cells(self.cells)

    def get_filter_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        kwarg: int = 0,
        salt: int = 1,
    ):
        if materialize:
            to_int = (lambda x: x.astype(int)) if batched else int
            return {
                "fn": lambda x, k=0: to_int(x + salt + k) % 2 == 0,
                "expected_result": self.col.lz[
                    (self.data + 1 + salt + kwarg).astype(int) % 2 == 0
                ],
            }
        else:
            if batched:
                return {
                    "fn": lambda x, k=0: (
                        np.array([c.data for c in x]).astype(int) + salt + k
                    )
                    % 2
                    == 0,
                    "expected_result": self.col.lz[
                        (self.data + salt + kwarg).astype(int) % 2 == 0
                    ],
                }
            else:
                return {
                    "fn": lambda x, k=0: int(x.data + salt + k) % 2 == 0,
                    "expected_result": self.col.lz[
                        (self.data + salt + kwarg).astype(int) % 2 == 0
                    ],
                }

    def get_map_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        kwarg: int = 0,
        salt: int = 1,
    ):
        if materialize:
            return {
                "fn": lambda x, k=0: x + salt + k,
                "expected_result": NumpyArrayColumn.from_array(
                    self.data + 1 + salt + kwarg
                ),
            }
        else:
            if batched:
                return {
                    "fn": lambda x, k=0: np.array([cell.data for cell in x.data])
                    + salt
                    + k,
                    "expected_result": NumpyArrayColumn.from_array(
                        self.data + salt + kwarg
                    ),
                }
            else:
                return {
                    "fn": lambda x, k=0: x.data + salt + k,
                    "expected_result": NumpyArrayColumn.from_array(
                        self.data + salt + kwarg
                    ),
                }

    def get_data(self, index, materialize=True):
        if materialize:
            return self.data[index] + 1
        else:
            if isinstance(index, int):
                return self.cells[index]
            else:
                index = np.arange(len(self.cells))[index]
                return [self.cells[idx] for idx in index]

    def get_data_to_set(self, data_index):
        data_index = self.col._translate_index(data_index)
        if isinstance(data_index, int):
            return SimpleCell(0, add_one)

        return [SimpleCell(idx, add_one) for idx in range(len(data_index))]

    @staticmethod
    def assert_data_equal(
        data1: Union[np.number, np.ndarray, List[AbstractCell], AbstractCell],
        data2: Union[np.number, np.ndarray, List[AbstractCell], AbstractCell],
    ):
        assert isinstance(data1, type(data2))
        if isinstance(data1, np.ndarray):
            assert (data1 == data2).all()
        else:
            assert data1 == data2


@pytest.fixture(**column_parametrize([CellColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)
