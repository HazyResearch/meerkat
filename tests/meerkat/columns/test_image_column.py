"""Unittests for NumpyColumn."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from attr import s
from PIL import Image
from torchvision.transforms.functional import to_tensor

from meerkat import ImageColumn
from meerkat.columns.list_column import ListColumn
from meerkat.columns.tensor_column import TensorColumn

from .abstract import AbstractColumnTestBed, TestAbstractColumn


class ImageColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {"transform": [True, False]}

    def __init__(
        self,
        tmpdir: str,
        length: int = 16,
        transform: bool = False,
        seed: int = 123,
    ):
        self.image_paths = []
        self.image_arrays = []
        self.ims = []
        self.data = []

        transform = to_tensor if transform else None

        for i in range(0, length):
            self.image_paths.append(os.path.join(tmpdir, "{}.png".format(i)))
            self.image_arrays.append((i * np.ones((4, 4, 3))).astype(np.uint8))
            im = Image.fromarray(self.image_arrays[-1])
            self.ims.append(im)
            self.data.append(transform(im) if transform else im)
            im.save(self.image_paths[-1])

        self.transform = transform
        self.col = ImageColumn.from_filepaths(self.image_paths, transform=transform)

    def get_map_spec(
        self, key: str = "map1", batched: bool = True, materialize: bool = False
    ):
        if not materialize:
            pass
        else:
            if self.transform is None:
                return {
                    "fn": lambda x: x.rotate(45),
                    "expected_result": ListColumn([im.rotate(45) for im in self.ims]),
                }
            else:
                print(np.stack(self.image_arrays).shape)
                return {
                    "fn": lambda x: x + 1,
                    "expected_result": TensorColumn(np.stack(self.image_arrays) + 1),
                }

    def get_data(self, index):
        return self.data[index]

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        assert (data1 == data2).all()


@pytest.fixture
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


class TestImageColumn(TestAbstractColumn):

    __test__ = True
    testbed_class: type = ImageColumnTestBed
    column_class: type = ImageColumn

    def _get_data_to_set(self, testbed, data_index):
        return np.zeros_like(testbed.get_data(data_index))

    @ImageColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_set_item(self, testbed, index_type: type):
        return super().test_set_item(testbed, index_type=index_type)

    @ImageColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_getitem(self, testbed, index_type: type):
        return super().test_getitem(testbed, index_type=index_type)

    @ImageColumnTestBed.parametrize(
        config={"num_dims": [1], "dim_length": [1]}, params={"batched": [True, False]}
    )
    def test_filter_1(self, testbed: AbstractColumnTestBed, batched: bool):
        return super().test_filter_1(testbed, batched)

    @ImageColumnTestBed.parametrize(params={"batched": [True, False]})
    def test_map_return_multiple(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        return super().test_map_return_multiple(testbed, batched)

    @ImageColumnTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True]}
    )
    def test_map_return_single(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        return super().test_map_return_single(testbed, batched, materialize)

    @ImageColumnTestBed.parametrize()
    def test_copy(self, testbed: AbstractColumnTestBed):
        return super().test_copy(testbed)

    @ImageColumnTestBed.parametrize()
    def test_io(self, tmp_path, testbed):
        super().test_io(tmp_path, testbed)

    @ImageColumnTestBed.parametrize()
    def test_pickle(self, testbed):
        super().test_pickle(testbed)

    @ImageColumnTestBed.parametrize()
    def test_to_tensor(self, testbed):
        col, _ = testbed.col, testbed.data

        tensor = col.to_tensor()

        assert torch.is_tensor(tensor)
        assert (col == tensor.numpy()).all()

    def test_from_array(self):
        # Build a dataset from a batch
        array = np.random.rand(10, 3, 3)
        col = NumpyArrayColumn.from_array(array)

        assert (col == array).all()
        np_test.assert_equal(len(col), 10)

    @ImageColumnTestBed.parametrize()
    def test_to_pandas(self, testbed):
        series = testbed.col.to_pandas()

        assert isinstance(series, pd.Series)

        if testbed.col.shape == 1:
            assert (series.values == testbed.col.data).all()
        else:
            for idx in range(len(testbed.col)):
                assert (series.iloc[idx] == testbed.col[idx]).all()

    @ImageColumnTestBed.parametrize()
    def test_repr_pandas(self, testbed):
        series = testbed.col.to_pandas()
        assert isinstance(series, pd.Series)
