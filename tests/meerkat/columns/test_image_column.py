"""Unittests for NumpyColumn."""
from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.datasets.folder as folder
from PIL import Image
from torchvision.transforms.functional import to_tensor

import meerkat
from meerkat import ImageColumn
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.file_column import FileCell
from meerkat.columns.lambda_column import LambdaCell
from meerkat.columns.list_column import ListColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.columns.tensor_column import TensorColumn

from .abstract import AbstractColumnTestBed, TestAbstractColumn


class ImageColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "transform": [True, False],
        "use_base_dir": [True, False],
    }

    def __init__(
        self,
        tmpdir: str,
        length: int = 16,
        transform: bool = False,
        use_base_dir: bool = False,
        seed: int = 123,
    ):
        self.image_paths = []
        self.image_arrays = []
        self.ims = []
        self.data = []

        transform = to_tensor if transform else None

        self.base_dir = tmpdir if use_base_dir else None

        for i in range(0, length):
            self.image_arrays.append((i * np.ones((4, 4, 3))).astype(np.uint8))
            im = Image.fromarray(self.image_arrays[-1])
            self.ims.append(im)
            self.data.append(transform(im) if transform else im)
            filename = "{}.png".format(i)
            im.save(os.path.join(tmpdir, filename))
            if use_base_dir:
                self.image_paths.append(filename)
            else:
                self.image_paths.append(os.path.join(tmpdir, filename))

        if transform is not None:
            self.data = torch.stack(self.data)
        self.transform = transform
        self.col = ImageColumn.from_filepaths(
            self.image_paths,
            transform=transform,
            loader=folder.default_loader,
            base_dir=self.base_dir,
        )

    def get_map_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        kwarg: int = 0,
        salt: int = 1,
    ):
        if not materialize:
            if batched:
                return {"fn": lambda x, k=0: x, "expected_result": self.col}
            else:
                # can't check for cell column equivalence because the `fn` is a bound
                # method of different objects (since we perform batching then convert)
                # non-batched fns to batched functions, so we call get
                if self.transform is None:
                    return {
                        "fn": lambda x, k=0: x.get().rotate(45 + salt + k),
                        "expected_result": ListColumn(
                            [im.rotate(45 + salt + kwarg) for im in self.ims]
                        ),
                    }
                else:
                    return {
                        "fn": lambda x, k=0: x.get() + salt + k,
                        "expected_result": TensorColumn(
                            torch.stack([self.transform(im) for im in self.ims])
                            + salt
                            + kwarg
                        ),
                    }
        else:
            if self.transform is None:
                return {
                    "fn": (lambda x, k=0: [im.rotate(45 + salt + k) for im in x])
                    if batched
                    else (lambda x, k=0: x.rotate(45 + salt + k)),
                    "expected_result": ListColumn(
                        [im.rotate(45 + salt + kwarg) for im in self.ims]
                    ),
                }
            else:
                return {
                    "fn": lambda x, k=0: x + salt + k,
                    "expected_result": TensorColumn(
                        torch.stack([self.transform(im) for im in self.ims])
                        + salt
                        + kwarg
                    ),
                }

    def get_filter_spec(
        self,
        batched: bool = True,
        materialize: bool = False,
        salt: int = 1,
        kwarg: int = 0,
    ):
        if not materialize:
            if batched:
                return {
                    "fn": lambda x, k=0: [
                        int(os.path.splitext(os.path.basename(cell.data))[0])
                        < (4 + salt + k)
                        for cell in x.lz
                    ],
                    "expected_result": self.col.lz[: 4 + salt + kwarg],
                }
            else:
                return {
                    "fn": (
                        lambda x, k=0: int(
                            os.path.splitext(os.path.basename(x.data))[0]
                        )
                        < (4 + salt + k)
                    ),
                    "expected_result": self.col.lz[: 4 + salt + kwarg],
                }
        else:
            if self.transform is None:
                return {
                    "fn": (lambda x, k=0: [im.rotate(45 + salt + k) for im in x])
                    if batched
                    else (lambda x, k=0: x.rotate(45 + salt + k)),
                    "expected_result": ListColumn(
                        [im.rotate(45 + salt + kwarg) for im in self.ims]
                    ),
                }
            else:
                return {
                    "fn": lambda x, k=0: (
                        (x.mean(dim=[1, 2, 3]) if batched else x.mean()) > salt + k
                    ).to(bool),
                    "expected_result": self.col.lz[
                        torch.stack([self.transform(im) for im in self.ims])
                        .mean(dim=[1, 2, 3])
                        .numpy()
                        > salt + kwarg
                    ],
                }

    def get_data(self, index, materialize: bool = True):
        if materialize:
            if isinstance(index, int):
                return self.data[index]

            if self.transform is not None:
                return self.data[index]
            else:
                index = np.arange(len(self.data))[index]
                return [self.data[idx] for idx in index]
        else:
            if isinstance(index, int):
                return FileCell(
                    data=self.image_paths[index],
                    loader=self.col.loader,
                    transform=self.col.transform,
                    base_dir=self.base_dir,
                )
            index = np.arange(len(self.data))[index]
            return PandasSeriesColumn([self.image_paths[idx] for idx in index])

    @staticmethod
    def assert_data_equal(
        data1: Union[Image.Image, AbstractColumn, List, torch.Tensor],
        data2: Union[Image.Image, AbstractColumn, List, torch.Tensor],
    ):
        if isinstance(data1, Image.Image) or isinstance(data1, List):
            assert data1 == data2
        elif isinstance(data1, AbstractColumn):
            assert data1.is_equal(data2)
        elif torch.is_tensor(data1):
            assert (data1 == data2).all()
        elif isinstance(data1, LambdaCell):
            assert data1 == data2
        else:
            raise ValueError(
                "Cannot assert data equal between objects type:"
                f" {type(data1), type(data2)}"
            )


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

    @ImageColumnTestBed.parametrize(single=True, params={"index_type": [np.ndarray]})
    def test_set_item(self, testbed, index_type: type):
        with pytest.raises(ValueError, match="Cannot setitem on a `LambdaColumn`."):
            testbed.col[0] = 0

    @ImageColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_getitem(self, testbed, index_type: type):
        return super().test_getitem(testbed, index_type=index_type)

    @ImageColumnTestBed.parametrize(
        config={"transform": [True]},
        params={"batched": [True, False], "materialize": [True, False]},
    )
    def test_filter_1(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        return super().test_filter_1(testbed, batched, materialize=materialize)

    @ImageColumnTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_map_return_multiple(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        return super().test_map_return_multiple(
            testbed, batched, materialize=materialize
        )

    @ImageColumnTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_map_return_single(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        return super().test_map_return_single(testbed, batched, materialize)

    @ImageColumnTestBed.parametrize(
        params={"batched": [True, False], "materialize": [True, False]}
    )
    def test_map_return_single_w_kwarg(
        self, testbed: AbstractColumnTestBed, batched: bool, materialize: bool
    ):
        return super().test_map_return_single_w_kwarg(testbed, batched, materialize)

    @ImageColumnTestBed.parametrize(params={"n": [1, 2, 3]})
    def test_concat(self, testbed: AbstractColumnTestBed, n: int):
        return super().test_concat(testbed, n=n)

    @ImageColumnTestBed.parametrize()
    def test_copy(self, testbed: AbstractColumnTestBed):
        return super().test_copy(testbed)

    @ImageColumnTestBed.parametrize()
    def test_io(self, tmp_path, testbed):
        # uses the tmp_path fixture which will provide a
        # temporary directory unique to the test invocation,
        # important for dataloader
        col, _ = testbed.col, testbed.data

        path = os.path.join(tmp_path, "test")
        col.write(path)

        new_col = self.column_class.read(path)

        assert isinstance(new_col, self.column_class)
        # can't check if the functions are the same since they point to different
        # methods
        assert col.data.is_equal(new_col.data)

    @ImageColumnTestBed.parametrize()
    def test_pickle(self, testbed):
        super().test_pickle(testbed)

    @ImageColumnTestBed.parametrize(params={"max_rows": [6, 16, 20]})
    def test_repr_pandas(self, testbed, max_rows):
        meerkat.config.display.max_rows = max_rows
        series, _ = testbed.col._repr_pandas_()
        assert isinstance(series, pd.Series)
        assert len(series) == min(len(series), max_rows + 1)
