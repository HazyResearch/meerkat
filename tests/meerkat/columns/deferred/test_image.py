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
from meerkat.block.deferred_block import DeferredCellOp, DeferredOp
from meerkat.columns.abstract import Column
from meerkat.columns.deferred.base import DeferredCell
from meerkat.columns.deferred.file import FileCell
from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.scalar import ScalarColumn
from meerkat.columns.tensor.torch import TorchTensorColumn

from ....utils import product_parametrize
from ..abstract import AbstractColumnTestBed, column_parametrize


class ImageColumnTestBed(AbstractColumnTestBed):
    DEFAULT_CONFIG = {
        "transform": [True, False],
        "use_base_dir": [True, False],
    }

    marks = pytest.mark.image_col

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
                        "expected_result": ObjectColumn(
                            [im.rotate(45 + salt + kwarg) for im in self.ims]
                        ),
                    }
                else:
                    return {
                        "fn": lambda x, k=0: x.get() + salt + k,
                        "expected_result": TorchTensorColumn(
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
                    "expected_result": ObjectColumn(
                        [im.rotate(45 + salt + kwarg) for im in self.ims]
                    ),
                }
            else:
                return {
                    "fn": lambda x, k=0: x + salt + k,
                    "expected_result": TorchTensorColumn(
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
                        for cell in x
                    ],
                    "expected_result": self.col[: 4 + salt + kwarg],
                }
            else:
                return {
                    "fn": (
                        lambda x, k=0: int(
                            os.path.splitext(os.path.basename(x.data))[0]
                        )
                        < (4 + salt + k)
                    ),
                    "expected_result": self.col[: 4 + salt + kwarg],
                }
        else:
            if self.transform is None:
                return {
                    "fn": (lambda x, k=0: [im.rotate(45 + salt + k) for im in x])
                    if batched
                    else (lambda x, k=0: x.rotate(45 + salt + k)),
                    "expected_result": ObjectColumn(
                        [im.rotate(45 + salt + kwarg) for im in self.ims]
                    ),
                }
            else:
                return {
                    "fn": lambda x, k=0: (
                        (x.mean(dim=[1, 2, 3]) if batched else x.mean()) > salt + k
                    ).to(bool),
                    "expected_result": self.col[
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
                    DeferredCellOp(
                        args=[self.image_paths[index]],
                        kwargs={},
                        fn=self.col.fn,
                        is_batched_fn=False,
                        return_index=None,
                    )
                )

            index = np.arange(len(self.data))[index]
            col = ScalarColumn([self.image_paths[idx] for idx in index])
            return DeferredOp(
                args=[col], kwargs={}, fn=self.col.fn, is_batched_fn=False, batch_size=1
            )

    @staticmethod
    def assert_data_equal(
        data1: Union[Image.Image, Column, List, torch.Tensor],
        data2: Union[Image.Image, Column, List, torch.Tensor],
    ):
        if isinstance(data1, Image.Image) or isinstance(data1, List):
            assert data1 == data2
        elif isinstance(data1, Column):
            assert data1.is_equal(data2)
        elif torch.is_tensor(data1):
            print(data2)
            assert (data1 == data2).all()
        elif isinstance(data1, DeferredCell):
            assert data1 == data2
        elif isinstance(data1, DeferredOp):
            assert data1.is_equal(data2)
        else:
            raise ValueError(
                "Cannot assert data equal between objects type:"
                f" {type(data1), type(data2)}"
            )


@pytest.fixture(**column_parametrize([ImageColumnTestBed]))
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


@product_parametrize(params={"max_rows": [6, 16, 20]})
def test_repr_pandas(testbed, max_rows):
    meerkat.config.display.max_rows = max_rows
    series, _ = testbed.col._repr_pandas_()
    assert isinstance(series, pd.Series)
    assert len(series) == min(len(series), max_rows + 1)


def test_repr_when_transform_produces_invalid_image(testbed):
    from torchvision.transforms import ToTensor

    def mean_transform(cell):
        return ToTensor()(cell).mean(dim=[1, 2])

    testbed.col.transform = mean_transform
    testbed.col._repr_html_()
