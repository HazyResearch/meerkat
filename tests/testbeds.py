"""A collection of simple testbeds to build test cases."""
import os
from functools import wraps
from itertools import product
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from meerkat.columns.deferred.file import FileColumn

# from meerkat.columns.deferred.image import ImageColumn
from meerkat.columns.object.base import ObjectColumn
from meerkat.dataframe import DataFrame


class AbstractColumnTestBed:
    DEFAULT_CONFIG = {}

    @classmethod
    def get_params(cls, config: dict = None, params: dict = None):
        updated_config = cls.DEFAULT_CONFIG.copy()
        if config is not None:
            updated_config.update(config)
        configs = list(
            map(
                dict,
                product(*[[(k, v) for v in vs] for k, vs in updated_config.items()]),
            )
        )
        if params is None:
            return {
                "argnames": "config",
                "argvalues": configs,
                "ids": [str(config) for config in configs],
            }
        else:
            argvalues = list(product(configs, *params.values()))
            return {
                "argnames": "config," + ",".join(params.keys()),
                "argvalues": argvalues,
                "ids": [",".join(map(str, values)) for values in argvalues],
            }

    @classmethod
    @wraps(pytest.mark.parametrize)
    def parametrize(cls, config: dict = None, params: dict = None):
        return pytest.mark.parametrize(**cls.get_params(config=config, params=params))


class MockDatapanel:
    def __init__(
        self,
        length: int,
        use_visible_rows: bool = False,
        use_visible_columns: bool = False,
        include_image_column: bool = False,
        tmpdir: str = None,
    ):
        batch = {
            "a": np.arange(length),
            "b": ObjectColumn(np.arange(length)),
            "c": [{"a": 2}] * length,
            "d": torch.arange(length),
            # offset the index to test robustness to nonstandard indices
            "e": pd.Series(np.arange(length), index=np.arange(1, 1 + length)),
            # test multidimensional
            "f": np.ones((length, 5)).astype(int),
            "g": torch.ones(length, 5).to(int),
        }

        if include_image_column:
            assert tmpdir is not None
            self.img_col = MockImageColumn(length=length, tmpdir=tmpdir)
            batch["img"] = self.img_col.col

        self.df = DataFrame.from_batch(batch)

        self.visible_rows = [0, 4, 6, 11] if use_visible_rows else np.arange(length)
        if use_visible_rows:
            for column in self.df.values():
                column.visible_rows = self.visible_rows

        self.visible_columns = ["a", "b"] if use_visible_columns else self.df.columns
        if use_visible_columns:
            self.df.visible_columns = self.visible_columns


class MockColumn:
    def __init__(
        self,
        use_visible_rows: bool = False,
        col_type: type = ObjectColumn,
        dtype: str = "int",
    ):
        self.array = np.arange(16, dtype=dtype)
        self.col = col_type(self.array)

        if use_visible_rows:
            self.visible_rows = np.array([0, 4, 6, 11])
            self.col.visible_rows = self.visible_rows
        else:
            self.visible_rows = np.arange(16)


class MockStrColumn:
    def __init__(self, use_visible_rows: bool = False, col_type: type = ObjectColumn):
        self.array = np.array([f"row_{idx}" for idx in range(16)])
        self.col = col_type(self.array)

        if use_visible_rows:
            self.visible_rows = np.array([0, 4, 6, 11])
            self.col.visible_rows = self.visible_rows
        else:
            self.visible_rows = np.arange(16)


class MockAnyColumn:
    def __init__(
        self,
        data: Sequence,
        use_visible_rows: bool = False,
        col_type: type = ObjectColumn,
    ):
        self.array = data
        self.col = col_type(self.array)

        if use_visible_rows:
            self.visible_rows = [0, 4, 6, 11]
            self.col.visible_rows = self.visible_rows
        else:
            self.visible_rows = np.arange(16)


class MockImageColumn:
    def __init__(self, length: int, tmpdir: str):
        """[summary]

        Args:
            wrap_dataset (bool, optional): If `True`, create a
            `meerkat.DataFrame`
            ,
                otherwise create a
                `meerkat.core.dataformats.vision.VisionDataPane`
                Defaults to False.
        """
        self.image_paths = []
        self.image_arrays = []
        self.images = []

        for i in range(0, length):
            self.image_paths.append(os.path.join(tmpdir, "{}.png".format(i)))
            self.image_arrays.append((i * np.ones((10, 10, 3))).astype(np.uint8))
            im = Image.fromarray(self.image_arrays[-1])
            im.save(self.image_paths[-1])

        self.col = FileColumn(self.image_paths)
