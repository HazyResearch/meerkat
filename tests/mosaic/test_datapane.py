"""Unittests for Datasets."""
import os
import shutil
from unittest import TestCase

import numpy as np
import torch

from robustnessgym.mosaic.datapane import DataPane
from robustnessgym.mosaic import NumpyArrayColumn
from tests.testbeds import MockTestBedv0


class TestDataPane(TestCase):
    def setUp(self):
        # Arrange
        pass

    def test_from_batch(self):
        # Build a dataset from a batch
        datapane = DataPane.from_batch(
            {
                "a": [1, 2, 3],
                "b": [True, False, True],
                "c": ["x", "y", "z"],
                "d": [{"e": 2}, {"e": 3}, {"e": 4}],
                "e": torch.ones(3),
                "f": np.ones(3),
            },
        )

        self.assertEqual(
            set(datapane.column_names), {"a", "b", "c", "d", "e", "f", "index"}
        )
        self.assertEqual(len(datapane), 3)
    
    def test_mixed_batched_map_return_single(self):
        datapane = DataPane.from_batch(
            {
                "a": np.arange(16),
                "b": list(np.arange(16)),
                "c": [{"a": 2}] * 16
            },
        )

        def func(x):
            out = (x["a"] + np.array(x["b"])) * 2
            return out

        result = datapane.map(func, batch_size=4, batched=True)
        self.assertTrue(isinstance(result, NumpyArrayColumn))
        self.assertEqual(len(result), 16)
        self.assertTrue((result == np.arange(16) * 4).all())
    
    def test_mixed_batched_map_return_multiple(self):
        datapane = DataPane.from_batch(
            {
                "a": np.arange(16),
                "b": list(np.arange(16)),
                "c": [{"a": 2}] * 16
            },
        )

        def func(x):
            out = {
                "x": (x["a"] + np.array(x["b"])) * 2,
                "y": np.array([x["c"][i]["a"] for i in range(len(x["c"]))]), 
            }
            return out

        result = datapane.map(func, batch_size=4, batched=True)
        self.assertTrue(isinstance(result, DataPane))
        self.assertEqual(len(result["x"]), 16)
        self.assertEqual(len(result["y"]), 16)
        self.assertTrue((result["x"] == np.arange(16) * 4).all())
        self.assertTrue((result["y"] == np.ones(16) * 2).all())

