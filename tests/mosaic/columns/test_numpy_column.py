"""Unittests for Datasets."""
from unittest import TestCase
import pickle

import numpy as np

from robustnessgym.mosaic.datapane import DataPane
from robustnessgym.mosaic import NumpyArrayColumn


class TestNumpyColumn(TestCase):
    def setUp(self):
        # Arrange
        pass

    def test_from_array(self):
        # Build a dataset from a batch
        array = np.random.rand(10, 3, 3)
        col = NumpyArrayColumn.from_array(
            array
        )

        self.assertTrue((col == array).all())
        self.assertEqual(len(col), 10)
    
    def test_mixed_batched_map_return_single(self):
        array = np.random.rand(16, 3)
        col = NumpyArrayColumn.from_array(array)

        def func(x):
            out = x.mean(axis=-1)
            return out

        result = col.map(func, batch_size=4, batched=True)
        self.assertTrue(isinstance(result, NumpyArrayColumn))
        self.assertEqual(len(result), 16)
        self.assertTrue((result == array.mean(axis=-1)).all())
    
    def test_mixed_batched_map_return_multiple(self):
        array = np.random.rand(16, 3)
        col = NumpyArrayColumn.from_array(array)

        def func(x):
            return {
                "mean": x.mean(axis=-1),
                "std": x.std(axis=-1)
            }

        result = col.map(func, batch_size=4, batched=True)
        self.assertTrue(isinstance(result, DataPane))
        self.assertTrue(isinstance(result["std"], NumpyArrayColumn))
        self.assertTrue(isinstance(result["mean"], NumpyArrayColumn))
        self.assertEqual(len(result), 16)
        self.assertTrue((result["mean"] == array.mean(axis=-1)).all())
        self.assertTrue((result["std"] == array.std(axis=-1)).all())

    def test_pickle(self):
        array = np.random.rand(10, 3, 3)
        col = NumpyArrayColumn.from_array(array)
        buf = pickle.dumps(col)
        new_col = pickle.loads(buf)

        self.assertTrue(isinstance(new_col, NumpyArrayColumn))
        self.assertTrue((col == new_col).all())