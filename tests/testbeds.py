"""A collection of simple testbeds to build test cases."""
import os
from copy import deepcopy

import numpy as np
from PIL import Image

from meerkat.columns.image_column import ImageCellColumn, ImageColumn
from meerkat.columns.list_column import ListColumn
from meerkat.datapanel import DataPanel
from meerkat.tools.identifier import Identifier


class MockTestBedv0:
    """Simple mock dataset with 6 examples."""

    def __init__(self):
        # Create a fake batch of data
        self.batch = {
            "text": [
                "The man is walking.",
                "The man is running.",
                "The woman is sprinting.",
                "The woman is resting.",
                "The hobbit is flying.",
                "The hobbit is swimming.",
            ],
            "label": [0, 0, 1, 1, 0, 0],
            "z": [1, 0, 1, 0, 1, 0],
            "fast": [False, True, True, False, False, False],
            "metadata": [
                {"source": "real"},
                {"source": "real"},
                {"source": "real"},
                {"source": "real"},
                {"source": "fictional"},
                {"source": "fictional"},
            ],
        }
        # Create a fake dataset
        self.dataset = DataPanel.from_batch(
            self.batch,
            identifier=Identifier(_name="MockDataPane", version="1.0"),
        )

        # Keep a copy of the original
        self.original_dataset = deepcopy(self.dataset)

        assert len(self.dataset) == 6

    def test_attributes(self):
        # Both datasets use the same cache files for backing
        print(self.dataset.cache_files)
        print(self.original_dataset.cache_files)
        print(self.dataset.identifier)

    def problems(self):
        # FIXME(karan): this shouldn't be happening: why is otherlabel disappearing here
        with self.assertRaises(AssertionError):
            # Create an additional integer column in the dataset
            dataset = self.testbed.dataset.map(lambda x: {"otherlabel": x["label"] + 1})
            dataset_0_0 = self.cachedop(dataset, columns=["label"])
            self.assertTrue("otherlabel" in dataset_0_0.column_names)


class MockTestBedv1:
    """Simple mock dataset with 4 examples containing pairs of sentences."""

    def __init__(self):
        # Create a fake dataset
        self.dataset = DataPanel.from_batch(
            {
                "text_a": [
                    "Before the actor slept, the senator ran.",
                    "The lawyer knew that the judges shouted.",
                    "If the actor slept, the judge saw the artist.",
                    "The lawyers resigned, or the artist slept.",
                ],
                "text_b": [
                    "The actor slept.",
                    "The judges shouted.",
                    "The actor slept.",
                    "The artist slept.",
                ],
                "label": [0, 0, 1, 1],
                "z": [1, 0, 1, 0],
                "fast": [False, True, True, False],
            },
            identifier=Identifier(_name="MockDataPane", version="2.0"),
        )

        # Keep a copy of the original
        self.original_dataset = deepcopy(self.dataset)

        assert len(self.dataset) == 4


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
            "b": ListColumn(np.arange(length)),
            "c": [{"a": 2}] * length,
        }

        if include_image_column:
            assert tmpdir is not None
            self.img_col = MockImageColumn(length=length, tmpdir=tmpdir)
            batch["img"] = self.img_col.col

        self.dp = DataPanel.from_batch(batch)

        self.visible_rows = [0, 4, 6, 11] if use_visible_rows else np.arange(length)
        if use_visible_rows:
            for column in self.dp.values():
                column.visible_rows = self.visible_rows

        self.visible_columns = (
            ["a", "b", "index"] if use_visible_columns else self.dp.columns
        )
        if use_visible_columns:
            self.dp.visible_columns = self.visible_columns


class MockColumn:
    def __init__(self, use_visible_rows: bool = False, col_type: type = ListColumn):
        self.array = np.arange(16)
        self.col = col_type(self.array)

        if use_visible_rows:
            self.visible_rows = [0, 4, 6, 11]
            self.col.visible_rows = self.visible_rows
        else:
            self.visible_rows = np.arange(16)


class MockImageColumn:
    def __init__(self, length: int, tmpdir: str, use_cell_column: bool = False):
        """[summary]

        Args:
            wrap_dataset (bool, optional): If `True`, create a
            `meerkat.DataPanel`
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

        if use_cell_column:
            self.col = ImageCellColumn.from_filepaths(self.image_paths)
        else:
            self.col = ImageColumn.from_filepaths(self.image_paths)
