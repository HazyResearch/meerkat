import time

import numpy as np
from PIL import Image

from meerkat.columns.object.base import ObjectColumn
from meerkat.interactive.formatter.image import ImageFormatterGroup


def test_formatters_image():
    """Test formatters when the object column is full of images."""
    images = [
        Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        for _ in range(10)
    ]

    # Automatically detect the formatters.
    col = ObjectColumn(images)
    assert isinstance(col.formatters, ImageFormatterGroup)

    # make sure the formatters do not modify the object in place.
    for key in col.formatters.keys():
        size = images[0].size
        col.formatters[key].encode(images[0])
        assert images[0].size == size


def test_to_numpy_speed():
    """Casting an object column to numpy should be fast (< 1 second)."""
    objs = [[0, 0, 0, 0] for _ in range(10000)]
    col = ObjectColumn(objs)

    start = time.perf_counter()
    col.to_numpy()
    time_elapsed = time.perf_counter() - start
    assert time_elapsed < 1.0

    start = time.perf_counter()
    np.array(col)
    time_elapsed = time.perf_counter() - start
    assert time_elapsed < 1.0
