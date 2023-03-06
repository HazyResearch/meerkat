import numpy as np
import pytest
from PIL import Image

from meerkat.interactive.formatter.image import ImageFormatter


@pytest.mark.parametrize("skip_copy", [True, False])
def test_image_formatter_encode_skip_copy(skip_copy: bool):
    """Test image formatter on object columns."""
    formatter = ImageFormatter(max_size=(20, 20))

    image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    formatter.encode(image, skip_copy=skip_copy)
    if skip_copy:
        assert image.size == (20, 20)
    else:
        assert image.size == (100, 100)
