from __future__ import annotations

import logging
from typing import Callable

from meerkat.columns.file_column import FileColumn
from meerkat.display import image_file_formatter
from meerkat.tools.lazy_loader import LazyLoader

folder = LazyLoader("torchvision.datasets.folder")

logger = logging.getLogger(__name__)


class ImageColumn(FileColumn):
    """A column where each cell represents an image stored on disk. The
    underlying data is a `PandasSeriesColumn` of strings, where each string is
    the path to an image. The column materializes the images into memory when
    indexed. If the column is lazy indexed with the ``lz`` indexer, the images
    are not materialized and an ``ImageCell`` or an ``ImageColumn`` is returned
    instead.

    Args:
        data (Sequence[str]): A list of filepaths to images.
        transform (callable): A function that transforms the image (e.g.
            ``torchvision.transforms.functional.center_crop``).

            .. warning::
                In order for the column to be serializable, the transform function must
                be pickleable.


        loader (callable): A callable with signature ``def loader(filepath: str) ->
            PIL.Image:``. Defaults to ``torchvision.datasets.folder.default_loader``.

            .. warning::
                In order for the column to be serializable with ``write()``, the loader
                function must be pickleable.

        base_dir (str): A base directory that the paths in ``data`` are relative to. If
            ``None``, the paths are assumed to be absolute.
    """

    @staticmethod
    def _get_default_formatter() -> Callable:
        return image_file_formatter

    @classmethod
    def default_loader(cls, *args, **kwargs):
        return folder.default_loader(*args, **kwargs)
