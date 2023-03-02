from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Sequence, Union

from PIL import Image

import meerkat.tools.docs as docs
from meerkat.columns.deferred.file import FILE_SHARED_DOCS, FileColumn
from meerkat.interactive.formatter import ImageFormatterGroup
from meerkat.interactive.formatter.base import deferred_formatter_group

logger = logging.getLogger(__name__)


def load_image(f: Union[str, BytesIO, Path]):
    img = Image.open(f)
    return img.convert("RGB")


@docs.doc(source=FILE_SHARED_DOCS)
def image(
    filepaths: Sequence[str],
    base_dir: Optional[str] = None,
    downloader: Union[callable, str] = None,
    loader: callable = load_image,
    cache_dir: str = None,
):
    """Create a :class:`FileColumn` where each cell represents an image stored
    on disk. The underlying data is a :class:`ScalarColumn` of strings, where
    each string is the path to an image.

    Args:
        filepaths (Sequence[str]): A list of filepaths to images.
        ${loader}
        ${base_dir}
        ${downloader}
        ${fallback_downloader}
        ${cache_dir}
    """
    return FileColumn(
        filepaths,
        type="image",
        base_dir=base_dir,
        loader=loader,
        downloader=downloader,
        cache_dir=cache_dir,
        formatters=deferred_formatter_group(ImageFormatterGroup()),
    )


class ImageColumn(FileColumn):
    """DEPRECATED A column where each cell represents an image stored on disk.
    The underlying data is a `PandasSeriesColumn` of strings, where each string
    is the path to an image. The column materializes the images into memory
    when indexed. If the column is lazy indexed with the ``lz`` indexer, the
    images are not materialized and an ``ImageCell`` or an ``ImageColumn`` is
    returned instead.

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

    @classmethod
    def default_loader(cls, *args, **kwargs):
        return load_image(*args, **kwargs)
