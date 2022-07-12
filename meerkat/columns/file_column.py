from __future__ import annotations

import logging
import os
import urllib.request
from typing import Collection, Sequence
from urllib.error import HTTPError
from urllib.parse import urlparse

from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.lambda_column import LambdaCell, LambdaColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.tools.lazy_loader import LazyLoader

folder = LazyLoader("torchvision.datasets.folder")

logger = logging.getLogger(__name__)


class FileLoaderMixin:
    def fn(self, filepath: str):
        absolute_path = (
            os.path.join(self.base_dir, filepath)
            if self.base_dir is not None
            else filepath
        )
        image = self.loader(absolute_path)

        if self.transform is not None:
            image = self.transform(image)
        return image


class FileCell(FileLoaderMixin, LambdaCell):
    def __init__(
        self,
        transform: callable = None,
        loader: callable = None,
        data: str = None,
        base_dir: str = None,
    ):
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform
        self._data = data
        self.base_dir = base_dir

    @property
    def absolute_path(self):
        return (
            os.path.join(self.base_dir, self.data)
            if self.base_dir is not None
            else self.data
        )

    def __eq__(self, other):
        return (
            (other.__class__ == self.__class__)
            and (self.data == other.data)
            and (self.transform == other.transform)
            and (self.loader == other.loader)
        )

    def __repr__(self):
        transform = getattr(self.transform, "__qualname__", repr(self.transform))
        dirs = self.data.split("/")
        short_path = ("" if len(dirs) <= 2 else ".../") + "/".join(dirs[-2:])
        return f"{self.__class__.__name__}.({short_path}, transform={transform})"


class FileColumn(FileLoaderMixin, LambdaColumn):
    """A column where each cell represents an file stored on disk or the web.
    The underlying data is a `PandasSeriesColumn` of strings, where each string
    is the path to a file. The column materializes the files into memory when
    indexed. If the column is lazy indexed with the ``lz`` indexer, the files
    are not materialized and a ``FileCell`` or a ``FileColumn`` is returned
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

    def __init__(
        self,
        data: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        base_dir: str = None,
        *args,
        **kwargs,
    ):

        if not isinstance(data, PandasSeriesColumn):
            data = PandasSeriesColumn(data)
        super(FileColumn, self).__init__(data, *args, **kwargs)
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform
        self.base_dir = base_dir

    def _create_cell(self, data: object) -> FileCell:
        return FileCell(
            data=data,
            loader=self.loader,
            transform=self.transform,
            base_dir=self.base_dir,
        )

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str],
        loader: callable = None,
        transform: callable = None,
        base_dir: str = None,
        *args,
        **kwargs,
    ):
        return cls(
            data=filepaths,
            loader=loader,
            transform=transform,
            base_dir=base_dir,
            *args,
            **kwargs,
        )

    @classmethod
    def default_loader(cls, *args, **kwargs):
        return folder.default_loader(*args, **kwargs)

    @classmethod
    def _state_keys(cls) -> Collection:
        return (super()._state_keys() | {"transform", "loader", "base_dir"}) - {"fn"}

    def _set_state(self, state: dict):
        state["base_dir"] = state.get("base_dir", None)  # backwards compatibility
        super()._set_state(state)

    def is_equal(self, other: AbstractColumn) -> bool:
        return (
            (other.__class__ == self.__class__)
            and (self.loader == other.loader)
            and (self.transform == other.transform)
            and self.data.is_equal(other.data)
        )


def download_image(url: str, cache_dir: str):
    parse = urlparse(url)
    local_path = os.path.join(cache_dir, parse.netloc + parse.path)

    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local_path)
        except (HTTPError, ConnectionResetError):
            logger.warning(f"Could not download {url}. Skipping.")
            return None

    return folder.default_loader(local_path)


class Downloader:
    def __init__(
        self,
        cache_dir: str,
        downloader: callable = None,
    ):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if downloader is None:
            self.downloader = download_image

    def __call__(self, url: str):
        return self.downloader(url, self.cache_dir)
