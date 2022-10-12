from __future__ import annotations
from ctypes import Union
import io
import functools

import logging
import os
import urllib.request
import warnings
from string import Template
from typing import BinaryIO, BinaryIO, Callable, Sequence
from urllib.error import HTTPError
from urllib.parse import urlparse

import dill
import yaml

from meerkat.block.lambda_block import LambdaCellOp, LambdaOp
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.lambda_column import LambdaCell, LambdaColumn
from meerkat.columns.pandas_column import PandasSeriesColumn
from meerkat.tools.lazy_loader import LazyLoader

folder = LazyLoader("torchvision.datasets.folder")

logger = logging.getLogger(__name__)


class FileLoader:
    def __init__(
        self,
        transform: callable = None,
        loader: callable = None,
        downloader: Union[str, Callable] = None,
        cache_dir: str = None,
        base_dir: str = None,
    ):
        """
        Args:
            downloader (callable): a callable that accepts two positional arguments -
                a URI and a destination (which could be either a string or file object)
        """
        self.transform = transform
        self.loader = loader
        self.base_dir = base_dir
        if downloader == "url":
            self.downloader = download_url
        elif downloader == "gcs":
            self.downloader = download_gcs
        else:
            self.downloader = downloader
        self.cache_dir = cache_dir

    def __call__(self, filepath: str):
        """
        Args:
            filepath (str): If `downloader` is None, this is interpreted as a local
                filepath. Otherwise, it is interpreted as a URI from which the file can
                be downloaded.
        """
        # support including environment varaiables in the base_dir so that DataPanels
        # can be easily moved between machines
        if self.base_dir is not None:

            # need to convert Path objects to strings for Template to work
            base_dir = str(self.base_dir)

            try:
                # we don't use os.expanvars because it raises an error
                base_dir = Template(base_dir).substitute(os.environ)
            except KeyError:
                raise ValueError(
                    f'`base_dir="{base_dir}"` contains an undefined environment'
                    "variable."
                )
            filepath = os.path.join(base_dir, filepath)

        if self.downloader is not None:
            parse = urlparse(filepath)

            if self.cache_dir is not None:
                # need to convert Path objects to strings for Template to work
                cache_dir = str(self.cache_dir)

                try:
                    # we don't use os.expanvars because it raises an error
                    cache_dir = Template(cache_dir).substitute(os.environ)
                except KeyError:
                    raise ValueError(
                        f'`cache_dir="{cache_dir}"` contains an undefined environment'
                        "variable."
                    )
                dst = os.path.join(cache_dir, parse.netloc + parse.path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            else:
                # if there's no cache_dir, we just download to a temporary directory
                dst = io.BytesIO()
            
            if isinstance(dst, io.BytesIO) or not os.path.exists(dst):
                self.downloader(filepath, dst)
            filepath = dst

        data = self.loader(filepath)

        if self.transform is not None:
            data = self.transform(data)
        return data

    def __eq__(self, other: FileLoader) -> bool:
        return (
            (other.__class__ == self.__class__)
            and (self.loader == other.loader)
            and (self.transform == other.transform)
            and (self.base_dir == other.base_dir)
        )

    def __hash__(self) -> int:
        # needs to be hasable for block signature
        return hash((self.loader, self.transform, self.base_dir))
    
    def __setstate__(self, state):
        # need to add downloader if it is missing from state, for backwards compatibility
        if "downloader" not in state:
            state["downloader"] = None
        self.__dict__.update(state)


class FileCell(LambdaCell):
    def from_filepath(
        self,
        transform: callable = None,
        loader: callable = None,
        path: str = None,
        base_dir: str = None,
    ):
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform

        self.base_dir = base_dir

        data = LambdaCellOp(
            fn=self.fn,
            args=[
                path,
            ],
            kwargs={},
            is_batched_fn=False,
            return_index=None,
        )

        super().__init__(data=data)

    @property
    def absolute_path(self):
        return (
            os.path.join(self.base_dir, self.data)
            if self.base_dir is not None
            else self.data
        )

    def __eq__(self, other):
        return (other.__class__ == self.__class__) and other.data.is_equal(self.data)


class FileColumn(LambdaColumn):
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
        loader: callable = None,
        transform: callable = None,
        base_dir: str = None,
        *args,
        **kwargs,
    ):
        if isinstance(loader, FileLoader):
            fn = loader
            if fn.loader is None:
                fn.loader = self.default_loader

        else:
            fn = FileLoader(
                transform=transform,
                loader=self.default_loader if loader is None else loader,
                base_dir=base_dir,
            )

        if not isinstance(data, PandasSeriesColumn):
            data = PandasSeriesColumn(data)

        data = LambdaOp(
            args=[data],
            kwargs={},
            batch_size=1,
            fn=fn,
            is_batched_fn=False,
        )

        super(FileColumn, self).__init__(data, *args, **kwargs)

    @property
    def loader(self):
        return self.data.fn.loader

    @loader.setter
    def loader(self, loader: callable):
        self.data.fn.loader = loader

    @property
    def transform(self):
        return self.data.fn.transform

    @transform.setter
    def transform(self, transform: callable):
        self.data.fn.transform = transform

    @property
    def base_dir(self):
        return self.data.fn.base_dir

    @base_dir.setter
    def base_dir(self, base_dir: str):
        self.data.fn.base_dir = base_dir

    def _create_cell(self, data: object) -> LambdaCell:
        return FileCell(data=data)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str],
        loader: callable = None,
        transform: callable = None,
        base_dir: str = None,
    ):
        return cls(
            data=filepaths,
            loader=loader,
            transform=transform,
            base_dir=base_dir,
        )

    @classmethod
    def default_loader(cls, *args, **kwargs):
        return folder.default_loader(*args, **kwargs)

    @staticmethod
    def _read_data(path: str):
        try:
            return LambdaOp.read(path=os.path.join(path, "data"))
        except KeyError:
            # TODO(Sabri): Remove this in a future version, once we no longer need to
            # support old DataPanels.
            warnings.warn(
                "Reading a LambdaColumn stored in a format that will not be"
                " supported in the future. Please re-write the column to the new"
                " format.",
                category=FutureWarning,
            )
            meta = yaml.load(
                open(os.path.join(path, "data", "meta.yaml")),
                Loader=yaml.FullLoader,
            )
            if issubclass(meta["dtype"], AbstractColumn):
                col = AbstractColumn.read(os.path.join(path, "data"))
            else:
                raise ValueError(
                    "Support for LambdaColumns based on a DataPanel is deprecated."
                )

            state = dill.load(open(os.path.join(path, "state.dill"), "rb"))

            fn = FileLoader(
                transform=state["transform"],
                loader=state["loader"],
                base_dir=state["base_dir"],
            )

            return LambdaOp(
                args=[col],
                kwargs={},
                fn=fn,
                is_batched_fn=False,
                batch_size=1,
            )

    def is_equal(self, other: AbstractColumn) -> bool:
        return (other.__class__ == self.__class__) and self.data.is_equal(other.data)


def download_url(url: str, dst: Union[str, io.BytesIO]):
    if isinstance(dst, str):
        return urllib.request.urlretrieve(url=url, filename=dst)
    else:
        import requests

        response = requests.get(url)
        dst.write(response.content)
        dst.seek(0)
        return dst


@functools.lru_cache
def _get_gcs_bucket(bucket_name: str, project: str = None):
    """Get a GCS bucket."""
    from google.cloud import storage

    client = storage.Client(project=project)
    return client.bucket(bucket_name)


def download_gcs(uri: str, dst: Union[str, io.BytesIO]):
    """Download a file from GCS."""
    from google.cloud import exceptions
    bucket, blob_name = urlparse(uri).netloc, urlparse(uri).path.lstrip("/")
    bucket = _get_gcs_bucket(bucket_name=uri.split("/")[2])

    try:
        if isinstance(dst, io.BytesIO):
            dst.write(bucket.blob(str(blob_name)).download_as_bytes())
            dst.seek(0)
            return dst
        else:
            bucket.blob(str(blob_name)).download_to_filename(dst)
            return dst
    except exceptions.NotFound:
        os.remove(dst)

        raise FileNotFoundError(uri)

