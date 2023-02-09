from __future__ import annotations

import functools
import io
import logging
import os
import urllib.request
import warnings
from ctypes import Union
from string import Template
from typing import IO, Any, Callable, Sequence
from urllib.parse import urlparse

import dill
import yaml

import meerkat.tools.docs as docs
from meerkat.block.deferred_block import DeferredOp
from meerkat.columns.abstract import Column
from meerkat.columns.deferred.base import DeferredCell, DeferredColumn
from meerkat.columns.scalar import ScalarColumn

logger = logging.getLogger(__name__)


FILE_SHARED_DOCS = {
    "loader": docs.Arg(
        """
        loader (Union[str, Callable[[Union[str, IO]], Any]]): a callable that
            accepts a filepath or an I/O stream and returns data.
        """
    ),
    "cache_dir": docs.Arg(
        """
        cache_dir (str, optional): the directory on disk where downloaded
                files are to be cached. Defaults to None, in which case files will be
                re-downloaded on every access of the data. The ``cache_dir`` can also
                include environment variables (e.g. ``$DATA_DIR/images``) which will
                be expanded prior to loading. This is useful when sharing DataFrames
                between machines.
        """
    ),
    "base_dir": docs.Arg(
        """
        base_dir (str, optional): an absolute path to a directory containing the
            files. If provided, the ``filepath`` to be loaded will be joined with
            the ``base_dir``. As such, this argument should only be used if the
            loader will be applied to relative paths. T

            The ``base_dir`` can also
            include environment variables (e.g. ``$DATA_DIR/images``) which will
            be expanded prior to loading. This is useful when sharing DataFrames
            between machines.
        """
    ),
    "downloader": docs.Arg(
        """
        downloader (Union[str, callable], optional):  a callable that accepts at
            least two  positional arguments - a URI and a destination (which could
            be either a string or file object).

            Meerkat includes a small set of built-in downloaders ["url", "gcs"]
            which can be specified via string.
        """
    ),
    "fallback_downloader": docs.Arg(
        """
        fallback_downloader (callable, optional): a callable that will be run each
            time the the downloader fails (for any reason). This is useful, for
            example, if you expect some of the URIs in a dataset to be broken
            ``fallback_downloader`` could write an empty file in place of the
            original. If ``fallback_downloader`` is not supplied, the original
            exception is re-raised.
        """
    ),
}


class FileLoader:
    @docs.doc(source=FILE_SHARED_DOCS)
    def __init__(
        self,
        loader: Union[str, Callable[[Union[str, IO]], Any]],
        base_dir: str = None,
        downloader: Union[str, Callable] = None,
        fallback_downloader: Callable[[Union[str, IO]], None] = None,
        cache_dir: str = None,
    ):
        """A simple file loader with support for both local paths and remote
        URIs.

        .. warning::
            In order for the column to be serializable with ``write()``, the
            callables passed to the constructor must be pickleable.

        Args:
            ${loader}
            ${base_dir}
            ${downloader}
            ${fallback_downloader}
            ${cache_dir}
        """
        self.loader = loader
        self.base_dir = base_dir
        if downloader == "url":
            self.downloader = download_url
        elif downloader == "gcs":
            self.downloader = download_gcs
        else:
            self.downloader = downloader
        self.fallback_downloader = fallback_downloader
        self.cache_dir = cache_dir

    def __call__(self, filepath: str):
        """
        Args:
            filepath (str): If `downloader` is None, this is interpreted as a local
                filepath. Otherwise, it is interpreted as a URI from which the file can
                be downloaded.
        """
        # support including environment varaiables in the base_dir so that DataFrames
        # can be easily moved between machines
        if self.base_dir is not None:

            # need to convert Path objects to strings for Template to work
            base_dir = str(self.base_dir)
            base_dir = os.path.expanduser(base_dir)

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
                cache_dir = os.path.expanduser(cache_dir)

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
                try:
                    self.downloader(filepath, dst)
                except Exception as e:
                    if self.fallback_downloader is not None:
                        # if user passes fallback_downloader, then on any
                        # failed download, we write the default data to the
                        # destination and continue
                        warnings.warn(
                            f"Failed to download {filepath} with error {e}. Falling "
                            "back to default data."
                        )
                        self.fallback_downloader(dst)
                    else:
                        raise e

            filepath = dst

        data = self.loader(filepath)

        return data

    def __eq__(self, other: FileLoader) -> bool:
        return (
            (other.__class__ == self.__class__)
            and (self.loader == other.loader)
            and (self.base_dir == other.base_dir)
        )

    def __hash__(self) -> int:
        # needs to be hasable for block signature
        return hash((self.loader, self.base_dir))

    def __setstate__(self, state):
        # need to add downloader if it is missing from state,
        # for backwards compatibility
        if "downloader" not in state:
            state["downloader"] = None
        if "fallback_downloader" not in state:
            state["fallback_downloader"] = None
        self.__dict__.update(state)


class FileCell(DeferredCell):
    @property
    def base_dir(self):
        return self.data.fn.base_dir

    @property
    def absolute_path(self):
        return (
            os.path.join(self.base_dir, self.data.args[0])
            if self.base_dir is not None
            else self.data.args[0]
        )

    def __eq__(self, other):
        return (other.__class__ == self.__class__) and other.data.is_equal(self.data)


class FileColumn(DeferredColumn):
    """A column where each cell represents an file stored on disk or the web.
    The underlying data is a `PandasSeriesColumn` of strings, where each string
    is the path to a file. The column materializes the files into memory when
    indexed. If the column is lazy indexed with the ``lz`` indexer, the files
    are not materialized and a ``FileCell`` or a ``FileColumn`` is returned
    instead.

    Args:
        data (Sequence[str]): A list of filepaths to images.

        ${loader}
        ${base_dir}
        ${downloader}
        ${cache_dir}
    """

    def __init__(
        self,
        data: Sequence[str] = None,
        loader: callable = None,
        downloader: Union[callable | str] = None,
        base_dir: str = None,
        cache_dir: str = None,
        *args,
        **kwargs,
    ):

        if not isinstance(data, ScalarColumn):
            data = ScalarColumn(data)

        # if base_dir is not provided and all paths are absolute, then
        # we can infer the base_dir
        if base_dir is None and data.str.startswith("/").all():
            base_dir = os.path.commonpath(data)
            data = data.str.replace(base_dir + "/", "")

        # if downloader is not provided then we can try to infer from the filepaths
        if downloader is None:
            if data.str.startswith("http").all():
                downloader = download_url
            elif data.str.startswith("gs://").all():
                downloader = download_gcs

        if isinstance(loader, FileLoader):
            if base_dir is not None or downloader is not None:
                raise ValueError(
                    "Cannot pass, `base_dir`, `downloader`, when loader is "
                    "a `FileLoader`."
                )

            fn = loader
            if fn.loader is None:
                fn.loader = self.default_loader
        else:
            fn = FileLoader(
                loader=self.default_loader if loader is None else loader,
                base_dir=base_dir,
                downloader=downloader,
                cache_dir=cache_dir,
            )

        data = DeferredOp(
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
    def base_dir(self):
        return self.data.fn.base_dir

    @base_dir.setter
    def base_dir(self, base_dir: str):
        self.data.fn.base_dir = base_dir

    def _create_cell(self, data: object) -> DeferredCell:
        return FileCell(data=data)

    @classmethod
    def from_filepaths(
        cls,
        filepaths: Sequence[str],
        loader: callable = None,
        base_dir: str = None,
    ):
        return cls(
            data=filepaths,
            loader=loader,
            base_dir=base_dir,
        )

    @classmethod
    def from_urls(
        cls,
        urls: Sequence[str],
    ):
        from PIL import Image

        return cls(
            data=urls,
            loader=FileLoader(
                loader=lambda bytes_io: Image.open(bytes_io).convert("RGB"),
                downloader="url",
            ),
        )

    @classmethod
    def default_loader(cls, path, *args, **kwargs):
        with open(path, "r") as f:
            return f.read()

    @staticmethod
    def _read_data(path: str):
        try:
            return DeferredOp.read(path=os.path.join(path, "data"))
        except KeyError:
            # TODO(Sabri): Remove this in a future version, once we no longer need to
            # support old DataFrames.
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
            if issubclass(meta["dtype"], Column):
                col = Column.read(os.path.join(path, "data"))
            else:
                raise ValueError(
                    "Support for LambdaColumns based on a DataFrame is deprecated."
                )

            state = dill.load(open(os.path.join(path, "state.dill"), "rb"))

            fn = FileLoader(
                loader=state["loader"],
                base_dir=state["base_dir"],
            )

            return DeferredOp(
                args=[col],
                kwargs={},
                fn=fn,
                is_batched_fn=False,
                batch_size=1,
            )

    def is_equal(self, other: Column) -> bool:
        return (other.__class__ == self.__class__) and self.data.is_equal(other.data)


def download_url(url: str, dst: Union[str, io.BytesIO]):
    if isinstance(dst, str):
        return urllib.request.urlretrieve(url=url, filename=dst)
    else:
        import requests

        response = requests.get(url)
        data = response.content
        dst.write(data)
        dst.seek(0)
        return dst


@functools.lru_cache()
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
