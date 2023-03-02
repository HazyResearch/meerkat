from __future__ import annotations

import functools
import io
import logging
import os
import urllib.request
import warnings
from ctypes import Union
from pathlib import Path
from string import Template
from typing import IO, Any, Callable, Sequence
from urllib.parse import urlparse

import dill
import yaml
from PIL import Image

import meerkat.tools.docs as docs
from meerkat.block.deferred_block import DeferredOp
from meerkat.columns.abstract import Column
from meerkat.columns.deferred.base import DeferredCell, DeferredColumn
from meerkat.columns.scalar import ScalarColumn
from meerkat.interactive.formatter import (
    CodeFormatterGroup,
    HTMLFormatterGroup,
    PDFFormatterGroup,
    TextFormatterGroup,
)
from meerkat.interactive.formatter.base import FormatterGroup
from meerkat.interactive.formatter.image import DeferredImageFormatterGroup

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

    def _get_state(self):
        return {
            "loader": self.loader,
            "base_dir": self.base_dir,
            "downloader": self.downloader,
            "fallback_downloader": self.fallback_downloader,
            "cache_dir": self.cache_dir,
        }

    def _set_state(self, state):
        self.__dict__.update(state)

    def __setstate__(self, state):
        """Set state used by Pickle."""
        if "downloader" not in state:
            state["downloader"] = None
        if "fallback_downloader" not in state:
            state["fallback_downloader"] = None
        self._set_state(state)

    @staticmethod
    def to_yaml(dumper: yaml.Dumper, data: FileLoader):
        """This function is called by the YAML dumper to convert an
        :class:`Formatter` object into a YAML node.

        It should not be called directly.
        """
        data = {
            "class": type(data),
            "state": data._get_state(),
        }
        return dumper.represent_mapping("!FileLoader", data)

    @staticmethod
    def from_yaml(loader, node):
        """This function is called by the YAML loader to convert a YAML node
        into an :class:`Formatter` object.

        It should not be called directly.
        """
        data = loader.construct_mapping(node, deep=True)
        formatter = data["class"].__new__(data["class"])
        formatter._set_state(data["state"])
        return formatter


yaml.add_multi_representer(FileLoader, FileLoader.to_yaml)
yaml.add_constructor("!FileLoader", FileLoader.from_yaml)


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


@docs.doc(source=FILE_SHARED_DOCS)
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
        type: str = None,
        loader: callable = None,
        downloader: Union[callable | str] = None,
        base_dir: str = None,
        cache_dir: str = None,
        formatters: FormatterGroup = None,
        *args,
        **kwargs,
    ):

        if not isinstance(data, ScalarColumn):
            data = ScalarColumn(data)

        if type is None and (loader is None or formatters is None):
            # infer the type from the file extension
            type = _infer_file_type(data)

        if type not in FILE_TYPES:
            raise ValueError(f"Invalid file type {type}.")

        if type is not None and loader is None:
            loader = FILE_TYPES[type]["loader"]

        if type is not None and formatters is None:
            formatters = FILE_TYPES[type]["formatters"]()
            if FILE_TYPES[type].get("defer", True):
                formatters = formatters.defer()

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

        super(FileColumn, self).__init__(data, formatters=formatters, *args, **kwargs)

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
        if isinstance(path, io.BytesIO):
            return path.read().decode("utf-8")

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


def _infer_file_type(filepaths: ScalarColumn):
    """Infer the type of a file from its extension.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The type of the file.
    """

    NUM_SAMPLES = 100
    filepaths = filepaths[:NUM_SAMPLES]
    # extract the extension, taking into account that it may not exist
    # FIXME: make this work for URLs with `.com/...`
    ext = filepaths.str.extract(r"(?P<ext>\.[^\.]+)$")["ext"].str.lower()

    # if the extension is not present, then we assume it is a text file
    for type, info in FILE_TYPES.items():
        if ext.isin(info["exts"]).any():
            return type
    return "text"


def load_image(f: Union[str, io.BytesIO, Path]):
    img = Image.open(f)
    return img.convert("RGB")


def load_bytes(path: Union[str, io.BytesIO]):
    if isinstance(path, io.BytesIO):
        return path.read()

    with open(path, "rb") as f:
        return f.read()


def load_text(path: Union[str, io.BytesIO]):
    if isinstance(path, io.BytesIO):
        return path.read().decode("utf-8")

    with open(path, "r") as f:
        return f.read()


FILE_TYPES = {
    "image": {
        "loader": load_image,
        "formatters": DeferredImageFormatterGroup,
        "exts": [".jpg", ".jpeg", ".png", ".heic", ".JPEG"],
        "defer": False,
    },
    "pdf": {
        "loader": load_bytes,
        "formatters": PDFFormatterGroup,
        "exts": [".pdf"],
    },
    "html": {
        "loader": load_text,
        "formatters": HTMLFormatterGroup,
        "exts": [".html", ".htm"],
    },
    "text": {
        "loader": load_text,
        "formatters": TextFormatterGroup,
        "exts": [".txt"],
    },
    "code": {
        "loader": load_text,
        "formatters": CodeFormatterGroup,
        "exts": [".py", ".js", ".css", ".json", ".java", ".cpp", ".c", ".h", ".hpp"],
    },
}


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
