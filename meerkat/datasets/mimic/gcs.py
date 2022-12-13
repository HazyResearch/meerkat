from __future__ import annotations

import io
import os
from typing import Sequence

from google.cloud import storage

from meerkat import ImageColumn
from meerkat.columns.deferred.base import DeferredCell, DeferredColumn
from meerkat.columns.pandas_column import ScalarColumn


class GCSImageCell(DeferredCell):
    def __init__(
        self,
        transform: callable = None,
        loader: callable = None,
        data: str = None,
    ):
        self.loader = self.default_loader if loader is None else loader
        self.transform = transform
        self._data = data

    def fn(self, filepath: str):
        image = self.loader(filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image

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
        return f"ImageCell({short_path}, transform={transform})"


class GCSImageColumn(ImageColumn):
    def __init__(
        self,
        blob_names: ScalarColumn = None,
        bucket_name: str = None,
        project: str = None,
        transform: callable = None,
        loader: callable = None,
        writer: callable = None,
        local_dir: str = None,
        _skip_cache: bool = False,
        *args,
        **kwargs,
    ):
        super(GCSImageColumn, self).__init__(
            blob_names, transform, loader, *args, **kwargs
        )
        self.project = project
        self.bucket_name = bucket_name
        self._set_state()
        storage_client = storage.Client(project=project)

        self.bucket = storage_client.bucket(bucket_name, user_project=project)
        self.loader = (lambda x: x) if loader is None else loader
        self.writer = writer
        self.local_dir = local_dir
        self._skip_cache = _skip_cache

    def _get_formatter(self) -> callable:
        # downloading the images from gcp for every visualization is probably not
        # what we want as it makes dataframe visualization very slow
        return None

    def _create_cell(self, data: object) -> DeferredCell:
        # don't want to create a lambda
        return DeferredColumn._create_cell(self, data)

    def fn(self, blob_name: str):
        if (
            self.local_dir is not None
            and os.path.exists(os.path.join(self.local_dir, str(blob_name)))
            and not self._skip_cache
        ):
            # fetch locally if it's been cached locally
            return super(GCSImageColumn, self).fn(
                os.path.join(self.local_dir, str(blob_name))
            )
        # otherwise pull from GCP
        out = self.loader(
            io.BytesIO(self.bucket.blob(str(blob_name)).download_as_bytes())
        )

        if self.writer is not None and self.local_dir is not None:
            # cache locally if writer and local dir are both provided
            path = os.path.join(self.local_dir, str(blob_name))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.writer(path, out)
        return out

    @classmethod
    def from_blob_names(
        cls,
        blob_names: Sequence[str],
        loader: callable = None,
        transform: callable = None,
        *args,
        **kwargs,
    ):
        if not isinstance(blob_names, ScalarColumn):
            blob_names = ScalarColumn(blob_names)
        return cls(
            blob_names=blob_names,
            loader=loader,
            transform=transform,
            *args,
            **kwargs,
        )

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return super()._state_keys() | {
            "bucket_name",
            "project",
            "local_dir",
            "writer",
            "_skip_cache",
        }

    @classmethod
    def _clone_keys(cls) -> set:
        # need to avoid reaccessing bucket on clone, too slow
        return {"bucket"}

    def _set_state(self, state: dict = None):
        if state is not None:
            state["base_dir"] = state.get("base_dir", None)  # backwards compatibility
            self.__dict__.update(state)

        if state is None or "bucket" not in state:
            storage_client = storage.Client(project=self.project)
            self.bucket = storage_client.bucket(
                self.bucket_name, user_project=self.project
            )
