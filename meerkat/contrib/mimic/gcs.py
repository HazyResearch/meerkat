from __future__ import annotations

import io
import os
from typing import Sequence

from google.cloud import storage

from meerkat import ImageColumn
from meerkat.columns.pandas_column import PandasSeriesColumn


class GCSImageColumn(ImageColumn):
    def __init__(
        self,
        blob_names: PandasSeriesColumn = None,
        bucket_name: str = None,
        project: str = None,
        transform: callable = None,
        loader: callable = None,
        writer: callable = None,
        local_dir: str = None,
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

    def fn(self, blob_name: str):
        if self.local_dir is not None and os.path.exists(
            os.path.join(self.local_dir, str(blob_name))
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
        if not isinstance(blob_names, PandasSeriesColumn):
            blob_names = PandasSeriesColumn(blob_names)
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
        return super()._state_keys() | {"bucket_name", "project", "local_dir", "writer"}

    classmethod

    def _clone_keys(cls) -> set:
        # need to avoid reaccessing bucket on clone, too slow
        return {"bucket"}

    def _set_state(self, state: dict = None):
        if state is not None:
            self.__dict__.update(state)
        if state is None or "bucket" not in state:
            storage_client = storage.Client(project=self.project)
            self.bucket = storage_client.bucket(
                self.bucket_name, user_project=self.project
            )
