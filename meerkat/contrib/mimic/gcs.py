from __future__ import annotations

import io
import os
import subprocess
from typing import Sequence

import google.auth
import numpy as np
from google.cloud import bigquery, bigquery_storage, storage

from meerkat import ImageColumn


class GCSImageColumn(ImageColumn):
    def __init__(
        self,
        bucket_name: str = None,
        project: str = None,
        filepaths: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        writer: callable = None,
        local_dir: str = None,
        *args,
        **kwargs,
    ):
        super(GCSImageColumn, self).__init__(
            filepaths, transform, loader, *args, **kwargs
        )
        self.project = project
        self.bucket_name = bucket_name
        storage_client = storage.Client(project=project)

        self.bucket = storage_client.bucket(bucket_name, user_project=project)
        self.loader = (lambda x: x) if loader is None else loader
        self.writer = writer
        self.local_dir = local_dir

    def fn(self, filepath: str):

        blob_name = self._data.iloc[index]
        if self.local_dir is not None and os.path.exists(
            os.path.join(self.local_dir, str(blob_name))
        ):
            # fetch locally if it's been cached locally
            cell = mk.ImagePath(
                os.path.join(self.local_dir, str(blob_name)),
                loader=self.loader,
                transform=self.transform,
            )
            return cell.get()

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

    def _get_batch(self, indices: np.ndarray, materialize: bool = True):
        if materialize:
            # if materializing, return a batch (by default, a list of objects returned
            # by `.get`, otherwise the batch format specified by `self.collate`)
            return self.collate(
                [self._get_cell(idx, materialize=materialize) for idx in indices]
            )

        else:
            data = self._data.iloc[indices].reset_index(drop=True)
            return self.__class__(
                filepaths=data,
                loader=self.loader,
                transform=self.transform,
                bucket_name=self.bucket_name,
                project=self.project,
                local_dir=self.local_dir,
            )

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return super()._state_keys() | {"bucket_name", "project", "local_dir"}

    @staticmethod
    def concat(columns: Sequence[GCSImageColumn]):
        loader, transform = (
            columns[0].loader,
            columns[0].transform,
        )

        return GCSImageColumn(
            filepaths=pd.concat([c.data for c in columns]),
            loader=loader,
            transform=transform,
            bucket_name=columns[0].bucket_name,
            project=columns[0].project,
            local_dir=columns[0].local_dir,
        )
