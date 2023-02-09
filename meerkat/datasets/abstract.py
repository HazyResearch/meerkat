import json
import os
from abc import ABC, abstractmethod
from typing import List

import meerkat as mk

from ..config import DATASETS_ENV_VARIABLE
from .info import DatasetInfo
from .utils import download_url


class DatasetBuilder(ABC):
    REVISIONS: List[str]

    info: DatasetInfo = None

    def __init__(
        self,
        dataset_dir: str = None,
        version: str = None,
        download_mode: str = "reuse",
        **kwargs,
    ):
        self.name = self.__class__.__name__
        self.version = self.VERSIONS[0] if version is None else version
        self.download_mode = download_mode
        self.kwargs = kwargs

        if dataset_dir is None:
            self.dataset_dir = self._get_dataset_dir(self.name, self.version)
            self.var_dataset_dir = os.path.join(
                f"${DATASETS_ENV_VARIABLE}", self.name, self.version
            )
        else:
            self.dataset_dir = dataset_dir
            self.var_dataset_dir = dataset_dir

    def download_url(self, url: str):
        return download_url(url, self.dataset_dir, force=self.download_mode == "force")

    def dump_download_meta(self):
        data = {
            "name": self.name,
            "version": self.version,
            "dataset_dir": self.dataset_dir,
        }
        json.dump(
            data,
            open(os.path.join(self.dataset_dir, "meerkat_download.json"), "w"),
        )

    def __call__(self):
        if self.download_mode in ["force", "extract"] or (
            self.download_mode == "reuse" and not self.is_downloaded()
        ):
            self.download()
            self.dump_download_meta()

        if not self.is_downloaded():
            raise ValueError(
                f"Dataset {self.name} is not downloaded to {self.dataset_dir}."
            )

        return self.build()

    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()

    def is_downloaded(self) -> bool:
        """This is a very weak check for the existence of the dataset.

        Subclasses should ideally implement more thorough checks.
        """
        return os.path.exists(self.dataset_dir)

    @staticmethod
    def _get_dataset_dir(name: str, version: str) -> str:
        return os.path.join(mk.config.datasets.root_dir, name, version)
