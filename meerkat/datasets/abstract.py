from abc import ABC, abstractmethod
from typing import List
import os
from meerkat.config import DatasetsOptions

from .info import DatasetInfo

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
            self.dataset_dir = os.path.join(
                DatasetsOptions.root_datasets_dir, self.name, self.version
            )
        else:
            self.dataset_dir = dataset_dir

    def __call__(self):

        if self.download_mode == "force" or (
            self.download_mode == "reuse" and not self.is_downloaded
        ):
            self.download()

        if not self.is_downloaded():
            raise ValueError(f"Dataset {self.name} is not downloaded.")

        return self.build()

    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def download(self):
        raise NotImplementedError()

    @abstractmethod
    def is_downloaded(self) -> bool:
        raise NotImplementedError()
