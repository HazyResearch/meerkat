from abc import ABC, abstractmethod


class DatasetBuilder(ABC):
    def __init__(
        name: str,
        dataset_dir: str = None,
        revision: str = None,
        download_mode: str = "reuse",
        **kwargs,
    ):
        self.name = name
        self.dataset_dir = dataset_dir
        self.revision = revision
        self.download_mode = download_mode
        self.kwargs = kwargs

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
