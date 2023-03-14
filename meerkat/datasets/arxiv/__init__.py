import os
import pandas as pd

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import extract


@datasets.register()
class arxiv(DatasetBuilder):

    VERSIONS = ["full"]

    info = DatasetInfo(
        name="arxiv",
        full_name="arXiv Dataset",
        description=(
            "This dataset contains metadata for all 1.7 million+ arXiv articles "
            "from the period of 1991 to 2021 across various research fields."
        ),
        homepage="https://www.kaggle.com/Cornell-University/arxiv",
        tags=["arxiv", "metadata", "research", "papers"],
    )

    @property
    def data_dir(self):
        return os.path.join(self.dataset_dir, "arxiv-metadata-oai-snapshot")

    def build(self):
        df = mk.from_json(
            os.path.join(self.dataset_dir, "arxiv-metadata-oai-snapshot.json"),
            lines=True,
            backend="arrow",
        )
        df.set_primary_key("id", inplace=True)
        return df

    def download(self):

        self.download_kaggle_dataset("Cornell-University/arxiv", self.dataset_dir)
        extract(
            os.path.join(self.dataset_dir, "arxiv.zip"),
            self.dataset_dir,
        )

    @staticmethod
    def download_kaggle_dataset(dataset, dest_dir):
        import kaggle.api

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=dest_dir, unzip=False)
