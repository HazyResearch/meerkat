import json
import os
import subprocess

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

REPO = "https://github.com/NationalGalleryOfArt/opendata.git"


@datasets.register()
class ngoa(DatasetBuilder):
    from meerkat.columns.file_column import Downloader

    VERSIONS = ["main"]

    info = DatasetInfo(
        name="ngoa",
        full_name="National Gallery of Art Open Data",
        # flake8: noqa
        description="The dataset provides data records relating to the 130,000+ artworks in our collection and the artists who created them. You can download the dataset free of charge without seeking authorization from the National Gallery of Art.",
        homepage="https://github.com/NationalGalleryOfArt/opendata",
        tags=["art"],
        citation=None,
    )

    def build(self):
        from meerkat.columns.file_column import Downloader

        base_dir = os.path.join(self.dataset_dir, "data")
        db = {}
        db["objects"] = mk.DataPanel.from_csv(
            os.path.join(base_dir, "objects.csv"),
        )
        db["published_images"] = mk.DataPanel.from_csv(
            os.path.join(base_dir, "published_images.csv"),
        )
        db["published_images"]["image"] = mk.ImageColumn.from_filepaths(
            db["published_images"]["iiifthumburl"],
            loader=Downloader(cache_dir=os.path.join(base_dir, "iiifthumburl")),
        )
        return db

    def download(self):

        # clone the repo using subprocess
        subprocess.call(["git", "clone", REPO, self.dataset_dir])
