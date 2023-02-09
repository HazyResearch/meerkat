import os
import subprocess

import PIL

import meerkat as mk
from meerkat.columns.deferred.image import load_image

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets

REPO = "https://github.com/NationalGalleryOfArt/opendata.git"


def _write_empty_image(dst):
    img = PIL.Image.new("RGB", (32, 32), color="black")
    img.save(dst, format="JPEG")


@datasets.register()
class ngoa(DatasetBuilder):
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
        base_dir = os.path.join(self.dataset_dir, "data")
        db = {}
        db["objects"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "objects.csv"), low_memory=False
        )
        db["published_images"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "published_images.csv"),
        )
        db["published_images"]["image"] = mk.ImageColumn.from_filepaths(
            db["published_images"]["iiifthumburl"],
            loader=mk.FileLoader(
                downloader="url",
                loader=load_image,
                # replace images for which the download fails with a black image
                fallback_downloader=_write_empty_image,
                cache_dir=os.path.join(base_dir, "iiifthumburl"),
            ),
        )

        db["published_images"]["image_224"] = mk.ImageColumn.from_filepaths(
            db["published_images"]["iiifurl"].apply(
                lambda x: f"{x}/full/!224,224/0/default.jpg"
            ),
            loader=mk.FileLoader(
                downloader="url",
                loader=load_image,
                cache_dir=os.path.join(base_dir, "iiifimage"),
                # replace images for which the download fails with a black image
                fallback_downloader=_write_empty_image,
            ),
        )
        db["objects_constituents"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "objects_constituents.csv"),
        )
        db["constituents"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "constituents.csv"),
        )
        db["constituents_text_entries"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "constituents_text_entries.csv"),
        )
        db["locations"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "locations.csv"),
        )
        db["objects_text_entries"] = mk.DataFrame.from_csv(
            os.path.join(base_dir, "objects_text_entries.csv"),
        )
        return db

    def download(self):
        # clone the repo using subprocess
        subprocess.call(["git", "clone", REPO, self.dataset_dir])
