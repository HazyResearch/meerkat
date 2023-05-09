import os
import re

import pandas as pd

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

_URL = "https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN{version}.zip"  # noqa: E501


@datasets.register()
class gtsdb(DatasetBuilder):
    """German Traffic Sign Detection Benchmark GTSDB."""

    VERSIONS = ["2013"]

    info = DatasetInfo(
        name="gtsdb",
        full_name="German Traffic Sign Detection Benchmark GTSDB",
        description=("Image data set to detect street signs."),
        homepage="https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html",  # noqa: E501
        tags=["image", "object recognition"],
        citation=None,
    )

    def build(self):
        """Get the processed dataframe hosted on huggingface."""
        folder = os.path.join(self.dataset_dir, f"FullIJCNN{self.version}")
        gt_ann = pd.read_csv(os.path.join(folder, "gt.txt"), sep=";", header=None)

        # Format categories
        readme = os.path.join(folder, "ReadMe.txt")
        with open(readme, "r") as f:
            lines = [
                x.strip() for x in f.readlines() if re.match("^[0-9]* = .*$", x.strip())
            ]

        categories = []
        for line in lines:
            category_id, category_full_name = line.split(" = ")
            category_id = int(category_id)
            category_full_name = category_full_name.strip()

            category_name, supercategory = category_full_name.rsplit(" ", 1)
            category_name = category_name.strip()
            supercategory = supercategory.strip().strip("(").strip(")")

            categories.append(
                {
                    "category_id": int(category_id),
                    "category": category_name,
                    "supercategory": supercategory,
                }
            )

        categories = pd.DataFrame(categories)

        # Format dataframe
        df = gt_ann.rename(
            {0: "filename", 1: "x1", 2: "y1", 3: "x2", 4: "y2", 5: "category_id"},
            axis=1,
        )
        df = df.merge(categories, on="category_id")

        # Split
        images_files = sorted([x for x in os.listdir(folder) if x.endswith(".ppm")])
        image_df = pd.DataFrame({"filename": images_files})
        image_df["split"] = "train"
        image_df.loc[600:, "split"] = "test"
        df = df.merge(image_df, on="filename")

        df = mk.DataFrame.from_pandas(df).drop("index")
        df["image"] = mk.files(df["filename"], base_dir=folder, type="image")
        df["image_crop"] = mk.defer(df, crop)
        return df

    def download(self):
        downloaded_path = download_url(
            _URL.format(version=self.version), self.dataset_dir
        )
        extract(downloaded_path, self.dataset_dir)

    def is_downloaded(self):
        return os.path.exists(self.dataset_dir) and os.path.exists(
            os.path.join(self.dataset_dir, f"FullIJCNN{self.version}")
        )


def crop(image, x1, y1, x2, y2):
    out = image.crop((x1, y1, x2, y2))
    return out
