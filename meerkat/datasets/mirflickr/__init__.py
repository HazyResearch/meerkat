import os
from asyncio import subprocess

import numpy as np
import pandas as pd

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import extract


@datasets.register()
class mirflickr(DatasetBuilder):
    VERSIONS = ["25k"]

    VERSION_TO_URLS = {
        "25k": [
            "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip",
            # flake8: noqa
            "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip",
        ]
    }

    info = DatasetInfo(
        name="mirflickr",
        full_name="PASCAL",
        description=(
            "The MIRFLICKR-25000 open evaluation project consists of 25000 images "
            "downloaded from the social photography site Flickr through its public API "
            " coupled with complete manual annotations, pre-computed descriptors and "
            "software for bag-of-words based similarity and classification and a "
            "matlab-like tool for exploring and classifying imagery."
        ),
        homepage="https://press.liacs.nl/mirflickr/",
        tags=["image", "retrieval"],
        citation=(
            "@inproceedings{huiskes08,"
            "    author = {Mark J. Huiskes and Michael S. Lew},"
            "    title = {The MIR Flickr Retrieval Evaluation},"
            "    booktitle = {MIR '08: Proceedings of the 2008 ACM International"
            " Conference on Multimedia Information Retrieval},"
            "    year = {2008},"
            "    location = {Vancouver, Canada},"
            "    publisher = {ACM},"
            "    address = {New York, NY, USA},"
            "}"
        ),
    )

    def download(self):
        urls = self.VERSION_TO_URLS[self.version]
        for url in urls:
            downloaded_path = self.download_url(url)
            extract(downloaded_path, self.dataset_dir)

    def build(self) -> mk.DataFrame:
        # get list of image ids
        file_names = pd.Series(
            [
                f
                for f in os.listdir(os.path.join(self.dataset_dir, "mirflickr"))
                if f.endswith(".jpg")
            ]
        )
        # remove jpg extension
        ids = file_names.str.replace(".jpg", "", regex=False)
        df = mk.DataFrame({"id": ids, "file_name": file_names})

        df["image"] = mk.ImageColumn.from_filepaths(
            df["file_name"], base_dir=os.path.join(self.var_dataset_dir, "mirflickr")
        )

        for class_name in MIR_FLICKR_25K_CLASSES:
            ids = (
                "im"
                + pd.read_csv(
                    os.path.join(self.dataset_dir, class_name + ".txt"),
                    header=None,
                    names=["id"],
                ).astype(str)["id"]
            )
            df[class_name] = np.zeros(len(df))
            df[class_name][df["id"].isin(ids)] = 1
        return df


def build_mirflickr_25k_df(dataset_dir: str, download: bool = False):
    if download:
        subprocess.run(
            [
                "wget",
                "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip",
            ]
        )
        subprocess.run(["unzip", "mirflickr25k.zip"])
        os.remove("mirflickr25k.zip")

        subprocess.run(
            [
                "wget",
                "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip",  # noqa: E501
            ]
        )
        subprocess.run(["unzip", "mirflickr25k_annotations_v080.zip"])
        os.remove("mirflickr25k_annotations_v080.zip")

    # get list of image ids
    file_names = pd.Series(
        [
            f
            for f in os.listdir(os.path.join(dataset_dir, "mirflickr"))
            if f.endswith(".jpg")
        ]
    )

    # remove jpg extension
    ids = file_names.str.replace(".jpg", "", regex=False)
    df = mk.DataFrame({"id": ids, "file_name": file_names})
    df["image"] = mk.ImageColumn.from_filepaths(
        df["file_name"], base_dir=os.path.join(dataset_dir, "mirflickr")
    )

    for class_name in MIR_FLICKR_25K_CLASSES:
        ids = (
            "im"
            + pd.read_csv(
                os.path.join(dataset_dir, class_name + ".txt"),
                header=None,
                names=["id"],
            ).astype(str)["id"]
        )
        df[class_name] = np.zeros(len(df))
        df[class_name][df["id"].isin(ids)] = 1
    return df


MIR_FLICKR_25K_CLASSES = [
    "animals",
    "baby",
    "bird",
    "car",
    "clouds",
    "dog",
    "female",
    "flower",
    "food",
    "indoor",
    "lake",
    "male",
    "night",
    "people",
    "plant_life",
    "portrait",
    "river",
    "sea",
    "sky",
    "structures",
    "sunset",
    "transport",
    "tree",
    "water",
]
