import os
from asyncio import subprocess

import numpy as np
import pandas as pd

import meerkat as mk


def build_mirflickr_25k_dp(dataset_dir: str, download: bool = False):

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
                "http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip",
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
    dp = mk.DataPanel({"id": ids, "file_name": file_names})
    dp["image"] = mk.ImageColumn.from_filepaths(
        dp["file_name"], base_dir=os.path.join(dataset_dir, "mirflickr")
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
        dp[class_name] = np.zeros(len(dp))
        dp[class_name][dp["id"].isin(ids)] = 1
    return dp


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
