import json
import os
from typing import List

import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive

import meerkat as mk

IMAGES_URLS = {
    "train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz",
    "val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
    "test": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.tar.gz",  # noqa: E501
}

INFO_URLS = {
    "train": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.json.tar.gz",  # noqa: E501
    "val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
    "test": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/public_test.json.tar.gz",  # noqa: E501
}


def build_inaturalist_dp(
    dataset_dir: str, download: bool = True, splits: List[str] = None
) -> mk.DataPanel:
    """
    Build a DataPanel from the inaturalist dataset.

    Args:
        dataset_dir: The directory to store the dataset in.
        download: Whether to download the dataset if it does not yet exist.
        splits: A list of splits to include. Defaults to all splits.
    """

    if splits is None:
        splits = ["train", "test", "val"]

    dps = []
    for split in splits:
        if not os.path.exists(os.path.join(dataset_dir, split)) and download:
            download_and_extract_archive(
                IMAGES_URLS[split], download_root=dataset_dir, remove_finished=True
            )
        if not os.path.exists(os.path.join(dataset_dir, f"{split}.json")) and download:
            download_and_extract_archive(
                INFO_URLS[split], download_root=dataset_dir, remove_finished=True
            )

        with open(os.path.join(dataset_dir, f"{split}.json"), "r") as f:
            info = json.load(f)
            dp = mk.DataPanel(info["images"])

            dp["date"] = pd.to_datetime(dp["date"])
            dp["image"] = mk.ImageColumn(dp["file_name"], base_dir=dataset_dir)
            dps.append(dp)

    return mk.concat(dps)
