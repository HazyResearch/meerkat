import json
import os
from typing import List

import pandas as pd

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


def build_inaturalist_df(
    dataset_dir: str, download: bool = True, splits: List[str] = None
) -> mk.DataFrame:
    """Build a DataFrame from the inaturalist dataset.

    Args:
        dataset_dir: The directory to store the dataset in.
        download: Whether to download the dataset if it does not yet exist.
        splits: A list of splits to include. Defaults to all splits.
    """
    from torchvision.datasets.utils import download_and_extract_archive

    if splits is None:
        splits = ["train", "test", "val"]

    dfs = []
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
            df = mk.DataFrame(info["images"])

            # need to rename "id" so there aren't conflicts with the "id" in other
            # dataframes (see annotations below)
            df["image_id"] = df["id"]
            df.remove_column("id")

            # add image column
            df["image"] = mk.ImageColumn(df["file_name"], base_dir=dataset_dir)

            df["date"] = pd.to_datetime(df["date"])

            # add annotations for each image
            if split != "test":  # no annotations for test set
                annotation_df = mk.DataFrame(info["annotations"])
                annotation_df["annotation_id"] = annotation_df["id"]
                annotation_df.remove_column("id")
                df = df.merge(annotation_df, on="image_id")

                # join on the category table to get the category name
                category_df = mk.DataFrame(info["categories"])
                category_df["category_id"] = category_df["id"]
                category_df.remove_column("id")
                df = df.merge(category_df, on="category_id")

            dfs.append(df)

    return mk.concat(dfs) if len(dfs) > 1 else dfs[0]
