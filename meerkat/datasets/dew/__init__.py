import os

import meerkat as mk
from meerkat.columns.file_column import Downloader

DATASET_URL = "https://www.radar-service.eu/radar/en/dataset/tJzxrsYUkvPklBOw"


def build_dew_dp(dataset_dir: str, download: bool = True) -> mk.DataPanel:

    if not os.path.exists(os.path.join(dataset_dir)):
        print(
            f"Please download the dataset from {DATASET_URL} and place it at "
            f"{dataset_dir}."
        )

    dp = mk.DataPanel.from_csv(
        os.path.join(dataset_dir, "data/dataset/meta.csv"), parse_dates=["date_taken"]
    )

    dp["image"] = mk.ImageColumn(
        dp["url"],
        loader=Downloader(cache_dir=os.path.join(dataset_dir, "data/images")),
    )

    return dp
