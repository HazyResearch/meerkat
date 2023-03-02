import os

import meerkat as mk
from meerkat.columns.deferred.file import Downloader

DATASET_URL = "https://www.radar-service.eu/radar/en/dataset/tJzxrsYUkvPklBOw"


def build_dew_df(dataset_dir: str, download: bool = True) -> mk.DataFrame:
    if not os.path.exists(os.path.join(dataset_dir)):
        print(
            f"Please download the dataset from {DATASET_URL} and place it at "
            f"{dataset_dir}."
        )

    df = mk.DataFrame.from_csv(
        os.path.join(dataset_dir, "data/dataset/meta.csv"), parse_dates=["date_taken"]
    )

    df["image"] = mk.ImageColumn(
        df["url"],
        loader=Downloader(cache_dir=os.path.join(dataset_dir, "data/images")),
    )

    return df
