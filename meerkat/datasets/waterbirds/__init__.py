import os

import pandas as pd
from wilds import get_dataset
from wilds.datasets.wilds_dataset import WILDSDataset

import meerkat as mk

# flake8: noqa
URL = "http://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"


def build_waterbirds_df(
    dataset_dir: str,
    download: bool = True,
):
    """Download and load the Waterbirds dataset.

    Args:
        download_dir (str): The directory to save to.

    Returns:
        a DataFrame containing columns `image`, `y`, "background", and `split`,

    References:
    """
    dataset = get_dataset(dataset="waterbirds", root_dir=dataset_dir, download=download)

    df = pd.DataFrame(dataset.metadata_array, columns=dataset.metadata_fields)
    df["filepath"] = dataset._input_array

    df = mk.DataFrame.from_pandas(df)
    df["image"] = mk.ImageColumn(
        df["filepath"], base_dir=os.path.join(dataset_dir, "waterbirds_v1.0")
    )
    df["split"] = pd.Series(dataset._split_array).map(
        {
            v: k if k != "val" else "valid"
            for k, v in WILDSDataset.DEFAULT_SPLITS.items()
        }
    )

    backgrounds = dataset._metadata_map["background"]
    birds = dataset._metadata_map["y"]
    group_mapping = {
        f"{bird_idx}{bground_idx}": f"{birds[bird_idx]}-{backgrounds[bground_idx]}"
        for bird_idx in [0, 1]
        for bground_idx in [0, 1]
    }

    df["group"] = (df["y"].astype(str) + df["background"].data.astype(str)).map(
        group_mapping
    )
    return df
