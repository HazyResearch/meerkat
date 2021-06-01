import hashlib
import os
from functools import partial
from typing import List

import pandas as pd
from torchvision.datasets import CelebA

ATTRIBUTES = [
    "5_o_clock_shadow",
    "arched_eyebrows",
    "attractive",
    "bags_under_eyes",
    "bald",
    "bangs",
    "big_lips",
    "big_nose",
    "black_hair",
    "blond_hair",
    "blurry",
    "brown_hair",
    "bushy_eyebrows",
    "chubby",
    "double_chin",
    "eyeglasses",
    "goatee",
    "gray_hair",
    "heavy_makeup",
    "high_cheekbones",
    "male",
    "mouth_slightly_open",
    "mustache",
    "narrow_eyes",
    "no_beard",
    "oval_face",
    "pale_skin",
    "pointy_nose",
    "receding_hairline",
    "rosy_cheeks",
    "sideburns",
    "smiling",
    "straight_hair",
    "wavy_hair",
    "wearing_earrings",
    "wearing_hat",
    "wearing_lipstick",
    "wearing_necklace",
    "wearing_necktie",
    "young",
]


def download_celeba(root_dir: str):
    CelebA(root_dir, download=True)
    build_celeba_df(os.path.join(root_dir, "celeba"))


def build_celeba_df(
    dataset_dir: str = "/afs/cs.stanford.edu/u/sabrieyuboglu/data/datasets/celeba",
    split_configs: List[dict] = None,
    salt: str = "abc",
):
    """Build the dataframe by joining on the attribute, split and identity
    CelebA CSVs."""
    identity_df = pd.read_csv(
        os.path.join(dataset_dir, "identity_CelebA.txt"),
        delim_whitespace=True,
        header=None,
        names=["file", "identity"],
    )
    attr_df = pd.read_csv(
        os.path.join(dataset_dir, "list_attr_celeba.txt"),
        delim_whitespace=True,
        header=1,
    )
    attr_df.columns = pd.Series(attr_df.columns).apply(lambda x: x.lower())
    attr_df = ((attr_df + 1) // 2).rename_axis("file").reset_index()

    celeb_df = identity_df.merge(attr_df, on="file", validate="one_to_one")

    celeb_df["img_path"] = celeb_df.file.apply(
        lambda x: os.path.join(dataset_dir, "img_align_celeba", x)
    )

    # add splits by hashing each file name to a number between 0 and 1
    if split_configs is None:
        split_configs = [{"split": "train", "size": len(celeb_df)}]

    # hash on identity to avoid same person straddling the train-test divide
    example_hash = celeb_df.identity.apply(partial(_hash_for_split, salt=salt))
    total_size = sum([config["size"] for config in split_configs])

    if total_size > len(celeb_df):
        raise ValueError("Total size cannot exceed full dataset size.")

    start = 0
    celeb_df["example_hash"] = example_hash
    dfs = []
    for config in split_configs:
        frac = config["size"] / total_size
        end = start + frac
        df = celeb_df[(start < example_hash) & (example_hash <= end)]
        df = df.sample(n=config["size"])
        df["split"] = config["split"]
        dfs.append(df)
        start = end
    df = pd.concat(dfs)
    df.to_csv(os.path.join(dataset_dir, "celeba.csv"))
    return df


def _hash_for_split(example_id: str, salt=""):
    GRANULARITY = 100000
    hashed = hashlib.sha256((str(example_id) + salt).encode())
    hashed = int(hashed.hexdigest().encode(), 16) % GRANULARITY + 1
    return hashed / float(GRANULARITY)
