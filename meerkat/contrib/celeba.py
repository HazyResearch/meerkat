import os
import subprocess

import pandas as pd

import meerkat as mk

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


def get_celeba(dataset_dir: str, download: bool = False):
    """Build the dataframe by joining on the attribute, split and identity
    CelebA CSVs."""
    if download:
        download_celeba(dataset_dir=dataset_dir)
    df = build_celeba_df(dataset_dir=dataset_dir)
    dp = mk.DataPanel.from_pandas(df)
    dp["image"] = mk.ImageColumn.from_filepaths(
        filepaths=dp["img_path"], base_dir=dataset_dir
    )
    return dp


def download_celeba(dataset_dir: str):
    # if not os.path.exists(dataset_dir):
    #     CelebA(os.path.split(dataset_dir)[:-1][0], download=True)
    if os.path.exists(dataset_dir):
        return
    curr_dir = os.getcwd()
    os.makedirs(dataset_dir, exist_ok=True)
    os.chdir(dataset_dir)
    subprocess.run(
        args=["kaggle datasets download " "-d jessicali9530/celeba-dataset"],
        shell=True,
        check=True,
    )
    subprocess.run(["unzip", "-q", "celeba-dataset.zip"])
    os.chdir(curr_dir)


def build_celeba_df(dataset_dir: str):
    """Build the dataframe by joining on the attribute, split and identity
    CelebA CSVs."""
    identity_df = pd.read_csv(
        os.path.join(dataset_dir, "identity_CelebA.txt"),
        delim_whitespace=True,
        header=None,
        names=["file", "identity"],
    )
    attr_df = pd.read_csv(
        os.path.join(dataset_dir, "list_attr_celeba.csv"),
        index_col=0,
    )
    attr_df.columns = pd.Series(attr_df.columns).apply(lambda x: x.lower())
    attr_df = ((attr_df + 1) // 2).rename_axis("file").reset_index()

    celeb_df = identity_df.merge(attr_df, on="file", validate="one_to_one")

    celeb_df["img_path"] = celeb_df.file.apply(
        lambda x: os.path.join("img_align_celeba/img_align_celeba", x)
    )

    split_df = pd.read_csv(os.path.join(dataset_dir, "list_eval_partition.csv"))
    split_df["split"] = split_df["partition"].replace(
        {0: "train", 1: "valid", 2: "test"}
    )
    celeb_df = celeb_df.merge(
        split_df[["image_id", "split"]], left_on="file", right_on="image_id"
    )
    return celeb_df
