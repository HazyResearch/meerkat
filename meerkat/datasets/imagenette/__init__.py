import os
import tarfile

import pandas as pd

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

ID_TO_WORDS = {
    "n02979186": "cassette player",
    "n03417042": "garbage truck",
    "n01440764": "tench",
    "n02102040": "english springer spaniel",
    "n03028079": "church",
    "n03888257": "parachute",
    "n03394916": "french horn",
    "n03000684": "chainsaw",
    "n03445777": "golf ball",
    "n03425413": "gas pump",
}

ID_TO_IDX = {
    "n02979186": 482,
    "n03417042": 569,
    "n01440764": 0,
    "n02102040": 217,
    "n03028079": 497,
    "n03888257": 701,
    "n03394916": 566,
    "n03000684": 491,
    "n03445777": 574,
    "n03425413": 571,
}


@datasets.register()
class imagenette(DatasetBuilder):
    VERSION_TO_URL = {
        "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "320px": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        "160px": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    }
    VERSIONS = ["full", "320px", "160px"]

    info = DatasetInfo(
        name="imagenette",
        full_name="ImageNette",
        description=(
            "Imagenette is a subset of 10 easily classified classes from Imagenet "
            "(tench, English springer, cassette player, chain saw, church, "
            "French horn, garbage truck, gas pump, golf ball, parachute)."
        ),
        homepage="https://github.com/fastai/imagenette",
        tags=["image", "classification"],
    )

    @property
    def data_dir(self):
        return os.path.join(
            self.dataset_dir,
            self.VERSION_TO_URL[self.version].split("/")[-1].split(".")[0],
        )

    def build(self):
        df = self._build_df()
        df = mk.DataFrame.from_pandas(df)
        df["img"] = mk.files(df["img_path"], base_dir=self.data_dir, type="image")
        df.set_primary_key("img_id", inplace=True)
        return df

    def download(self):
        url = self.VERSION_TO_URL[self.version]
        path = self.download_url(url)
        extract(path, self.dataset_dir)

    def _build_df(
        self,
    ):
        csv_path = os.path.join(self.data_dir, "noisy_imagenette.csv")

        df = pd.read_csv(csv_path)
        df["label_id"] = df["noisy_labels_0"]
        df["label"] = df["label_id"].replace(ID_TO_WORDS)
        df["label_idx"] = df["label_id"].replace(ID_TO_IDX)
        df["split"] = df["is_valid"].replace({False: "train", True: "valid"})
        df["img_path"] = df.path
        df["img_id"] = df["path"].apply(lambda x: x.split("/")[-1].split(".")[0])
        return df


def download_imagenette(
    download_dir,
    version="160px",
    overwrite: bool = False,
    return_df: bool = False,
):
    """Download Imagenette dataset.

    Args:
        download_dir (str): The directory path to save to.
        version (str, optional): Imagenette version.
            Choices: ``"full"``, ``"320px"``, ``"160px"``.
        overwrite (bool, optional): If ``True``, redownload the dataset.
        return_df (bool, optional): If ``True``, return a ``pd.DataFrame``.

    Returns:
        Union[str, pd.DataFrame]: If ``return_df=True``, returns a pandas DataFrame.
            Otherwise, returns the directory path where the data is stored.

    References:
        https://github.com/fastai/imagenette
    """
    tar_path = os.path.join(
        download_dir, os.path.basename(imagenette.VERSION_TO_URL[version])
    )
    dir_path = os.path.splitext(tar_path)[0]
    csv_path = os.path.join(dir_path, "imagenette.csv")
    if not overwrite and os.path.isfile(csv_path):
        return (pd.read_csv(csv_path), dir_path) if return_df else dir_path

    if overwrite or not os.path.exists(dir_path):
        cached_tar_path = download_url(
            url=imagenette.VERSION_TO_URL[version],
            dataset_dir=download_dir,
        )
        print("Extracting tar archive, this may take a few minutes...")
        tar = tarfile.open(cached_tar_path)
        tar.extractall(download_dir)
        tar.close()
        # os.remove(tar_path)
    else:
        print(f"Directory {dir_path} already exists. Skipping download.")

    # build dataframe
    df = pd.read_csv(os.path.join(dir_path, "noisy_imagenette.csv"))
    df["label_id"] = df["noisy_labels_0"]
    df["label"] = df["label_id"].replace(ID_TO_WORDS)
    df["label_idx"] = df["label_id"].replace(ID_TO_IDX)
    df["split"] = df["is_valid"].replace({False: "train", True: "valid"})
    df["img_path"] = df.path
    df[["label", "split", "img_path"]].to_csv(csv_path, index=False)


def build_imagenette_df(
    dataset_dir: str,
    download: bool = False,
    version: str = "160px",
) -> mk.DataFrame:
    """Build DataFrame for the Imagenette dataset.

    Args:
        download_dir (str): The directory path to save to or load from.
        version (str, optional): Imagenette version.
            Choices: ``"full"``, ``"320px"``, ``"160px"``.
        overwrite (bool, optional): If ``True``, redownload the datasets.

    Returns:
        mk.DataFrame: A DataFrame corresponding to the dataset.

    References:
        https://github.com/fastai/imagenette
    """
    if download:
        df, dir_path = download_imagenette(
            dataset_dir, version=version, overwrite=False, return_df=True
        )
    else:
        csv_path = os.path.join(dataset_dir, "imagenette.csv")
        if not os.path.isfile(csv_path):
            raise ValueError("Imagenette is not downloaded. Pass `download=True`.")
        df = pd.read_csv(csv_path)

    df = mk.DataFrame.from_pandas(df)
    df["img"] = mk.ImageColumn.from_filepaths(df["img_path"], base_dir=dir_path)
    return df
