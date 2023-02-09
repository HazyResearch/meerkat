import json
import os
import subprocess

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

IMAGE_URL = "http://images.cocodataset.org/zips/{split}{version}.zip"
# flake8: noqa
TEST_LABEL_URL = (
    "http://images.cocodataset.org/annotations/image_info_test{version}.zip"
)
# flake8: noqa
TRAIN_VAL_LABEL_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval{version}.zip"
)


@datasets.register()
class coco(DatasetBuilder):
    VERSIONS = ["2014"]

    info = DatasetInfo(
        name="coco",
        full_name="Common Objects in Context",
        description="Image data sets for object class recognition.",
        homepage="https://cocodataset.org/#home",
        tags=["image", "object recognition"],
        citation=None,
    )

    def build(self):
        dfs = []
        for split in ["train", "val"]:
            dct = json.load(
                open(
                    os.path.join(
                        self.dataset_dir, f"annotations/instances_{split}2014.json"
                    ),
                    "rb",
                )
            )
            breakpoint()

            df = mk.DataFrame(dct["images"])
            df["split"] = [split] * len(df)
            dfs.append(df)

        df = mk.concat(dfs, axis=0)

        path = df["split"] + "2014/" + df["file_name"]
        df["image"] = mk.ImageColumn.from_filepaths(path, base_dir=self.var_dataset_dir)

        df.data.reorder(
            ["id", "image"] + [c for c in df.columns if c not in ["id", "image"]]
        )
        return df

    def download(self):
        for split in ["train", "val", "test"]:
            downloaded_path = download_url(
                IMAGE_URL.format(version=self.version, split=split), self.dataset_dir
            )
            extract(downloaded_path, self.dataset_dir)

        downloaded_path = download_url(
            TEST_LABEL_URL.format(version=self.version), self.dataset_dir
        )
        extract(downloaded_path, self.dataset_dir)

        downloaded_path = download_url(
            TRAIN_VAL_LABEL_URL.format(version=self.version), self.dataset_dir
        )
        extract(downloaded_path, self.dataset_dir)


def build_coco_2014_df(dataset_dir: str, download: bool = False):
    if download:
        curr_dir = os.getcwd()
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)
        for split in ["train", "val", "test"]:
            if not os.path.exists(f"{split}2014"):
                subprocess.run(
                    args=[
                        "wget",
                        f"http://images.cocodataset.org/zips/{split}2014.zip",
                    ],
                    shell=True,
                    check=True,
                )
                subprocess.run(["unzip", f"{split}2014.zip"])
                subprocess.run(["rm", f"{split}2014.zip"])

        # download train and test annotations
        if not os.path.exists("annotations/captions_train2014.json"):
            subprocess.run(
                args=[
                    "wget",
                    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",  # noqa: E501
                ]
            )
            subprocess.run(["unzip", "annotations_trainval2014.zip"])
            subprocess.run(["rm", "annotations_trainval2014.zip"])

        # download test image info
        if not os.path.exists("annotations/image_info_test2014.json"):

            subprocess.run(
                args=[
                    "wget",
                    "http://images.cocodataset.org/annotations/image_info_test2014.zip",
                ]
            )
            subprocess.run(["unzip", "image_info_test2014.zip"])
            subprocess.run(["rm", "image_info_test2014.zip"])

        os.chdir(curr_dir)

    dfs = []
    for split in ["train", "val"]:
        dct = json.load(
            open(
                os.path.join(dataset_dir, f"annotations/instances_{split}2014.json"),
                "rb",
            )
        )

        df = mk.DataFrame(dct["images"])
        df["split"] = [split] * len(df)
        dfs.append(df)

    df = mk.concat(dfs, axis=0)

    path = df["split"] + "2014/" + df["file_name"]
    df["image"] = mk.ImageColumn.from_filepaths(path, base_dir=dataset_dir)

    df.data.reorder(
        ["id", "image"] + [c for c in df.columns if c not in ["id", "image"]]
    )

    return df
