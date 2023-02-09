import os

import numpy as np
import pandas as pd
import PIL

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets


def concat_images(x: PIL.Image.Image, y: PIL.Image.Image):
    return PIL.Image.fromarray(np.concatenate([np.array(x), np.array(y)], axis=1))


@datasets.register()
class rfw(DatasetBuilder):
    VERSIONS = ["main"]

    GROUPS = ["Caucasian", "African", "Asian", "Indian"]

    info = DatasetInfo(
        name="fer",
        full_name="Racial Faces in-the-Wild",
        # flake8: noqa
        description="Racial Faces in-the-Wild (RFW) is a testing database for studying racial bias in face recognition. Four testing subsets, namely Caucasian, Asian, Indian and African, are constructed, and each contains about 3000 individuals with 6000 image pairs for face verification. They can be used to fairly evaluate and compare the recognition ability of the algorithm on different races.",
        # flake8: noqa
        homepage="http://www.whdeng.cn/RFW/testing.html",
        tags=["image", "facial recognition", "algorithmic bias"],
    )

    def build(self):
        dfs = []
        for group in self.GROUPS:
            df = pd.read_csv(
                os.path.join(self.dataset_dir, f"test/txts/{group}/{group}_images.txt"),
                delimiter="\t",
                names=["filename", "count"],
            )

            df["ethnicity"] = group.lower()
            df["identity"] = df["filename"].str.rsplit("_", n=1).str[0]
            df["image_id"] = df["filename"].str.rsplit(".", n=1).str[0]
            df["image_path"] = df.apply(
                lambda x: f"test/data/{group}/{x['identity']}/{x['filename']}", axis=1
            )
            df.drop(columns=["filename", "count"])
            dfs.append(df)

        df = pd.concat(dfs)

        # drop duplicate rows with the same image_id
        df = df.drop_duplicates(subset=["image_id"], keep=False)

        df = mk.DataFrame.from_pandas(df)
        df["image"] = mk.ImageColumn.from_filepaths(
            df["image_path"], base_dir=self.dataset_dir
        )
        return df[["image_id", "identity", "ethnicity", "image"]]

        return None

    def download(self):
        raise ValueError(
            "To download the RFW dataset, you must request access following the "
            "instructions at http://www.whdeng.cn/RFW/testing.html."
            "Once you've been granted access and downloaded the data, move it "
            f"to the directory {self.dataset_dir} and extract it."
        )
