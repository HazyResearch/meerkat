import os
import subprocess

import pandas as pd

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_google_drive


@datasets.register()
class expw(DatasetBuilder):
    VERSION_TO_GDRIVE_ID = {"main": "19Eb_WiTsWelYv7Faff0L5Lmo1zv0vzwR"}
    VERSIONS = ["main"]

    info = DatasetInfo(
        name="expw",
        full_name="Expression in-the-Wild",
        description=(
            "Imagenette is a subset of 10 easily classified classes from Imagenet "
            "(tench, English springer, cassette player, chain saw, church, "
            "French horn, garbage truck, gas pump, golf ball, parachute)."
        ),
        homepage="https://github.com/fastai/imagenette",
        tags=["image", "classification"],
    )

    def build(self):
        df = pd.read_csv(
            os.path.join(self.dataset_dir, "label/label.lst"),
            delimiter=" ",
            names=[
                "image_name",
                "face_id_in_image",
                "face_box_top",
                "face_box_left",
                "face_box_right",
                "face_box_bottom",
                "face_box_cofidence",
                "expression_label",
            ],
        )
        df = df.drop_duplicates()
        df = mk.DataFrame.from_pandas(df)

        # ensure that all the image files are downloaded
        if (
            not df["image_name"]
            .apply(
                lambda name: os.path.exists(
                    os.path.join(self.dataset_dir, "image/origin", name)
                )
            )
            .all()
        ):
            raise ValueError(
                "Some images are not downloaded to expected directory: "
                f"{os.path.join(self.dataset_dir, 'image/origin')}. Verify download."
            )

        # remove file extension and add the face_id
        df["example_id"] = (
            df["image_name"].str.replace(".jpg", "", regex=False)
            + "_"
            + df["face_id_in_image"].astype(str)
        )

        df["image"] = mk.ImageColumn.from_filepaths(
            "image/origin/" + df["image_name"], base_dir=self.dataset_dir
        )
        df["face_image"] = df[
            "image",
            "face_box_top",
            "face_box_left",
            "face_box_right",
            "face_box_bottom",
        ].defer(crop)

        return df

    def download(self):
        gdrive_id = self.VERSION_TO_GDRIVE_ID[self.version]
        download_google_drive(id=gdrive_id, dst=self.dataset_dir, is_folder=True)
        os.makedirs(os.path.join(self.dataset_dir, "image"), exist_ok=True)

        for file in os.listdir(os.path.join(self.dataset_dir, "image")):
            # run 7za to extract the file using subprocess
            subprocess.call(
                [
                    "7za",
                    "x",
                    os.path.join(self.dataset_dir, "image", file),
                    "-o" + os.path.join(self.dataset_dir, "image"),
                ]
            )


def crop(row: dict):
    return row["image"].crop(
        (
            row["face_box_left"],
            row["face_box_top"],
            row["face_box_right"],
            row["face_box_bottom"],
        )
    )
