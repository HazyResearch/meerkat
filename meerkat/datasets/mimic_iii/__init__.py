import os
import subprocess

import PIL

import meerkat as mk
from meerkat.columns.deferred.image import load_image

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets


def _write_empty_image(dst):
    img = PIL.Image.new("RGB", (32, 32), color="black")
    img.save(dst, format="JPEG")


@datasets.register()
class mimic_iii(DatasetBuilder):
    VERSIONS = ["main"]

    info = DatasetInfo(
        name="mimic-iii",
        full_name="MIMIC-III",
        # flake8: noqa
        description="MIMIC-III (Medical Information Mart for Intensive Care III) is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.",
        homepage="https://mimic.mit.edu/docs/iii/",
        tags=["medicine"],
        citation=None,
    )

    def build(self):
        pass

    def download(self):
        # clone the repo using subprocess
        # "wget -r -N -c -np --user seyuboglu --ask-password https://physionet.org/files/mimiciii/1.4/"
        subprocess.call(
            [
                "wget",
                "-r",
                "-N",
                "-c",
                "-np",
                "-P",
                self.dataset_dir,
                "--ask-user",
                "seyuboglu",
                "--ask-password",
                "https://physionet.org/files/mimiciii/1.4/",
            ]
        )
