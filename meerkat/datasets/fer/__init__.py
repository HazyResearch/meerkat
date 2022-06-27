import os
import subprocess
from typing import Dict

import numpy as np
import pandas as pd

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract


@datasets.register()
class fer(DatasetBuilder):

    VERSIONS = ["plus"]

    info = DatasetInfo(
        name="fer",
        full_name="ImageNet",
        # flake8: noqa
        description="ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images..",
        # flake8: noqa
        homepage="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv",
        tags=["image", "facial emotion recognition"],
    )

    def build(self):
        pass 
        return None

    def download(self):
        curr_dir = os.getcwd()
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.chdir(self.dataset_dir)
        subprocess.run(
            args=[
                "kaggle competitions download "
                "-c challenges-in-representation-learning-facial-expression-recognition-challenge",
            ],
            shell=True,
            check=True,
        )
        extract("fer2013.tar.gz", "fer2013")
        os.chdir(curr_dir)

