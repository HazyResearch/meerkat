import os
import subprocess

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import extract


@datasets.register()
class fer(DatasetBuilder):
    VERSIONS = ["plus"]

    info = DatasetInfo(
        name="fer",
        full_name="Facial Expression Recognition Challenge",
        # flake8: noqa
        description="ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images..",
        # flake8: noqa
        homepage="https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=icml_face_data.csv",
        tags=["image", "facial emotion recognition"],
    )

    def build(self):
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
        # extract(
        #     os.path.join(
        #         self.dataset_dir,
        #         "challenges-in-representation-learning-facial-expression-recognition-challenge.zip",
        #     ),
        #     "fer2013"
        # )
        extract(
            os.path.join(self.dataset_dir, "fer2013", "fer2013.tar.gz"),
            os.path.join(self.dataset_dir, "fer2013", "images"),
        )
        os.chdir(curr_dir)
