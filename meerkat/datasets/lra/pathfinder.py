"""Datasets for Long Range Arena (LRA) Pathfinder task."""
import itertools
import os

import pandas as pd

import meerkat as mk
from meerkat.tools.lazy_loader import LazyLoader

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url

tf = LazyLoader("tensorflow")


_RESOLUTIONS = ["32px", "64px", "128px", "256px"]
DIFFICULTY = {
    "curv_baseline": "easy",
    "curv_contour_length_9": "medium",
    "curv_contour_length_14": "hard",
}

_VERSIONS = _RESOLUTIONS + [
    f"{res}/{difficulty}"
    for res, difficulty in itertools.product(_RESOLUTIONS, DIFFICULTY.values())
]


@datasets.register()
class pathfinder(DatasetBuilder):
    """Long Range Arena (LRA) Pathfinder dataset."""

    VERSIONS = _VERSIONS

    info = DatasetInfo(
        name="pathfinder",
        full_name="Long Range Arena (LRA) Pathfinder",
        description=(
            "Image data set to determine whether two points represented "
            "as circles are connected by a path consisting of dashes."
        ),
        homepage="https://github.com/google-research/long-range-arena",
        tags=["image", "classification"],
        citation=None,
    )

    def build(self):
        """Get the processed dataframe hosted on huggingface."""
        difficulty = self.version.split("/")[-1] if "/" in self.version else None
        df = mk.DataFrame.read(
            self._get_huggingface_path(), overwrite=self.download_mode == "force"
        )
        if difficulty is not None:
            df = df[df["difficulty"] == difficulty]
        df["image"] = mk.files(df["path"], base_dir=self.dataset_dir, type="image")
        df.set_primary_key("path", inplace=True)
        return df

    def download(self):
        if not os.path.exists(self.dataset_dir):
            raise ValueError(
                f"Dataset {self.name} is not downloaded to {self.dataset_dir}. "
                "Please download the dataset from "
                "https://github.com/google-research/long-range-arena. "
                f"Extract the dataset to {self.dataset_dir}."
            )

    def is_downloaded(self):
        return os.path.exists(self.dataset_dir)

    def download_url(self, url: str):
        """Download files to a meerkat folder in the lra directory."""
        output_dir = os.path.join(self.dataset_dir, "meerkat")
        os.makedirs(output_dir, exist_ok=True)
        return download_url(url, output_dir, force=self.download_mode == "force")

    def _build_from_raw(self):
        """Build the dataset from the raw tensorflow files.

        This requires having tensorflow installed.
        """
        dfs = []
        difficulty = self.version.split("/")[-1] if "/" in self.version else None

        for subfolder in os.listdir(self.dataset_dir):
            # If a difficulty is specified, only include the specified difficulty.
            if difficulty is not None and DIFFICULTY[subfolder] != difficulty:
                continue

            dirpath = os.path.join(self.dataset_dir, subfolder, "")
            df = pd.DataFrame(_extract_metadata(dirpath, base_path=subfolder))
            df["subfolder"] = subfolder
            df["difficulty"] = DIFFICULTY[subfolder]
            dfs.append(df)

        # Concatenate the dataframes.
        df = pd.concat(dfs, axis=0)
        df = df.reset_index(drop=True)

        df = mk.DataFrame.from_pandas(df)
        df = df.drop("index")
        df["image"] = mk.files(df["path"], base_dir=self.dataset_dir, type="image")
        df.set_primary_key("path", inplace=True)
        return df

    @staticmethod
    def _get_dataset_dir(name: str, version: str) -> str:
        """
        self.dataset_dir will be: <root_dir>/lra_release/lra_release/pathfinder<res>
        e.g. /home/user/.meerkat/datasets/lra_release/lra_release/pathfinder32
        """
        res = _get_resolution(version)
        return os.path.join(
            mk.config.datasets.root_dir,
            "lra_release",
            "lra_release",
            f"pathfinder{res}",
        )

    def _get_huggingface_path(self):
        """Get path to the meerkat DataFrame uploaded to huggingface."""
        res = _get_resolution(self.version)
        return f"https://huggingface.co/datasets/meerkat-ml/pathfinder/resolve/main/pathfinder{res}.mk.tar.gz"  # noqa: E501


def _extract_metadata(dirpath, base_path: str):
    """Extract the filepath and label from the metadata file.

    Example metadata:
        ['imgs/43', 'sample_0.png', '0', '0', '1.8', '6', '2.0', '5', '1.5', '2', '1']

    Args:
        file_path: Path to the metadata file.
    """
    metadata_dir = os.path.join(dirpath, "metadata")
    image_paths = []
    labels = []
    for metadata_file in os.listdir(metadata_dir):
        file_path = os.path.join(metadata_dir, metadata_file)
        meta_examples = (
            tf.io.read_file(file_path).numpy().decode("utf-8").split("\n")[:-1]
        )
        for m_example in meta_examples:
            m_example = m_example.split(" ")
            image_paths.append(os.path.join(base_path, m_example[0], m_example[1]))
            labels.append(int(m_example[3]))
    return {"path": image_paths, "label": labels}


def _get_resolution(version: str):
    """Get the resolution from the version string."""
    return version.split("/")[0].split("px")[0]
