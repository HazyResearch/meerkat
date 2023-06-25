# Swedish Traffic Signs Dataset (STSD)

import os

import pandas as pd
from tqdm.auto import tqdm

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

_SETS_TO_URLS = {
    "Set1/annotations.txt": "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/annotations.txt",  # noqa: E501
    "Set2/annotations.txt": "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/annotations.txt",  # noqa: E501
    "Set1/Set1Part0": "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set1/Set1Part0.zip",  # noqa: E501
    "Set2/Set2Part0": "http://www.isy.liu.se/cvl/research/trafficSigns/swedishSignsSummer/Set2/Set2Part0.zip",  # noqa: E501
}


@datasets.register()
class stsd(DatasetBuilder):
    """Swedish Traffic Sign Dataset STSD."""

    VERSIONS = ["2019"]

    info = DatasetInfo(
        name="stsd",
        full_name="Swedish Traffic Sign Dataset STSD",
        description=("Image data set to detect street signs."),
        homepage="https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/download/",  # noqa: E501
        tags=["image", "object recognition"],
        citation=None,
    )

    def build(self):
        """Get the processed dataframe hosted on huggingface."""
        annotations = []
        for set_name in ["Set1", "Set2"]:
            ann_file = os.path.join(self.dataset_dir, f"{set_name}/annotations.txt")
            df = _format_annotations(ann_file)
            df["path"] = df["filename"].apply(
                lambda x: os.path.join(
                    self.dataset_dir, set_name, f"{set_name}Part0", x
                )
            )
            annotations.append(df)
        annotations = pd.concat(annotations).reset_index(drop=True)

        df = pd.DataFrame(annotations)
        df = mk.DataFrame.from_pandas(df).drop("index")
        df["image"] = mk.files(
            df["path"],
            type="image",
        )
        df["image_crop"] = mk.defer(df, crop)
        return df

    def download(self):
        for relative_path, url in tqdm(_SETS_TO_URLS.items(), verbose=True):
            downloaded_path = download_url(url, self.dataset_dir)
            path = os.path.join(self.dataset_dir, relative_path)
            if url.endswith(".zip"):
                os.makedirs(path, exist_ok=True)
                extract(downloaded_path, path)
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                os.rename(downloaded_path, path)

    def is_downloaded(self):
        return os.path.exists(self.dataset_dir) and all(
            os.path.exists(os.path.join(self.dataset_dir, x)) for x in _SETS_TO_URLS
        )


def crop(image, x1, y1, x2, y2):
    # Don't crop the image if the crop coordinates aren't valid.
    if any(v == -1 for v in [x1, y1, x2, y2]):
        return image.copy()
    out = image.crop((x1, y1, x2, y2))
    return out


def _format_annotations(ann_file):
    annotations = []
    with open(ann_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        filename, anns = (x.strip() for x in line.split(":"))
        for ann in anns.split(";"):
            ann = ann.strip()
            if len(ann) == 0:
                continue
            if ann == "MISC_SIGNS":
                annotations.append(
                    {
                        "filename": filename,
                        "visibility": "N/A",
                        "x1": -1,
                        "y1": -1,
                        "x2": -1,
                        "y2": -1,
                        "sign_type": "MISC_SIGNS",
                        "category": "MISC_SIGNS",
                    }
                )
                continue
            visibility, x2, y2, x1, y1, sign_type, name = (
                x.strip() for x in ann.split(",")
            )
            # Annotation file is malformed for this example.
            x2, y2, x1, y1 = [
                x.split("l")[0] if "l" in x else x for x in [x2, y2, x1, y1]
            ]
            annotations.append(
                {
                    "filename": filename,
                    "visibility": visibility,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "sign_type": sign_type,
                    "category": name,
                }
            )
    return pd.DataFrame(annotations)
