import os
import shutil

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract


@datasets.register()
class pascal(DatasetBuilder):
    VERSIONS = ["2012"]
    VERSION_TO_URL = {
        # flake8: noqa
        "2012": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    }

    info = DatasetInfo(
        name="pascal",
        full_name="PASCAL",
        description="Image data sets for object class recognition.",
        homepage="http://host.robots.ox.ac.uk/pascal/VOC/",
        tags=["image", "object recognition"],
        citation=(
            "@Article{Everingham10,"
            'author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn,'
            'J. and Zisserman, A.",'
            'title = "The Pascal Visual Object Classes (VOC) Challenge",'
            'journal = "International Journal of Computer Vision",'
            'volume = "88",'
            'year = "2010",'
            'number = "2",'
            "month = jun,"
            'pages = "303--338",}'
        ),
    )

    def build(self):
        if self.version == "2012":
            return build_pascal_2012_df(dataset_dir=self.dataset_dir)
        else:
            raise ValueError()

    def download(self):
        url = self.VERSION_TO_URL[self.version]

        downloaded_path = download_url(url, self.dataset_dir)
        extract(downloaded_path, os.path.join(self.dataset_dir, "tmp"))
        shutil.move(
            os.path.join(self.dataset_dir, "tmp/VOCdevkit"),
            os.path.join(self.dataset_dir, "VOCdevkit"),
        )


def build_pascal_2012_df(dataset_dir: str):
    base_dir = os.path.join(dataset_dir, "VOCdevkit/VOC2012")

    train_df = mk.DataFrame.from_csv(
        os.path.join(base_dir, "ImageSets/Main/train.txt"), header=None, names=["id"]
    )
    train_df["split"] = ["train"] * len(train_df)

    val_df = mk.DataFrame.from_csv(
        os.path.join(base_dir, "ImageSets/Main/val.txt"), header=None, names=["id"]
    )
    val_df["split"] = ["val"] * len(val_df)

    df = mk.concat([train_df, val_df], axis=0)

    # create filename column
    df["file_name"] = df["id"] + ".jpg"
    df["image"] = mk.ImageColumn.from_filepaths(
        df["file_name"], base_dir=os.path.join(base_dir, "JPEGImages")
    )

    for class_name in PASCAL_CLASSES:
        label_df = mk.DataFrame.from_csv(
            os.path.join(
                base_dir, f"ImageSets/Main/{class_name}_trainval.txt".format(class_name)
            ),
            header=None,
            delimiter="  ?",
            names=["id", class_name],
            engine="python",
        )
        label_df[class_name] = (label_df[class_name] == 1).astype(int)
        df = df.merge(label_df, on="id", validate="one_to_one")

    return df


PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedflant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
