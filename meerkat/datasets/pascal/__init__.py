import os

import meerkat as mk

from ..abstract import DatasetBuilder


class PascalDatasetBuilder(DatasetBuilder):
    REVISIONS = [2012]

    def build(self):
        if self.revision == "2012":
            return build_pascal_2012_dp(dataset_dir=self.dataset_dir)

    def download(self):
        pass 

    def is_downloaded(self):
        return False 


def build_pascal_2012_dp(dataset_dir: str):

    base_dir = os.path.join(dataset_dir, "VOCdevkit/VOC2012")

    train_dp = mk.DataPanel.from_csv(
        os.path.join(base_dir, "ImageSets/Main/train.txt"), header=None, names=["id"]
    )
    train_dp["split"] = ["train"] * len(train_dp)

    val_dp = mk.DataPanel.from_csv(
        os.path.join(base_dir, "ImageSets/Main/val.txt"), header=None, names=["id"]
    )
    val_dp["split"] = ["val"] * len(val_dp)

    dp = mk.concat([train_dp, val_dp], axis=0)

    # create filename column
    dp["file_name"] = dp["id"] + ".jpg"
    dp["image"] = mk.ImageColumn.from_filepaths(
        dp["file_name"], base_dir=os.path.join(base_dir, "JPEGImages")
    )

    for class_name in PASCAL_CLASSES:
        label_dp = mk.DataPanel.from_csv(
            os.path.join(
                base_dir, f"ImageSets/Main/{class_name}_trainval.txt".format(class_name)
            ),
            header=None,
            delimiter="  ?",
            names=["id", class_name],
            engine="python",
        )
        label_dp[class_name] = (label_dp[class_name] == 1).astype(int)
        dp = dp.merge(label_dp, on="id", validate="one_to_one")

    return dp


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
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
