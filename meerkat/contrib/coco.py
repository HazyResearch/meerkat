import json
import os
import subprocess

import meerkat as mk


def build_coco_2014_dp(dataset_dir: str, download: bool = False):
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
                    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
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

    dps = []
    for split in ["train", "val"]:
        dct = json.load(
            open(
                os.path.join(dataset_dir, f"annotations/instances_{split}2014.json"),
                "rb",
            )
        )

        dp = mk.DataPanel(dct["images"])
        dp["split"] = [split] * len(dp)
        dps.append(dp)

    dp = mk.concat(dps, axis=0)

    path = dp["split"] + "2014/" + dp["file_name"]
    dp["image"] = mk.ImageColumn.from_filepaths(path, base_dir=dataset_dir)

    dp.data.reorder(
        ["id", "image"] + [c for c in dp.columns if c not in ["id", "image"]]
    )

    return dp
