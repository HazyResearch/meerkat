import json
import os
from typing import Dict, Mapping

from tqdm import tqdm

import meerkat as mk


def crop_object(row: Mapping[str, object]):
    img = row["image"]

    length = max(row["h"], row["w"])
    box = (
        max(row["x"] - ((length - row["w"]) / 2), 0),
        max(row["y"] - ((length - row["h"]) / 2), 0),
        min(row["x"] + row["w"] + ((length - row["w"]) / 2), img.width),
        min(row["y"] + row["h"] + ((length - row["h"]) / 2), img.height),
    )
    return img.crop(box)


def build_gqa_dps(dataset_dir: str, write: bool = False) -> Dict[str, mk.DataPanel]:

    objects = []
    images = []
    relations = []
    attributes = []
    for split in ["train", "val"]:
        print(f"Loading {split} scene graphs...")
        with open(os.path.join(dataset_dir, f"{split}_sceneGraphs.json")) as f:
            graphs = json.load(f)
        for image_id, graph in tqdm(graphs.items()):
            image_id = int(image_id)  # convert to int for faster filtering and joins
            for object_id, obj in graph.pop("objects").items():
                object_id = int(
                    object_id
                )  # convert to int for faster filtering and joins
                for relation in obj.pop("relations"):
                    relations.append(
                        {
                            "subject_object_id": object_id,
                            "object_id": int(relation["object"]),
                            "name": relation["name"],
                        }
                    )
                for attribute in obj.pop("attributes"):
                    attributes.append({"object_id": object_id, "attribute": attribute})
                objects.append({"object_id": object_id, "image_id": image_id, **obj})
            images.append({"image_id": image_id, **graph})

    # prepare DataPanels
    print("Preparing DataPanels...")
    image_dp = mk.DataPanel(images)
    image_dp["image"] = mk.ImageColumn(
        image_dp["image_id"].map(
            lambda x: os.path.join(dataset_dir, "images", f"{x}.jpg")
        )
    )
    object_dp = mk.DataPanel(objects).merge(
        image_dp[["image_id", "image", "height", "width"]], on="image_id"
    )
    object_dp["object_image"] = object_dp.to_lambda(crop_object)
    # filter out objects with no width or height
    object_dp = object_dp.lz[(object_dp["h"] != 0) & (object_dp["w"] != 0)]
    # filter out objects whose bounding boxes are not contained within the image
    object_dp = object_dp.lz[
        (object_dp["x"] < object_dp["width"]) & (object_dp["y"] < object_dp["height"])
    ]

    dps = {
        "images": image_dp,
        "objects": object_dp,
        "relations": mk.DataPanel(relations),
        "attributes": mk.DataPanel(attributes),
    }

    if write:
        write_gqa_dps(dps=dps, dataset_dir=dataset_dir)
    return dps


def read_gqa_dps(dataset_dir: str) -> Dict[str, mk.DataPanel]:
    return {
        key: mk.DataPanel.read(os.path.join(dataset_dir, f"{key}.mk"))
        for key in ["attributes", "relations", "objects", "images"]
    }


def write_gqa_dps(dps: Mapping[str, mk.DataPanel], dataset_dir: str):
    for key, dp in dps.items():
        dp.write(os.path.join(dataset_dir, f"{key}.mk"))
