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


def build_gqa_dfs(dataset_dir: str, write: bool = False) -> Dict[str, mk.DataFrame]:
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

    # prepare DataFrames
    print("Preparing DataFrames...")
    image_df = mk.DataFrame(images)
    image_df["image"] = mk.ImageColumn(
        image_df["image_id"].map(
            lambda x: os.path.join(dataset_dir, "images", f"{x}.jpg")
        )
    )
    object_df = mk.DataFrame(objects).merge(
        image_df[["image_id", "image", "height", "width"]], on="image_id"
    )
    object_df["object_image"] = object_df.to_lambda(crop_object)
    # filter out objects with no width or height
    object_df = object_df[(object_df["h"] != 0) & (object_df["w"] != 0)]
    # filter out objects whose bounding boxes are not contained within the image
    object_df = object_df[
        (object_df["x"] < object_df["width"]) & (object_df["y"] < object_df["height"])
    ]

    dfs = {
        "images": image_df,
        "objects": object_df,
        "relations": mk.DataFrame(relations),
        "attributes": mk.DataFrame(attributes),
    }

    if write:
        write_gqa_dfs(dfs=dfs, dataset_dir=dataset_dir)
    return dfs


def read_gqa_dfs(dataset_dir: str) -> Dict[str, mk.DataFrame]:
    return {
        key: mk.DataFrame.read(os.path.join(dataset_dir, f"{key}.mk"))
        for key in ["attributes", "relations", "objects", "images"]
    }


def write_gqa_dfs(dfs: Mapping[str, mk.DataFrame], dataset_dir: str):
    for key, df in dfs.items():
        df.write(os.path.join(dataset_dir, f"{key}.mk"))
