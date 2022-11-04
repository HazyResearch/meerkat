import os
from typing import Dict, Mapping

import ujson as json

import meerkat as mk

# TODO (Sabri): Add support for downloading the data.
# For the time being, relevant files should be downloaded to the directory here
# https://visualgenome.org/VGViz/explore


def build_visual_genome_dfs(
    dataset_dir: str, write: bool = False
) -> Dict[str, mk.DataFrame]:
    dfs = {}
    print("Loading objects and attributes...")
    dfs.update(_build_object_dfs(dataset_dir))

    print("Loading images...")
    dfs.update(_build_image_df(dataset_dir=dataset_dir))

    print("Loading relationships...")
    dfs.update(_build_relationships_df(dataset_dir=dataset_dir))

    if write:
        write_visual_genome_dfs(dfs, dataset_dir=dataset_dir)
    return dfs


def read_visual_genome_dfs(dataset_dir: str) -> Dict[str, mk.DataFrame]:
    return {
        key: mk.DataFrame.read(os.path.join(dataset_dir, f"{key}.mk"))
        for key in ["attributes", "relationships", "objects", "images"]
    }


def write_visual_genome_dfs(dfs: Mapping[str, mk.DataFrame], dataset_dir: str):
    for key, df in dfs.items():
        df.write(os.path.join(dataset_dir, f"{key}.mk"))


def _build_object_dfs(dataset_dir: str):
    with open(os.path.join(dataset_dir, "attributes.json")) as f:
        objects = json.load(f)

    objects_df = []  # create one table for objects
    attributes_df = []  # create one table of attributes
    for image in objects:
        for obj in image["attributes"]:
            obj["image_id"] = image["image_id"]

            # all names are length 1
            names = obj.pop("names")
            obj["name"] = names[0]

            # add attributes to the table
            attributes = obj.pop("attributes", None)
            if attributes is not None:
                for attribute in attributes:
                    attributes_df.append(
                        {"object_id": obj["object_id"], "attribute": attribute}
                    )

            # the vast majority of objects (99.7%) have 0 or 1 synonym in their
            # synset, so we only consider the first synonym to keep things simple
            synset = obj.pop("synsets")
            obj["syn_name"] = synset[0] if len(synset) > 0 else ""

            objects_df.append(obj)

    return {
        "objects": mk.DataFrame(objects_df),
        "attributes": mk.DataFrame(attributes_df),
    }


def _build_image_df(dataset_dir: str):
    with open(os.path.join(dataset_dir, "image_data.json")) as f:
        images = json.load(f)
    image_df = mk.DataFrame(images)
    image_df.remove_column("coco_id")
    image_df.remove_column("flickr_id")

    image_df["local_path"] = dataset_dir + (image_df["url"].str.split("rak248")).apply(
        lambda x: x[-1]
    )

    image_df["image"] = mk.ImageColumn(image_df["local_path"])

    return {"images": image_df}


def _build_relationships_df(dataset_dir: str):
    with open(os.path.join(dataset_dir, "relationships.json")) as f:
        relationships = json.load(f)

    rel_df = []
    for image in relationships:
        image_id = image["image_id"]
        for r in image["relationships"]:
            object_synset = r["object"]["synsets"]
            subject_synset = r["subject"]["synsets"]

            rel_df.append(
                {
                    "image_id": image_id,
                    "predicate": r["predicate"],
                    "subject_object_id": r["subject"]["object_id"],
                    "subject_name": r["subject"]["name"]
                    if "name" in r["subject"]
                    else r["subject"]["names"][0],
                    "subject_syn": subject_synset[0] if len(subject_synset) > 0 else "",
                    "object_object_id": r["object"]["object_id"],
                    "object_name": r["object"]["name"]
                    if "name" in r["object"]
                    else r["object"]["names"][0],
                    "object_syn": object_synset[0] if len(object_synset) > 0 else "",
                }
            )
    rel_df = mk.DataFrame(rel_df)
    return {"relationships": rel_df}
