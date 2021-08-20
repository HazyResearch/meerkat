import os
from typing import Dict, Mapping

import ujson as json

import meerkat as mk

# TODO (Sabri): Add support for downloading the data.
# For the time being, relevant files should be downloaded to the directory here
# https://visualgenome.org/VGViz/explore


def build_visual_genome_dps(
    dataset_dir: str, write: bool = False
) -> Dict[str, mk.DataPanel]:
    dps = {}
    print("Loading objects and attributes...")
    dps.update(_build_object_dps(dataset_dir))

    print("Loading images...")
    dps.update(_build_image_dp(dataset_dir=dataset_dir))

    print("Loading relationships...")
    dps.update(_build_relationships_dp(dataset_dir=dataset_dir))

    if write:
        write_visual_genome_dps(dps, dataset_dir=dataset_dir)
    return dps


def read_visual_genome_dps(dataset_dir: str) -> Dict[str, mk.DataPanel]:
    return {
        key: mk.DataPanel.read(os.path.join(dataset_dir, f"{key}.mk"))
        for key in ["attributes", "relationships", "objects", "images"]
    }


def write_visual_genome_dps(dps: Mapping[str, mk.DataPanel], dataset_dir: str):
    for key, dp in dps.items():
        dp.write(os.path.join(dataset_dir, f"{key}.mk"))


def _build_object_dps(dataset_dir: str):
    with open(os.path.join(dataset_dir, "attributes.json")) as f:
        objects = json.load(f)

    objects_dp = []  # create one table for objects
    attributes_dp = []  # create one table of attributes
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
                    attributes_dp.append(
                        {"object_id": obj["object_id"], "attribute": attribute}
                    )

            # the vast majority of objects (99.7%) have 0 or 1 synonym in their
            # synset, so we only consider the first synonym to keep things simple
            synset = obj.pop("synsets")
            obj["syn_name"] = synset[0] if len(synset) > 0 else ""

            objects_dp.append(obj)

    return {
        "objects": mk.DataPanel(objects_dp),
        "attributes": mk.DataPanel(attributes_dp),
    }


def _build_image_dp(dataset_dir: str):
    with open(os.path.join(dataset_dir, "image_data.json")) as f:
        images = json.load(f)
    image_dp = mk.DataPanel(images)
    image_dp.remove_column("coco_id")
    image_dp.remove_column("flickr_id")

    image_dp["local_path"] = dataset_dir + (image_dp["url"].str.split("rak248")).apply(
        lambda x: x[-1]
    )

    image_dp["image"] = mk.ImageColumn(image_dp["local_path"])

    return {"images": image_dp}


def _build_relationships_dp(dataset_dir: str):

    with open(os.path.join(dataset_dir, "relationships.json")) as f:
        relationships = json.load(f)

    rel_dp = []
    for image in relationships:
        image_id = image["image_id"]
        for r in image["relationships"]:
            object_synset = r["object"]["synsets"]
            subject_synset = r["subject"]["synsets"]

            rel_dp.append(
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
    rel_dp = mk.DataPanel(rel_dp)
    return {"relationships": rel_dp}
