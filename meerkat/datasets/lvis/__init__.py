from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

import meerkat as mk

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

IMAGE_URL = "http://images.cocodataset.org/zips/{split}{version}.zip"
# flake8: noqa
LABEL_URL = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_{version}_{split}.json.zip"


@dataclass
class LVISDataset:
    images: mk.DataFrame
    annotations: mk.DataFrame
    categories: mk.DataFrame

    @property
    def tags(self) -> mk.DataFrame:
        tags = self.annotations["image_id", "category_id"].to_pandas().drop_duplicates()
        tags = mk.DataFrame.from_pandas(tags).merge(
            self.categories["id", "synset"], left_on="category_id", right_on="id"
        )
        tags["id"] = (
            tags["image_id"].to_pandas().astype(str)
            + "_"
            + tags["category_id"].astype(str)
        )
        return tags


@datasets.register()
class lvis(DatasetBuilder):
    VERSIONS = ["v1"]

    info = DatasetInfo(
        name="lvis",
        full_name="Common Objects in Context",
        description="A dataset for Large Vocabulary Instance Segmenation (LVIS).",
        homepage="https://www.lvisdataset.org/",
        tags=["image", "object recognition"],
        # flake8: noqa
        citation="@inproceedings{gupta2019lvis,title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},year={2019}}",
    )

    def __init__(
        self,
        dataset_dir: str = None,
        version: str = None,
        download_mode: str = "reuse",
        include_segmentations: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            version=version,
            download_mode=download_mode,
        )
        self.include_segmentations = include_segmentations

    def build(self):
        out = self._read_build_cache()
        if out is not None:
            return out

        image_dfs = []
        annot_dfs = []
        cat_dfs = []
        for split in ["train", "val"]:
            dct = json.load(
                open(
                    os.path.join(self.dataset_dir, f"lvis_v1_{split}.json"),
                    "rb",
                )
            )

            image_df = mk.DataFrame(dct["images"])
            image_df["split"] = [split] * len(image_df)
            image_dfs.append(image_df)

            annot_df = mk.DataFrame(dct["annotations"])
            if not self.include_segmentations:
                annot_df.remove_column("segmentation")
            annot_df["bbox"] = np.array(annot_df["bbox"])
            annot_dfs.append(annot_df)

            cat_dfs.append(mk.DataFrame(dct["categories"]))

        image_df = mk.concat(image_dfs, axis=0)

        # need to merge in image column from 2014 COCO because splits are not the same
        # with 2017. Eventually, we should just require that people download 2017.
        coco_df = mk.get("coco", version="2014")
        image_df = image_df.merge(coco_df["id", "image"], on="id")
        image_df["image"].base_dir = self.var_dataset_dir
        image_df.data.reorder(
            ["id", "image"] + [c for c in image_df.columns if c not in ["id", "image"]]
        )

        annot_df = mk.concat(annot_dfs, axis=0)

        cat_df = mk.concat(cat_dfs, axis=0)
        cat_df = cat_dfs[0]
        cat_df["val_image_count"] = cat_dfs[1]["image_count"]
        cat_df["val_instance_count"] = cat_dfs[1]["instance_count"]
        cat_df["train_image_count"] = cat_dfs[0]["image_count"]
        cat_df["train_instance_count"] = cat_dfs[0]["instance_count"]
        cat_df.remove_column("image_count")
        cat_df.remove_column("instance_count")

        out = LVISDataset(
            images=image_df,
            annotations=annot_df,
            categories=cat_df,
        )

        self._write_build_cache(out)
        return out

    def _build_cache_hash(self):
        return hashlib.sha1(
            json.dumps({"include_segmentations": self.include_segmentations}).encode(
                "utf-8"
            )
        ).hexdigest()[:10]

    def _build_cache_path(self, key: str):
        dir_path = os.path.join(self.dataset_dir, "build_cache")
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"out_{key}_{self._build_cache_hash()}.mk")

    def _read_build_cache(self) -> LVISDataset:
        out = {}
        for k in ["images", "annotations", "categories"]:
            path = self._build_cache_path(k)
            if not os.path.exists(path):
                return None
            out[k] = mk.DataFrame.read(path)

        return LVISDataset(**out)

    def _write_build_cache(self, out: LVISDataset):
        for k in ["images", "annotations", "categories"]:
            df = getattr(out, k)
            df.write(self._build_cache_path(k))

    def download(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        coco_dataset_dir = self._get_dataset_dir(name="coco", version="2014")
        if not os.path.exists(coco_dataset_dir):
            raise ValueError(
                "LVIS depends on COCO 2014, please download it first with "
                "`mk.get('coco', version='2014')`."
            )

        for split in ["train", "val", "test"]:
            lvis_path = os.path.join(self.dataset_dir, f"{split}2014")
            if not (os.path.exists(lvis_path)):
                os.symlink(
                    os.path.join(self._get_dataset_dir("coco", "2014"), f"{split}2014"),
                    lvis_path,
                )
            if split != "test":
                # no test labels
                downloaded_path = download_url(
                    LABEL_URL.format(version=self.version, split=split),
                    self.dataset_dir,
                )
                extract(downloaded_path, self.dataset_dir)
