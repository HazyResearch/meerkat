from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Mapping

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
    images: mk.DataPanel
    annotations: mk.DataPanel
    categories: mk.DataPanel

    @property
    def tags(self) -> mk.DataPanel:
        tags = self.annotations["image_id", "category_id"].to_pandas().drop_duplicates()
        tags = mk.DataPanel.from_pandas(tags).merge(
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

        image_dps = []
        annot_dps = []
        cat_dps = []
        for split in ["train", "val"]:
            dct = json.load(
                open(
                    os.path.join(self.dataset_dir, f"lvis_v1_{split}.json"),
                    "rb",
                )
            )

            image_dp = mk.DataPanel(dct["images"])
            image_dp["split"] = [split] * len(image_dp)
            image_dps.append(image_dp)

            annot_dp = mk.DataPanel(dct["annotations"])
            if not self.include_segmentations:
                annot_dp.remove_column("segmentation")
            annot_dp["bbox"] = np.array(annot_dp["bbox"])
            annot_dps.append(annot_dp)

            cat_dps.append(mk.DataPanel(dct["categories"]))

        image_dp = mk.concat(image_dps, axis=0)

        # need to merge in image column from 2014 COCO because splits are not the same
        # with 2017. Eventually, we should just require that people download 2017.
        coco_dp = mk.get("coco", version="2014")
        image_dp = image_dp.merge(coco_dp["id", "image"], on="id")
        image_dp["image"].base_dir = self.var_dataset_dir
        image_dp.data.reorder(
            ["id", "image"] + [c for c in image_dp.columns if c not in ["id", "image"]]
        )

        annot_dp = mk.concat(annot_dps, axis=0)

        cat_dp = mk.concat(cat_dps, axis=0)
        cat_dp = cat_dps[0]
        cat_dp["val_image_count"] = cat_dps[1]["image_count"]
        cat_dp["val_instance_count"] = cat_dps[1]["instance_count"]
        cat_dp["train_image_count"] = cat_dps[0]["image_count"]
        cat_dp["train_instance_count"] = cat_dps[0]["instance_count"]
        cat_dp.remove_column("image_count")
        cat_dp.remove_column("instance_count")

        out = LVISDataset(
            images=image_dp,
            annotations=annot_dp,
            categories=cat_dp,
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
            out[k] = mk.DataPanel.read(path)

        return LVISDataset(**out)

    def _write_build_cache(self, out: LVISDataset):
        for k in ["images", "annotations", "categories"]:
            dp = getattr(out, k)
            dp.write(self._build_cache_path(k))

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
