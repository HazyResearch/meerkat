from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_ENV_VARIABLE = "MEERKAT_CONFIG"


@dataclass
class MeerkatConfig:

    display: DisplayConfig
    datasets: DatasetsConfig

    @classmethod
    def from_yaml(cls, path: str = None):
        if path is None:
            path = os.environ.get(
                CONFIG_ENV_VARIABLE,
                os.path.join(os.path.join(Path.home(), ".meerkat"), "config.yaml"),
            )
            if not os.path.exists(path):
                # create empty config
                yaml.dump({"display": {}, "datasets": {}}, open(path, "w"))
        config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)

        return cls(
            display=DisplayConfig(**config.get("display", {})),
            datasets=DatasetsConfig(**config.get("datasets", {})),
        )


@dataclass
class DisplayConfig:
    max_rows: int = 10

    show_images: bool = True
    max_image_height: int = 128
    max_image_width: int = 128

    show_audio: bool = True


@dataclass
class DatasetsConfig:
    root_dir: str = os.path.join(Path.home(), ".meerkat/datasets")


config = MeerkatConfig.from_yaml()
