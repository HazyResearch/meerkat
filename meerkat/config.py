from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_ENV_VARIABLE = "MEERKAT_CONFIG"
DATASETS_ENV_VARIABLE = "MEERKAT_DATASETS"


@dataclass
class MeerkatConfig:
    display: DisplayConfig
    datasets: DatasetsConfig
    system: SystemConfig

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

        config = cls(
            display=DisplayConfig(**config.get("display", {})),
            datasets=DatasetsConfig(**config.get("datasets", {})),
            system=SystemConfig(**config.get("system", {})),
        )
        os.environ[DATASETS_ENV_VARIABLE] = config.datasets.root_dir

        return config


@dataclass
class DisplayConfig:
    max_rows: int = 10

    show_images: bool = True
    max_image_height: int = 128
    max_image_width: int = 128

    show_audio: bool = True


@dataclass
class SystemConfig:
    use_gpu: bool = True
    ssh_identity_file: str = os.path.join(Path.home(), ".meerkat/ssh/id_rsa")


class DatasetsConfig:
    def __init__(self, root_dir: str = None):
        if root_dir is None:
            self.root_dir: str = os.path.join(Path.home(), ".meerkat/datasets")
        else:
            self.root_dir: str = root_dir

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value):
        os.environ[DATASETS_ENV_VARIABLE] = value
        self._root_dir = value


config = MeerkatConfig.from_yaml()
