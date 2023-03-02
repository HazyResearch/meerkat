from typing import Dict, List, Union

from meerkat.dataframe import DataFrame

from .celeba import celeba
from .coco import coco
from .expw import expw
from .fer import fer
from .imagenet import imagenet
from .imagenette import imagenette
from .lvis import lvis
from .mimic_iii import mimic_iii
from .mirflickr import mirflickr
from .ngoa import ngoa
from .pascal import pascal
from .registry import datasets
from .rfw import rfw

__all__ = [
    "celeba",
    "imagenet",
    "imagenette",
    "mirflickr",
    "pascal",
    "lvis",
    "mimic_iii",
    "expw",
    "fer",
    "rfw",
    "ngoa",
    "coco",
]

DOWNLOAD_MODES = ["force", "extract", "reuse", "skip"]
REGISTRIES = ["meerkat", "huggingface"]


def get(
    name: str,
    dataset_dir: str = None,
    version: str = None,
    download_mode: str = "reuse",
    registry: str = None,
    **kwargs,
) -> Union[DataFrame, Dict[str, DataFrame]]:
    """Load a dataset into .

    Args:
        name (str): Name of the dataset.
        dataset_dir (str): The directory containing dataset data. Defaults to
            `~/.meerkat/datasets/{name}`.
        version (str): The version of the dataset. Defaults to `latest`.
        download_mode (str): The download mode. Options are: "reuse" (default) will
            download the dataset if it does not exist, "force" will download the dataset
            even if it exists, "extract" will reuse any downloaded archives but
            force extracting those archives, and "skip" will not download the dataset
            if it doesn't yet exist. Defaults to `reuse`.
        registry (str): The registry to use. If None, then checks each
            supported registry in turn. Currently, supported registries
            include `meerkat` and `huggingface`.
        **kwargs: Additional arguments passed to the dataset.
    """
    if download_mode not in DOWNLOAD_MODES:
        raise ValueError(
            f"Invalid download mode: {download_mode}."
            f"Must pass one of {DOWNLOAD_MODES}"
        )

    if registry is None:
        registry_order = REGISTRIES
    else:
        registry_order = [registry]

    errors = []
    for registry in registry_order:
        if registry == "meerkat":
            try:
                dataset = datasets.get(
                    name=name,
                    dataset_dir=dataset_dir,
                    version=version,
                    download_mode=download_mode,
                    **kwargs,
                )
                return dataset
            except KeyError as e:
                errors.append(e)

        elif registry == "huggingface":
            try:
                import datasets as hf_datasets

                mapping = {
                    "force": hf_datasets.DownloadMode.FORCE_REDOWNLOAD,
                    "reuse": hf_datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                }
                if download_mode == "skip":
                    raise ValueError(
                        "Download mode `skip` isn't supported for HuggingFace datasets."
                    )

                # Add version argument if specified
                if version is not None:
                    kwargs["name"] = version
                dataset = DataFrame.from_huggingface(
                    path=name,
                    download_mode=mapping[download_mode],
                    cache_dir=dataset_dir,
                    **kwargs,
                )
            except FileNotFoundError as e:
                errors.append(e)
            else:
                return dataset
        else:
            raise ValueError(
                f"Invalid registry: {registry}. Must be one of {REGISTRIES}"
            )
    raise ValueError(
        f"No dataset '{name}' found in registry. Errors:" + " ".join(errors)
    )


def versions(name: str) -> List[str]:
    """Get the versions of a dataset. These can be passed to the ``version``
    argument of the :func:`~meerkat.get` function.

    Args:
        name (str): Name of the dataset.

    Returns:
        List[str]: List of versions.
    """
    return datasets.get_obj(name).VERSIONS
