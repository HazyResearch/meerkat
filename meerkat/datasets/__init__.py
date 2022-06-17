from multiprocessing.sharedctypes import Value
from typing import Dict, Union

from meerkat.datapanel import DataPanel

from .registry import Registry

datasets = Registry("datasets")
datasets.__doc__ = """Registry for datasets in meerkat"""

DOWNLOAD_MODES = ["force", "reuse", "skip"]
REGISTRIES = ["meerkat", "huggingface"]


def get(
    name: str,
    dataset_dir: str = None,
    revision: str = None,
    download_mode: str = "reuse",
    registry: str = None,
    **kwargs,
) -> Union[DataPanel, Dict[str, DataPanel]]:
    """
    Load a dataset into .

    Args:
        name (str): Name of the dataset.
        dataset_dir (str): The directory containing dataset data. Defaults to
            `~/.meerkat/datasets/{name}`.
        revision (str): The revision of the dataset. Defaults to `latest`.
        download_mode (str): The download mode. Options are: "reuse" (default) will
            download the dataset if it does not exist, "force" will download the dataset
            even if it exists, and "skip" will not download the dataset if it doesn't
            yet exist. Defaults to `reuse`.
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
            dataset = datasets.get(
                name=name,
                dataset_dir=dataset_dir,
                revision=revision,
                download_mode=download_mode,
                **kwargs,
            )
            try:
                dataset = datasets.get(
                    name=name,
                    dataset_dir=dataset_dir,
                    revision=revision,
                    download_mode=download_mode,
                    **kwargs,
                )
            except:
                pass
            else:
                return dataset

        elif registry == "huggingface":
            try:
                import datasets as hf_datasets

                mapping = {
                    "force": hf_datasets.DownloadMode.FORCE_REDOWNLOAD,
                    "reuse": hf_datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                }
                if download_mode == "skip":
                    raise ValueError(
                        "Download mode `skip` is not supported for HuggingFace datasets."
                    )
                dataset = DataPanel.from_huggingface(
                    name=name,
                    download_mode=mapping[download_mode],
                    revision=revision,
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
    raise ValueError("pass")


@datasets.register()
def cifar10(dataset_dir: str = None, download: bool = True, **kwargs):
    """[summary]"""
    from .torchvision import get_cifar10

    return get_cifar10(download_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def imagenet(dataset_dir: str = None, download: bool = True, **kwargs):
    from .imagenet import build_imagenet_dps

    return build_imagenet_dps(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def imagenette(dataset_dir: str = None, download: bool = True, **kwargs):
    from .imagenette import build_imagenette_dp

    return build_imagenette_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def celeba(dataset_dir: str = None, download: bool = True, **kwargs):
    from .celeba import get_celeba

    return get_celeba(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def coco(dataset_dir: str = None, download: bool = True, **kwargs):
    """Common objects in context. 2014.

    [1] https://cocodataset.org/#download
    """
    from .coco import build_coco_2014_dp

    return build_coco_2014_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def pascal(dataset_dir: str = None, download: bool = True, **kwargs):
    """Pascal Visual Object Classes 2012 dataset.

    [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
    """
    from .pascal import build_pascal_2012_dp

    return build_pascal_2012_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def mir(dataset_dir: str = None, download: bool = True, **kwargs):
    """MIRFLICKR Retrieval Evaluation Dataset [1]_

    [1] https://press.liacs.nl/mirflickr/
    """
    from .mir import build_mirflickr_25k_dp

    return build_mirflickr_25k_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def inaturalist(dataset_dir: str = None, download: bool = True, **kwargs) -> DataPanel:
    """iNaturalist 2021 Dataset [1]_

    Columns:
        - ``image`` (``ImageColumn``): The image
        - ``image_id`` (``SeriesColumn``): Unique image id
        - ``date`` (``SeriesColumn``): The time at which the photo has taken.
        - ``latitude`` (``SeriesColumn``): Latitude at which the photo was taken
        - ``longitude`` (``SeriesColumn``): Longitude at which the photo was taken
        - ``location_uncertainty`` (``SeriesColumn``): Uncertainty in the location
        - ``license`` (``SeriesColumn``): License of the photo
        - ``rights_holder`` (``SeriesColumn``): Rights holder of the photo
        - ``width`` (``SeriesColumn``): Width of the image
        - ``height`` (``SeriesColumn``): Height of the image
        - ``file_name`` (``SeriesColumn``): Filepath relative to ``dataset_dir`` where
          the image is stored.



    [1] https://github.com/visipedia/inat_comp/tree/master/2021
    """
    from .inaturalist import build_inaturalist_dp

    return build_inaturalist_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def dew(dataset_dir: str = None, download: bool = True, **kwargs) -> DataPanel:
    """Date Estimation in the Wild Dataset (DEW) [1]_

    Columns:
        - ``image`` (``ImageColumn``):The image
        - ``img_id`` (``SeriesColumn``): Unique Flickr image id in the dataset.
        - ``GT`` (``SeriesColumn``): Ground truth acquisition year
        - ``date_taken`` (``SeriesColumn``): The time at which the photo has taken
          according to Flickr.
        - ``date_granularity`` (``SeriesColumn``): Accuracy to which we know the date to
          be accurate per Flickr https://www.flickr.com/services/api/misc.dates.html
        - ``url`` (``SeriesColumn``): Weblink for the image.
        - ``username`` (``SeriesColumn``): Flickr username of the author
        - ``title`` (``SeriesColumn``): Image title on Flickr
        - ``licence`` (``SeriesColumn``): Image license according to Flickr
        - ``licence_url`` (``SeriesColumn``): Weblink for the license (if available)


    [1] Müller, Eric; Springstein, Matthias; Ewerth, Ralph (2017): Date Estimation in
    the Wild Dataset. Müller, Eric; Springstein, Matthias; Ewerth, Ralph. DOI:
    10.22000/43
    """
    from .dew import build_dew_dp

    return build_dew_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def enron(dataset_dir: str = None, download: bool = True, **kwargs):
    from .enron import build_enron_dp

    return build_enron_dp(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def yesno(dataset_dir: str = None, download: bool = True, **kwargs):
    from .torchaudio import get_yesno

    return get_yesno(dataset_dir=dataset_dir, download=download, **kwargs)


@datasets.register()
def waterbirds(dataset_dir: str = None, download: bool = True, **kwargs):
    from .waterbirds import build_waterbirds_dp

    return build_waterbirds_dp(dataset_dir=dataset_dir, download=download, **kwargs)
