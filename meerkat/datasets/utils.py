import os
import shutil
import tarfile

from meerkat.dataframe import DataFrame


def download_url(url: str, dataset_dir: str, force: bool = False):
    os.makedirs(dataset_dir, exist_ok=True)

    from datasets.utils.file_utils import get_from_cache

    return get_from_cache(
        url, cache_dir=os.path.join(dataset_dir, "downloads"), force_download=force
    )


def extract(path: str, dst: str, extractor: str = None):
    from datasets.utils.extract import Extractor
    from datasets.utils.filelock import FileLock

    # Prevent parallel extractions
    lock_path = path + ".lock"
    with FileLock(lock_path):
        # Support for older versions of datasets that have list of extractors instead
        # of dict of extractors
        extractors = (
            Extractor.extractors
            if isinstance(Extractor.extractors, list)
            else Extractor.extractors.values()
        )
        if extractor:
            return extractor.extract(path, dst)

        for extractor in extractors:
            if extractor.is_extractable(path):
                return extractor.extract(path, dst)
        raise ValueError("Extraction method not found for {}".format(path))


def download_google_drive(
    url: str = None, id: str = None, dst: str = None, is_folder: bool = False
):
    os.makedirs(dst, exist_ok=True)
    if (url is None) == (id is None):
        raise ValueError("Exactly one of url or id must be provided.")

    if dst is None:
        raise ValueError("dst must be provided.")

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "Google Drive download requires gdown. Install with `pip install gdown`."
        )

    if is_folder:
        gdown.download_folder(url=url, id=id, output=dst)
    else:
        gdown.download(url=url, id=id, output=dst)


def download_df(
    url: str, overwrite: bool = False, download_dir: str = None
) -> DataFrame:
    """Download a dataframe from a url.

    Args:
        url: The url.
        overwrite: Whether to download the dataframe from the path again.
        download_dir: The directory to download the dataframe to.

    Returns:
        DataFrame: The downloaded dataframe.
    """
    if download_dir is None:
        download_dir = os.path.abspath(os.path.expanduser("~/.meerkat/dataframes"))

    # A hacky way of getting the name from the url.
    # This won't always work, because we could have name conflicts.
    # TODO: Find a better way to do this.
    local_path = url.split("/")[-1]
    local_path = local_path.split(".zip")[0].split(".tar")[0]
    dir_path = os.path.join(download_dir, local_path)

    mode = "r:gz" if url.endswith(".tar.gz") else "r"

    if overwrite and os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    if not os.path.exists(dir_path):
        cached_tar_path = download_url(
            url=url,
            dataset_dir=download_dir,
        )
        print("Extracting tar archive, this may take a few minutes...")
        extract_tar_file(filepath=cached_tar_path, download_dir=download_dir, mode=mode)
        os.remove(cached_tar_path)

    return DataFrame.read(dir_path)


def extract_tar_file(filepath: str, download_dir: str, mode: str = None):
    if mode is None:
        mode = "r:gz" if filepath.endswith(".tar.gz") else "r"
    tar = tarfile.open(filepath, mode=mode)
    tar.extractall(download_dir)
    tar.close()
