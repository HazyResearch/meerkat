import os


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
        if extractor:
            return extractor.extract(path, dst)
        for extractor in Extractor.extractors:
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
