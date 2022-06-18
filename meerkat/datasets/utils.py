import os


def download_url(url: str, dataset_dir: str, force: bool = False):
    os.makedirs(dataset_dir, exist_ok=True)

    from datasets.utils.file_utils import get_from_cache

    return get_from_cache(
        url, cache_dir=os.path.join(dataset_dir, "downloads"), force_download=force
    )


def extract(input_path: str, dataset_dir: str):
    from datasets.utils.extract import Extractor
    from datasets.utils.filelock import FileLock
    
    # Prevent parallel extractions
    lock_path = input_path + ".lock"
    with FileLock(lock_path):
        for extractor in Extractor.extractors:
            if extractor.is_extractable(input_path):
                return extractor.extract(input_path, dataset_dir)
        raise ValueError("Extraction method not found for {}".format(input_path))
