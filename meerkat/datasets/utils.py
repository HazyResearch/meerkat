import os


def download_url(url: str, dataset_dir: str, force: bool = False):
    os.makedirs(dataset_dir, exist_ok=True)

    from datasets.utils.file_utils import get_from_cache

    return get_from_cache(
        url, 
        cache_dir=os.path.join(dataset_dir, "downloads"),
        force_download=force
    )


def extract(input_path: str, output_path: str):
    from datasets.utils.extract import Extractor

    # need to check because extractor will rmtree 
    if os.path.exists(output_path):
        raise ValueError("Cannot extract, output path already exists.")

    ex = Extractor()
    return ex.extract(input_path, output_path)
