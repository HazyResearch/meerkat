import os
import tarfile

import pandas as pd
from torchvision.datasets.utils import download_url

VERSION_TO_URL = {
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    "320px": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "160px": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}

ID_TO_WORDS = {
    "n02979186": "cassette player",
    "n03417042": "garbage truck",
    "n01440764": "tench",
    "n02102040": "english springer spaniel",
    "n03028079": "church",
    "n03888257": "parachute",
    "n03394916": "french horn",
    "n03000684": "chainsaw",
    "n03445777": "golf ball",
    "n03425413": "gas pump",
}

ID_TO_IDX = {
    "n02979186": 482,
    "n03417042": 569,
    "n01440764": 0,
    "n02102040": 217,
    "n03028079": 497,
    "n03888257": 701,
    "n03394916": 566,
    "n03000684": 491,
    "n03445777": 574,
    "n03425413": 571,
}


def download_imagenette(download_dir, version="160px"):
    tar_path = os.path.join(download_dir, os.path.basename(VERSION_TO_URL[version]))
    dir_path = os.path.splitext(tar_path)[0]
    if not os.path.exists(dir_path):
        download_url(
            url=VERSION_TO_URL[version],
            root=download_dir,
        )
        print("Extracting tar archive, this may take a few minutes...")
        tar = tarfile.open(tar_path)
        tar.extractall(download_dir)
        tar.close()
        os.remove(tar_path)
    else:
        print(f"Directory {dir_path} already exists. Skipping download.")

    # build dataframe
    df = pd.read_csv(os.path.join(dir_path, "noisy_imagenette.csv"))
    df["label_id"] = df["noisy_labels_0"]
    df["label"] = df["label_id"].replace(ID_TO_WORDS)
    df["label_idx"] = df["label_id"].replace(ID_TO_IDX)
    df["split"] = df["is_valid"].replace({False: "train", True: "valid"})
    df["img_path"] = df.path.apply(lambda x: os.path.join(dir_path, x))
    df[["img_path", "label", "label_id", "label_idx", "split"]].to_csv(
        os.path.join(dir_path, "imagenette.csv"), index=False
    )
    return dir_path
