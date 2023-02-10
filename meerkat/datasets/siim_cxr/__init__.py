import os
import subprocess
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image

from meerkat.cells.volume import MedicalVolumeCell
from meerkat.tools.lazy_loader import LazyLoader

transforms = LazyLoader("torchvision.transforms")


GAZE_DATA_URL = "https://raw.githubusercontent.com/robustness-gym/meerkat/dev/examples/03-med_img/cxr_gaze_data.json"  # noqa: E501


def download_siim_cxr(
    dataset_dir: str,
    kaggle_username: str,
    kaggle_key: str,
    download_gaze_data: bool = True,
    include_mock_reports: bool = True,
):
    """Download the dataset from the SIIM-ACR Pneumothorax Segmentation
    challenge. https://www.kaggle.com/c/siim-acr-pneumothorax-
    segmentation/data.

    Args:
        dataset_dir (str): Path to directory where the dataset will be downloaded.
        kaggle_username (str): Your kaggle username.
        kaggle_key (str): A kaggle API key. In order to use the Kaggle’s public API, you
            must first authenticate using an API token. From the site header, click on
            your user profile picture, then on “My Account” from the dropdown menu. This
            will take you to your account settings at https://www.kaggle.com/account.
            Scroll down to the section of the page labelled API: To create a new token,
            click on the “Create New API Token” button. This will download a json file
            with a "username" and "key" field. Copy and paste the "key" field and pass
            it in as `kaggle_key`.
            Instructions copied from Kaggle API docs: https://www.kaggle.com/docs/api
        download_gaze_data (str): Download a pkl file containing eye-tracking data
            collected on a radiologist interpreting the xray.
    """
    # download and integrate gaze data
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "seesee/siim-train-test",
            "-p",
            dataset_dir,
        ]
    )
    if os.path.exists(os.path.join(dataset_dir, "siim-train-test.zip")):
        subprocess.run(
            [
                "unzip",
                "-q",
                os.path.join(dataset_dir, "siim-train-test.zip"),
                "-d",
                dataset_dir,
            ]
        )
        os.remove(os.path.join(dataset_dir, "siim-train-test.zip"))

    # get segment annotations
    segment_df = pd.read_csv(os.path.join(dataset_dir, "siim", "train-rle.csv"))
    segment_df = segment_df.rename(
        columns={"ImageId": "image_id", " EncodedPixels": "encoded_pixels"}
    )
    # there are some images that were segemented by multiple annotators, we'll just take
    # the first
    segment_df = segment_df[~segment_df.image_id.duplicated(keep="first")]

    # get binary labels for pneumothorax, any row with a "-1" for encoded pixels is
    # considered a negative
    segment_df["pmx"] = (segment_df.encoded_pixels != "-1").astype(int)

    # start building up a main dataframe with a few `merge` operations (i.e. join)
    df = segment_df

    # get filepaths for all images in the "dicom-images-train" directory
    filepaths = sorted(
        glob(os.path.join(dataset_dir, "siim", "dicom-images-train/*/*/*.dcm"))
    )
    filepath_df = pd.DataFrame(
        [
            {
                "filepath": filepath,
                "image_id": os.path.splitext(os.path.basename(filepath))[0],
            }
            for filepath in filepaths
        ]
    )

    # important to perform a left join here, because there are some images in the
    # directory without labels in `segment_df` and we only want those with labelsy
    df = df.merge(filepath_df, how="left", on="image_id")

    if download_gaze_data:
        subprocess.run(
            [
                "curl",
                GAZE_DATA_URL,
                "--output",
                os.path.join(dataset_dir, "cxr_gaze_data.json"),
            ]
        )

    if include_mock_reports:
        df["report"] = (df["pmx"] == 1).apply(_get_mock_report)

    df.to_csv(os.path.join(dataset_dir, "siim_cxr.csv"), index=False)


CXR_MEAN = 0.48865
CXR_STD = 0.24621
CXR_SIZE = 224


def cxr_transform_pil(volume: MedicalVolumeCell):
    array = volume._volume.squeeze()
    return Image.fromarray(np.uint8(array))


def cxr_transform(volume: MedicalVolumeCell):
    img = cxr_transform_pil(volume)
    img = transforms.Compose(
        [
            transforms.Resize([CXR_SIZE, CXR_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(CXR_MEAN, CXR_STD),
        ]
    )(img)
    return img.repeat([3, 1, 1])


def _get_mock_report(pmx: bool):
    state = (np.random.choice(["severe", "moderate"])) if pmx else "no"
    return np.random.choice(
        [
            (
                "Cardiac size cannot be evaluated. Large left pleural effusion is new. "
                "Small right effusion is new. The upper lungs are clear. Right lower "
                f" lobe opacities are better seen in prior CT. There is {state} "
                " pneumothorax. There are mild degenerative changes in the thoracic "
                "spine."
            ),
            (
                f"There is {state} pneumothorax. There are mild degenerative changes "
                "in the thoracic spine. The upper lungs are clear. Right lower lobe "
                "opacities are better seen in prior CT."
                "There are mild degenerative changes in the thoracic spine."
            ),
            (
                "The upper lungs are clear. Right lower lobe opacities are better "
                f"seen in prior CT. There is {state} pneumothorax. "
                "There are mild degenerative changes in the thoracic spine."
            ),
        ]
    )
