import os
import pickle
import shutil
import subprocess
from glob import glob

import numpy as np
import pandas as pd
import PIL
from PIL import Image

from meerkat import column, env
from meerkat.cells.volume import MedicalVolumeCell
from meerkat.columns.deferred.file import FileColumn
from meerkat.dataframe import DataFrame
from meerkat.tools.lazy_loader import LazyLoader

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets
from ..utils import download_url, extract

transforms = LazyLoader("torchvision.transforms")

if env.package_available("pydicom"):
    import pydicom
else:
    pydicom = None


GAZE_DATA_URL = "https://raw.githubusercontent.com/robustness-gym/meerkat/dev/examples/03-med_img/cxr_gaze_data.json"  # noqa: E501


@datasets.register()
class siim_cxr(DatasetBuilder):
    """The SIIM-CXR dataset from Kaggle.

    Reference:
        https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data
    """

    VERSIONS = [
        "stage_2",
        "stage_1",
    ]

    info = DatasetInfo(
        name="siim_cxr",
        full_name="SSIM-ACR Pneumothorax Segmentation",
        description=(
            "SSIM CXR is a dataset of chest X-rays of patients with and without "
            "pneumothorax. "
            "This dataset consists of RLE encoded masks for the pneumothorax regions. "
        ),
        homepage="https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data",  # noqa: E501
        tags=["image", "classification", "segmentation"],
    )

    def download(self):
        if self.version == "stage_1":
            self._download_stage_1()
        elif self.version == "stage_2":
            self._download_stage_2()

    def _download_stage_1(self):
        tar_file = os.path.join(self.dataset_dir, "dicom-images-train.tar.gz")
        dirpath = os.path.join(self.dataset_dir, "dicom-images-train")

        if not os.path.exists(tar_file):
            raise ValueError("Please download the stage 1 dataset tar file.")
        if not os.path.exists(dirpath):
            extract(tar_file, self.dataset_dir)
        assert os.path.isdir(dirpath)

        # Download the pneumothorax labels.
        labels = os.path.join(self.dataset_dir, "train-rle.csv")
        if not os.path.exists(labels):
            raise ValueError("Please download the stage 1 labels.")

        # Download the chest tube labels.
        path = download_url(
            "https://github.com/khaledsaab/spatial_specificity/raw/main/cxr_tube_dict.pkl",  # noqa: E501
            self.dataset_dir,
        )  # noqa: E501
        shutil.move(path, os.path.join(self.dataset_dir, "cxr_tube_dict.pkl"))

    def _download_stage_2(self):
        """Download the SIIM CXR dataset from kaggle."""
        if not env.package_available("kaggle"):
            raise ImportError("Please install kaggle using `pip install kaggle`")

        # download and integrate gaze data
        # os.environ["KAGGLE_USERNAME"] = self.kaggle_username
        # os.environ["KAGGLE_KEY"] = self.kaggle_key
        out = subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "siim-acr-pneumothorax-segmentation",
                "-p",
                self.dataset_dir,
            ]
        )
        if out.returncode != 0:
            raise ValueError("Downloading the kaggle dataset failed.")

        expected_zip_file = os.path.join(
            self.dataset_dir, "siim-acr-pneumothorax-segmentation.zip"
        )
        if not os.path.exists(expected_zip_file):
            raise ValueError("Downloaded dataset is not in the expected format.")
        extract(expected_zip_file, self.dataset_dir)

    def build(self):
        if self.version == "stage_1":
            return self._build_stage_1()
        elif self.version == "stage_2":
            return self._build_stage_2()

    def _build_stage_1(self):
        """Build the SIIM CXR dataset (stage 1 version)."""
        # Get filenames.
        dcm_folder = os.path.join(self.dataset_dir, "dicom-images-train")
        _files = _collect_all_dicoms(dcm_folder)
        df = pd.DataFrame({"filename": column(_files)})
        df["img_id"] = df["filename"].map(
            lambda filename: os.path.splitext(os.path.basename(filename))[0]
        )

        # Get pneumothorax labels.
        label_df = self._build_stage_1_labels()

        # important to perform a left join here, because there are some images in the
        # directory without labels in `segment_df` and we only want those with labelsy
        df = df.merge(label_df, how="left", on="img_id")

        df = DataFrame.from_pandas(df, primary_key="img_id").drop("index")

        # Load the data
        df["img"] = FileColumn(
            _files, type="image", loader=_load_siim_cxr, base_dir=dcm_folder
        )
        # df["img_tensor"] = df["img"].defer(cxr_transform)

        # drop nan columns
        df = df[~df["pmx"].isna()]

        return df

    def _build_stage_1_labels(self):
        segment_df = pd.read_csv(os.path.join(self.dataset_dir, "train-rle.csv"))
        segment_df = segment_df.rename(
            columns={"ImageId": "img_id", " EncodedPixels": "encoded_pixels"}
        )
        # there are some images that were segemented by multiple annotators,
        # we'll just take the first
        segment_df = segment_df[~segment_df.img_id.duplicated(keep="first")]

        # get binary labels for pneumothorax, any row with a "-1" for
        # encoded pixels is considered a negative
        segment_df["pmx"] = (segment_df.encoded_pixels != "-1").astype(int)

        # Chest tube labels.
        with open(os.path.join(self.dataset_dir, "cxr_tube_dict.pkl"), "rb") as f:
            tube_dict = pickle.load(f)
        img_id = tube_dict.keys()
        values = [tube_dict[k] for k in img_id]
        tube_df = pd.DataFrame({"img_id": img_id, "tube": values})
        segment_df = segment_df.merge(tube_df, how="left", on="img_id")

        return segment_df[["img_id", "pmx", "tube"]]

    def _build_stage_2(self):
        """Build the SIIM CXR dataset."""
        dcm_folder = os.path.join(self.dataset_dir, "stage_2_images")
        _files = os.listdir(dcm_folder)
        _files = [fname for fname in _files if fname.endswith(".dcm")]
        df = DataFrame({"fname": column(_files)})

        # Load the data
        df["img"] = FileColumn(
            _files, type="image", loader=_load_siim_cxr, base_dir=dcm_folder
        )
        df["img_tensor"] = df["img"].defer(cxr_transform)

        return df


def _collect_all_dicoms(root_dir: str):
    """Return the relative paths for all dicoms in a directory."""
    # TODO: make this work with windows
    remove_str = root_dir
    if remove_str[-1] != "/":
        remove_str += "/"

    relative_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                # Remove the root directory from the file path.
                file_path = file_path.replace(remove_str, "")
                relative_paths.append(file_path)

    return relative_paths


def _load_siim_cxr(filepath) -> PIL.Image:
    """Load a single image from the SIIM-CXR dataset."""
    return PIL.Image.fromarray(pydicom.read_file(filepath).pixel_array)


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
    if not env.package_available("kaggle"):
        raise ImportError("Please install kaggle using `pip install kaggle`")

    # download and integrate gaze data
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    out = subprocess.run(
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
    if out.returncode != 0:
        raise ValueError("Downloading the kaggle dataset failed.")
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
    if isinstance(volume, MedicalVolumeCell):
        img = cxr_transform_pil(volume)
    else:
        img = volume

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
