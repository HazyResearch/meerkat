import logging
import os
import re
import subprocess
from typing import Iterable

import google.auth
import pandas as pd
from google.cloud import bigquery, bigquery_storage, storage
from PIL import Image
from pydicom.filereader import dcmread

import meerkat as mk

from .gcs import GCSImageColumn
from .modules import TABLES
from .reports import ReportColumn

logger = logging.getLogger(__name__)


def build_mimic_df(
    dataset_dir: str,
    gcp_project: str,
    tables: Iterable[str] = None,
    excluded_tables: Iterable[str] = None,
    reports: bool = False,
    cxr_dicom: bool = True,
    cxr_jpg: bool = True,
    split: bool = True,
    download_jpg: bool = False,
    download_resize: int = None,
    write: bool = False,
) -> mk.DataFrame:
    """Builds a `DataFrame` for accessing data from the MIMIC-CXR database
    https://physionet.org/content/mimic-cxr/2.0.0/ The MIMIC-CXR database
    integrates chest X-ray imaging data with structured EHR data from Beth
    Israel Deaconess Medical Center. The full database has an uncompressed size
    of over 5 TB. This function quickly builds a `DataFrame` that can be used
    to explore, slice and download the database. Building the DataFrame takes.

    ~1 minute (when not downloading the radiology reports). The large CXR DICOM
    and JPEG files are not downloaded, but lazily pulled from Google Cloud
    Storage (GCS) only when they are accessed. This makes it possible to
    inspect and explore that data without downloading the full 5 TB.

    Note: model training will likely bottleneck on the GCS downloads, so it is
    recommended that you cache the JPEG images locally bfore training. This can be
    accomplished by setting a `writer` and running a map over the data.
    ```
        df["jpg_img].writer = lambda path, img: x.save(path, img)
        df["jpg_img].map(lambda x: True)
    ```
    The images will be saved in `dataset_dir`. This will take several hours for the full
    dataset. You can also slice down to a subset of the dataset before running the map.

    Each row corresponds to a single chest X-ray image (stored in both DICOM format and
    JPEG in the MIMIC database). Each row is uniquely identified by the "dicom_id"
    column. Note that a single chest X-ray study (identified by "study_id" column) may
    consist of multiple images and a single patient (identified by "subject_id" column)
    may have multiple studies in the database. The columns in the DataFrame can be
    grouped into four categories:
        1. (`PandasSeriesColumn`) Metadata and labels pulled from tables in the MIMIC-IV
            EHR database (e.g."pneumonia", "ethnicity", "view_position", "gender"). For
            more information on the tables see: https://mimic.mit.edu/docs/iv/modules/.
            For more information on the CheXpert labels see:
            https://physionet.org/content/mimic-cxr-jpg/2.0.0/
        2. (`GCSImageColumn`) DICOM and JPEG image files of the chest xrays.These
            columns do not actually hold the images, but instead lazily load from the
            GCS when they're indexed.
        3. (`ReportColumn`) The radiology reports for each exam are downloaded to disk,
            and lazily loaded when accessed.
    The arguments below can be used to specify which of the columns to include in the
    DataFrame.

    Args:
        dataset_dir (str): A local directory in which downloaded data will be cached.
        gcp_project (str): The Google Cloud Platform project that will be billed for
            downloads from the database (MIMIC has requester-pays enabled). If you do
            not have GCP project, see instructions for creating one here:
            https://cloud.google.com/resource-manager/docs/creating-managing-projects
        tables (Iterable[str], optional): A subset of ["patient", "admit", "labels",
            "dicom_meta"] specifying which tables to include in the DataFrame.
            Defaults to None, in which case all of the tables listed in
            "meerkat.contrib.mimic.TABLES" will be included.
        excluded_tables (Iterable[str], optional): A subset of ["patient", "admit",
            "labels", "dicom_meta"] specifying which tables to exclude from the
            DataFrame. Defaults to None, in which case none are excluded.
        reports (bool, optional): Download reports if they aren't already downloaded
            in `dataset_dir` and add a "report" column to the DataFrame. Defaults to
            False.
        cxr_dicom (bool, optional): Add a `GCSImagecolumn` called "cxr_dicom" to the
            DataFrame for the DICOM files for each image. Defaults to True.
        cxr_jpg (bool, optional):  Add a `GCSImagecolumn` called "cxr_jpg" to the
            DataFrame for the JPEG files for each image. Defaults to True.
        split (bool, optional): Add a "split" column with "train", "validate" and "test"
            splits. Defaults to True.
        download_jpg (bool, optional): Download jpegs for all the scans in the dataset
            to `dataset_dir`. Expect this to take several hours. Defaults to False.
        download_resize (bool, optional): Resize the images before saving them to disk.
            Defaults to None, in which case the images are not resized.
        write (bool, optiional): Write the dataframe to the directory.

    Returns:
        DataFrame: The MIMIC `DataFrame` with columns
    """
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project

    tables = set(TABLES.keys() if tables is None else tables)
    if excluded_tables is not None:
        tables -= set(excluded_tables)
    # must include the cxr_records table
    tables |= set(["cxr_records"])

    fields = [
        (
            f"{table}.{field[0]} AS {field[1]}"
            if isinstance(field, tuple)
            else f"{table}.{field}"
        )
        for table in tables
        for field in TABLES[table]["fields"]
        if table != "admit"
    ]
    query_str = f"""
            SELECT {','.join(fields)}
            FROM `physionet-data.mimic_cxr.record_list` cxr_records
            LEFT JOIN `physionet-data.mimic_cxr.study_list` cxr_studies
            ON cxr_records.study_id = cxr_studies.study_id
    """

    if "labels" in tables:
        query_str += """
            LEFT JOIN `physionet-data.mimic_cxr.chexpert` labels
                ON cxr_records.study_id = labels.study_id
        """

    if "patients" in tables:
        query_str += """
            LEFT JOIN `physionet-data.mimic_core.patients` patients
                ON cxr_records.subject_id = patients.subject_id
        """

    if "dicom_meta" in tables:
        query_str += """
            LEFT JOIN `physionet-data.mimic_cxr.dicom_metadata_string` AS dicom_meta
            ON cxr_records.dicom_id = dicom_meta.dicom
        """
    elif "admit" in tables:
        # need the StudyDate to include admission data
        query_str += """
            LEFT JOIN (SELECT StudyDate, dicom
                FROM `physionet-data.mimic_cxr.dicom_metadata_string`) AS meta
            ON cxr_records.dicom_id = meta.dicom
        """

    print(f"Querying MIMIC database: `gcp_project`={gcp_project}, `tables`={tables}.")
    df = query_mimic_db(query_str, gcp_project=gcp_project)

    if "admit" in tables:
        # joining in admissions data is more complicated because the study_list table
        # does not include an index into the admissions table
        # instead we match each study to the first admission for which the study date
        # falls between the admission and discharge dates
        fields = ["admit." + field for field in TABLES["admit"]["fields"]] + [
            "subject_id"
        ]

        admit_df = query_mimic_db(
            query_str=f"""
                SELECT {','.join(fields)}
                FROM `physionet-data.mimic_core.admissions` admit
            """,
            gcp_project=gcp_project,
        )
        admit_df = df[["subject_id", "StudyDate", "study_id"]].merge(
            admit_df, on="subject_id"
        )
        study_date = pd.to_datetime(admit_df["StudyDate"])
        admit_df = admit_df[
            (study_date >= admit_df["admittime"].dt.date)
            & (study_date <= admit_df["dischtime"].dt.date)
        ]
        df = df.merge(
            admit_df.drop(columns=["StudyDate", "subject_id"]).drop_duplicates(
                subset="study_id"
            ),
            how="left",
            on="study_id",
        )

    # convert dicom metadata from str to float
    if "dicom_meta" in tables:
        for field in TABLES["dicom_meta"]["fields"]:
            try:
                df[field] = df[field].astype(float)
            except ValueError:
                # if we can't convert, just keep the field as str
                continue

    # convert to snake case
    df = df.rename(columns=lambda x: re.sub(r"(?<!^)(?<!_)(?=[A-Z])", "_", x).lower())

    print("Preparing DataFrame...")
    df = mk.DataFrame.from_pandas(df)

    # add GCSImageColumn for the jpg version of the xrays
    if cxr_jpg:
        paths = pd.Series(df["dicom_path"].data)
        df["jpg_path"] = paths.str.split(".").str[0] + ".jpg"
        df["cxr_jpg"] = GCSImageColumn.from_blob_names(
            blob_names=df["jpg_path"],
            bucket_name="mimic-cxr-jpg-2.0.0.physionet.org",
            project=gcp_project,
            loader=Image.open,
            local_dir=dataset_dir,
        )

    # add GCSImageColumn for the dicoms
    if cxr_dicom:
        df["cxr_dicom"] = GCSImageColumn.from_blob_names(
            blob_names=df["dicom_path"],
            bucket_name="mimic-cxr-2.0.0.physionet.org",
            project=gcp_project,
            loader=dcmread,
            local_dir=dataset_dir,
        )

    if reports:
        reports_dir = os.path.join(dataset_dir, "mimic-cxr-reports")
        if not os.path.exists(reports_dir):
            # download and unzip reports
            print("Downloading reports...")
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(
                "mimic-cxr-2.0.0.physionet.org", user_project=gcp_project
            )
            filepath = os.path.join(dataset_dir, "mimic-cxr-reports.zip")
            bucket.blob("mimic-cxr-reports.zip").download_to_filename(filepath)
            subprocess.run(
                [
                    "unzip",
                    filepath,
                    "-d",
                    os.path.join(dataset_dir, "mimic-cxr-reports"),
                ]
            )
        df["report"] = ReportColumn.from_filepaths(
            reports_dir + "/" + df["report_path"]
        )

    if split:
        print("Downloading splits...")
        storage_client = storage.Client(project=gcp_project)
        bucket = storage_client.bucket(
            "mimic-cxr-jpg-2.0.0.physionet.org", user_project=gcp_project
        )
        filepath = os.path.join(dataset_dir, "mimic-cxr-2.0.0-split.csv.gz")
        bucket.blob("mimic-cxr-2.0.0-split.csv.gz").download_to_filename(
            filepath,
        )
        subprocess.run(["gunzip", filepath])

        df = df.merge(
            mk.DataFrame.from_csv(
                os.path.join(dataset_dir, "mimic-cxr-2.0.0-split.csv")
            )[["split", "dicom_id"]],
            how="left",
            on="dicom_id",
        )

    if download_jpg:
        df = download_mimic_df(df, resize=download_resize)

    if write:
        df.write(os.path.join(dataset_dir, "mimic.mk"))

    return df


def query_mimic_db(query_str: str, gcp_project: str) -> pd.DataFrame:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # Make clients.
    bqclient = bigquery.Client(
        credentials=credentials,
        project=gcp_project,
    )
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

    df = (
        bqclient.query(query_str)
        .result()
        .to_dataframe(bqstorage_client=bqstorageclient, progress_bar_type="tqdm")
    )

    return df


def download_mimic_df(mimic_df: mk.DataFrame, resize: int = None, **kwargs):
    col = mimic_df["cxr_jpg"].view()
    dataset_dir = col.local_dir
    paths = mimic_df["jpg_path"]
    if resize:
        paths = paths.apply(
            lambda x: os.path.join(
                dataset_dir, os.path.splitext(x)[0] + f"_{resize}" + ".jpg"
            )
        )

    def _write_resized(path, img):
        if resize is not None:
            img.thumbnail((resize, resize))
            root, ext = os.path.splitext(path)
            path = root + f"_{resize}" + ext
        img.save(path)

    col._skip_cache = True
    col.writer = _write_resized
    col.map(
        lambda x: True,
        num_workers=6,
        pbar=True,
    )

    mimic_df[f"cxr_jpg_{resize}"] = mk.ImageColumn.from_filepaths(paths)
    return mimic_df
