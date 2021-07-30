import io
import os
import subprocess
from typing import Sequence

import google.auth
import numpy as np
import pandas as pd
from google.cloud import bigquery, bigquery_storage, storage
from PIL import Image
from pydicom.filereader import dcmread

import meerkat as mk

from .gcs import GCSImageColumn


def query_mimic(query_str: str, gcp_project: str) -> pd.DataFrame:
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
        .to_dataframe(bqstorage_client=bqstorageclient)
    )

    return df


# we only include a subset of the fields in the dicom metadata
DICOM_FIELDS = [
    "dicom",
    "StudyDate",
    "ImageType",
    "TableType",
    "DistanceSourceToDetector",
    "DistanceSourceToPatient",
    "Exposure",
    "ExposureTime",
    "XRayTubeCurrent",
    "FieldOfViewRotation",
    "FieldOfViewOrigin",
    "FieldOfViewHorizontalFlip",
    "ViewPosition",
    "PatientOrientation",
    "BurnedInAnnotation",
    "RequestingService",
    "DetectorPrimaryAngle",
    "DetectorElementPhysicalSize",
]


def build_mimic_dp(
    dataset_dir: str,
    gcp_project: str,
    include_cxr_jpg: bool = True,
    include_cxr_dicom: bool = True,
    include_labels: bool = True,
    include_dicom_meta: bool = True,
    include_patient: bool = True,
    include_admit: bool = True,
    include_split: bool = True,
):
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project

    query_str = """
            SELECT *
            FROM `physionet-data.mimic_cxr.record_list` cxr_records
            LEFT JOIN `physionet-data.mimic_cxr.study_list` cxr_studies
                ON cxr_records.study_id = cxr_studies.study_id
    """

    if include_labels:
        query_str += """
            LEFT JOIN `physionet-data.mimic_cxr.chexpert` chexpert
                ON cxr_records.study_id = chexpert.study_id
        """

    if include_patient:
        query_str += """
            LEFT JOIN `physionet-data.mimic_core.patients` patients
                ON cxr_records.subject_id = patients.subject_id
        """

    if include_dicom_meta:
        query_str += f"""
            LEFT JOIN (SELECT {','.join(DICOM_FIELDS)}
                FROM `physionet-data.mimic_cxr.dicom_metadata_string`) AS meta
            ON cxr_records.dicom_id = meta.dicom
        """
    elif include_admit:
        # need the StudyDate to include admission data
        query_str += """
            LEFT JOIN (SELECT StudyDate
                FROM `physionet-data.mimic_cxr.dicom_metadata_string`) AS meta
            ON cxr_records.dicom_id = meta.dicom
        """

    df = query_mimic(query_str, gcp_project=gcp_project)

    if include_admit:
        # joining in admissions data is more complicated because the study_list table
        # does not include an index into the admissions table
        # instead we match each study to the first admission for which the study date
        # falls between the admission and discharge dates
        admit_df = query_mimic(
            query_str="""
                SELECT *
                FROM `physionet-data.mimic_core.admissions`
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
            admit_df.drop_duplicates(subset="study_id"), how="left", on="study_id"
        )

    dp = mk.DataPanel.from_pandas(df)
    if include_cxr_jpg:
        paths = pd.Series(dp["path"].data)
        dp["jpg_path"] = paths.str.split(".").str[0] + ".jpg"
        dp["img"] = GCSImageColumn.from_filepaths(
            filepaths=dp["jpg_path"],
            bucket_name="mimic-cxr-jpg-2.0.0.physionet.org",
            project=gcp_project,
            loader=Image.open,
            local_dir=dataset_dir,
        )

    if include_cxr_dicom:
        dp["dicom"] = GCSImageColumn.from_filepaths(
            filepaths=dp["path"],
            bucket_name="mimic-cxr-2.0.0.physionet.org",
            project=gcp_project,
            loader=dcmread,
            local_dir=dataset_dir,
        )

    if include_split:
        storage_client = storage.Client(project=gcp_project)
        bucket = storage_client.bucket(
            "mimic-cxr-jpg-2.0.0.physionet.org", user_project=gcp_project
        )
        filepath = os.path.join(dataset_dir, "mimic-cxr-2.0.0-split.csv.gz")
        bucket.blob("mimic-cxr-2.0.0-split.csv.gz").download_to_filename(filepath)
        subprocess.run(["gunzip", filepath])

        dp = dp.merge(
            mk.DataPanel.from_csv(
                os.path.join(dataset_dir, "mimic-cxr-2.0.0-split.csv")
            )[["split", "dicom_id"]],
            how="left",
            on="dicom_id",
        )
    return dp
