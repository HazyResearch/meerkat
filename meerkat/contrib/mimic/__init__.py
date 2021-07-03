from __future__ import annotations

import io
import os
from typing import Sequence

import google.auth
import pandas as pd
from google.cloud import bigquery, bigquery_storage, storage
from PIL import Image

import meerkat as mk

MODULES = ["cxr", "core", "hosp", "icu", "ed"]

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


class GCSLoader:
    def __init__(
        self,
        bucket_name: str,
        project: str,
        loader: callable = None,
        dataset_dir: str = None,
        writer: callable = None,
    ):
        storage_client = storage.Client(project=project)
        self.bucket = storage_client.bucket(bucket_name, user_project=project)
        self.loader = (lambda x: x) if loader is None else loader
        self.dataset_dir = dataset_dir
        self.writer = writer

    def __call__(self, blob_name):
        out = self.loader(
            io.BytesIO(self.bucket.blob(str(blob_name)).download_as_bytes())
        )
        if self.writer is not None:
            path = os.path.join(self.dataset_dir, str(blob_name))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.writer(path, out)


class GCSImageColumn(mk.ImageColumn):
    def __init__(
        self,
        bucket_name: str = None,
        project: str = None,
        filepaths: Sequence[str] = None,
        transform: callable = None,
        loader: callable = None,
        writer: callable = None,
        local_dir: str = None,
        *args,
        **kwargs,
    ):
        super(GCSImageColumn, self).__init__(
            filepaths, transform, loader, *args, **kwargs
        )
        self.project = project
        self.bucket_name = bucket_name
        storage_client = storage.Client(project=project)

        self.bucket = storage_client.bucket(bucket_name, user_project=project)
        self.loader = (lambda x: x) if loader is None else loader
        self.local_dir = local_dir

    def _get_cell(self, index: int, materialize: bool = True):
        if materialize:
            blob_name = self._data.iloc[index]

            if self.local_dir is not None and os.path.exists(
                os.path.join(self.local_dir, str(blob_name))
            ):
                # fetch locally if it's been cached locally
                cell = mk.ImagePath(
                    self._data.iloc[index], loader=self.loader, transform=self.transform
                )
                return cell.get()

            # otherwise pull from GCP
            out = self.loader(
                io.BytesIO(self.bucket.blob(str(blob_name)).download_as_bytes())
            )

            if self.writer is not None and self.local_dir is not None:
                # cache locally if writer and local dir are both provided
                path = os.path.join(self.local_dir, str(blob_name))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.writer(path, out)
            return out

        else:
            cell = mk.ImagePath(
                self._data.iloc[index], loader=self.loader, transform=self.transform
            )
            return cell

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return super()._state_keys() | {"bucket_name", "project"}

    @staticmethod
    def concat(columns: Sequence[GCSImageColumn]):
        loader, transform = (
            columns[0].loader,
            columns[0].transform,
        )

        return GCSImageColumn(
            filepaths=pd.concat([c.data for c in columns]),
            loader=loader,
            transform=transform,
            bucket_name=bucket_name,
            project=self.project,
        )


def build_mimic_dp(
    dataset_dir: str,
    gcp_project: str,
    include_labels: bool = True,
    include_dicom_meta: bool = True,
    include_patient: bool = True,
    include_admit: bool = True,
):
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project

    query_str = f"""
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
        query_str += f"""
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

    loader = GCSLoader(
        bucket_name="mimic-cxr-jpg-2.0.0.physionet.org",
        project=gcp_project,
        loader=Image.open,
        writer=lambda x, y: y.save(x),
        dataset_dir=dataset_dir,
    )

    dp = mk.DataPanel.from_pandas(df)
    paths = pd.Series(dp["path"].data)
    dp["jpg_path"] = paths.str.split(".").str[0] + ".jpg"
    dp["img"] = mk.ImageColumn.from_filepaths(dp["jpg_path"], loader=loader)
    return dp


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
