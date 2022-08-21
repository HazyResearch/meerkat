# we only include a subset of the fields in the dicom metadata


TABLES = {
    "cxr_records": {
        "table": "physionet-data.mimic_cxr.record_list",
        "fields": [
            "study_id",
            "subject_id",
            "dicom_id",
            # use tuples to specify alias
            ("path", "dicom_path"),
        ],
    },
    "cxr_studies": {
        "table": "physionet-data.mimic_cxr.study_list",
        "fields": [
            ("path", "report_path"),
        ],
    },
    "labels": {
        "table": "physionet-data.mimic_cxr.chexpert",
        "fields": [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged_Cardiomediastinum",
            "Fracture",
            "Lung_Lesion",
            "Lung_Opacity",
            "No_Finding",
            "Pleural_Effusion",
            "Pleural_Other",
            "Pneumonia",
            "Pneumothorax",
            "Support_Devices",
        ],
    },
    "dicom_meta": {
        "table": "physionet-data.mimic_cxr.dicom_metadata_string",
        "fields": [
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
        ],
    },
    "patients": {
        "table": "physionet-data.mimic_core.patients",
        "fields": ["gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    },
    "admit": {
        "table": "physionet-data.mimic_core.admissions",
        "fields": [
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "ethnicity",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ],
    },
}
