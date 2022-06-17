# Meerkat + MIMIC

MIMIC (Medical Information Mart for Intensive Care) is a large database of deidentified EHR and medical imaging data. For more information, see the [MIMIC documentation](https://mimic.mit.edu/docs/about/). This Meerkat integration enables quick exploration the MIMIC database via the `DataPanel`. 

## Prerequisites

1. Access to MIMIC is limited to credentialed users. Researchers seeking to use the database must follow the instructions [here](https://mimic.mit.edu/docs/gettingstarted/).
    - Ensure that you have requested Google BigQuery access to **both** [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) by clicking "Request access" at the bottom of both pages ([here](https://physionet.org/content/mimiciv/1.0/) and [here](https://physionet.org/content/mimic-cxr/2.0.0/)).
    - Ensure that you have requested Google Cloud Storage access to [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) by clicking "Request access" at the bottom of the page [here](https://physionet.org/content/mimic-cxr/2.0.0/).

2. This integration accesses MIMIC data through the [Google Cloud Platform](https://cloud.google.com/gcp) API (via BigQuery and Google Cloud Storage). You must have:

    1. A valid Google account. This should be the same email address as your credentialed user account on [physionet](https://physionet.org/content/mimic-cxr/2.0.0/) (prerequisite 1). 
    2. A GCP project that can be billed for data transfer. MIMIC data is "Requester Pays" on GCP. 


## Getting Started

1. Login to gcloud from the command line
    ```
    gcloud auth application-default login
    ```
    Note: if you forget to login, you'll get an error looking like: 

    ```
    Forbidden: 403 Access Denied: Table physionet-data:mimic_core.patients: User does not have permission to query table physionet-data:mimic_core.patients.
    ```

## Build a MIMIC DataPanel: `build_mimic_dp`
This function builds a `DataPanel` for accessing data from the [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/). The MIMIC-CXR database integrates chest X-ray imaging data with structured EHR datafrom Beth Israel Deaconess Medical Center. The full database has an uncompressedsize of over 5 TB. This function quickly builds a `DataPanel` that can be used to explore, slice and download the database. Building the DataPanel takes ~1 minute (when not downloading the radiology reports). The large CXR DICOM and JPEG files are not downloaded, but lazily pulled from Google Cloud Storage (GCS) only when they are accessed. This makes it possible to inspect and explore that data without downloading the full 5 TB. 

Note: model training will likely bottleneck on the GCS downloads, so it is recommended that you cache the JPEG images locally bfore training. This can be accomplished by passing `download_jpg=True` to `build_mimic_dp`. The images will be saved in `dataset_dir`. This will take several hours for the full dataset. It's recommended that you save resized versions of the images (_e.g._ (512,512)). To do so, pass `download_resize=(512,512)`. 

Each row corresponds to a single chest X-ray image (stored in both DICOM format and JPEG in the MIMIC database). Each row is uniquely identified by the "dicom_id" column. Note that a single chest X-ray study (identified by "study_id" column) mayconsist of multiple images and a single patient (identified by "subject_id" column) may have multiple studies in the database. The columns in the DataPanel can be grouped into four categories:
    
1. (`PandasSeriesColumn`) Metadata and labels pulled from tables in the MIMIC-IV EHR database (e.g."pneumonia", "ethnicity", "view_position", "gender"). For more information on the tables see: https://mimic.mit.edu/docs/iv/modules/.For more information on the CheXpert labels see: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    
2. (`GCSImageColumn`) DICOM and JPEG image files of the chest xrays.These columns do not hold the images themselves, but lazily load from the GCP when they're accessed. 
    
3. (`ReportColumn`) The radiology reports for each exam are downloaded to disk,and lazily loaded when accessed.
