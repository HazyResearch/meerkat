# Meerkat + MIMIC

MIMIC (Medical Information Mart for Intensive Care) is a large database of deidentified EHR and medical imaging data. For more information, see the [MIMIC documentation](https://mimic.mit.edu/docs/about/). This Meerkat integration enables quick exploration the MIMIC database via the `DataPanel`. 

## Prerequisites

1. Access to MIMIC is limited to credentialed users. Researchers seeking to use the database must follow the instructions [here](https://mimic.mit.edu/docs/gettingstarted/).

2. This integration accesses MIMIC data through the [Google Cloud Platform](https://cloud.google.com/gcp) API (via BigQuery and Google Cloud Storage). You must have:

    - A valid Google account. This should be the same email address as your credentialed user account on [physionet](https://physionet.org/content/mimic-cxr/2.0.0/) (prerequisite 1). 
    - A GCP project that can be billed for data transfer. MIMIC data is "Requester Pays" on GCP. 


## Getting Started

1. Login to gcloud from the command line
    ```
    gcloud auth application-default login
    ```
    Note: if you forget to login, you'll get an error looking like: 

        ```
        Forbidden: 403 Access Denied: Table physionet-data:mimic_core.patients: User does not have permission to query table physionet-data:mimic_core.patients.
        ```

