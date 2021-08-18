# Meerkat + MetaDataset

[MetaDataset](https://metadataset.readthedocs.io/en/latest/) is a new "dataset of datasets" that can be used for evaluating distribution shifts and model robustness. This Meerkat integration allows users to directly load in MetaDataset into a `DataPanel`.

## Getting Started

1. Follow the [instructions](https://metadataset.readthedocs.io/en/latest/sub_pages/download_metadataset.html) to download the dataset and the necessary code required to process it.
2. Load the MetaDataset DataPanel using
```python
from meerkat.contrib.metadataset import load_metadataset

# The metadataset folder path should contain
# 1. proc/ folder that has the raw data extracted by MetaDataset's helper script
# 2. `train_SceneGraphs.json` file

dp = load_metadataset(path='/my/path/to/metadataset/folder')
```
