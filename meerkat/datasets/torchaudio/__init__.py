import pandas as pd
import torch
import torchaudio

import meerkat as mk


def get_yesno(dataset_dir: str, download: bool = True):
    """Load YESNO as a Meerkat DataPanel.

    Args:
        download_dir: download directory
        frac_val: fraction of training set to use for validation

    Returns:
        a DataPanel containing columns `raw_image`, `image` and `label`
    """
    if download:
        dataset = torchaudio.datasets.YESNO(root=dataset_dir, download=True)

    dp = mk.DataPanel(
        {
            "id": dataset._walker,
            "audio": mk.AudioColumn(
                pd.Series(dataset._walker) + ".wav", base_dir=dataset._path
            ),
            "labels": torch.tensor(
                [[int(c) for c in fileid.split("_")] for fileid in dataset._walker]
            ),
        }
    )

    return dp
