import pandas as pd

import meerkat as mk
from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")
torchaudio = LazyLoader("torchaudio")


def get_yesno(dataset_dir: str, download: bool = True):
    """Load YESNO as a Meerkat DataFrame.

    Args:
        download_dir: download directory
        frac_val: fraction of training set to use for validation

    Returns:
        a DataFrame containing columns `raw_image`, `image` and `label`
    """
    if download:
        dataset = torchaudio.datasets.YESNO(root=dataset_dir, download=True)

    df = mk.DataFrame(
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

    return df
