import os

import pandas as pd

import meerkat as mk
from meerkat.tools.lazy_loader import LazyLoader
from meerkat.tools.utils import deprecated

from ..abstract import DatasetBuilder
from ..info import DatasetInfo
from ..registry import datasets

torch = LazyLoader("torch")
torchaudio = LazyLoader("torchaudio")


@datasets.register()
class yesno(DatasetBuilder):
    """YESNO dataset.

    Reference:
        https://www.openslr.org/1/
    """

    info = DatasetInfo(
        name="yesno",
        full_name="YesNo",
        description=(
            "This dataset contains 60 .wav files, sampled at 8 kHz. "
            "All were recorded by the same male speaker, in Hebrew. "
            "In each file, the individual says 8 words; each word is either the "
            "Hebrew for 'yes' or 'no', so each file is a random sequence of 8 yes-es "
            "or noes. There is no separate transcription provided; the sequence is "
            "encoded in the filename, with 1 for yes and 0 for no."
        ),
        homepage="https://www.openslr.org/1/",
        tags=["audio", "classification"],
    )

    VERSIONS = ["release1"]

    def download(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        torchaudio.datasets.YESNO(root=self.dataset_dir, download=True)

    def is_downloaded(self) -> bool:
        return super().is_downloaded() and os.path.exists(
            os.path.join(self.dataset_dir, "waves_yesno")
        )

    def build(self):
        dataset = torchaudio.datasets.YESNO(root=self.dataset_dir, download=False)
        df = mk.DataFrame(
            {
                "id": dataset._walker,
                "audio": mk.files(
                    pd.Series(dataset._walker) + ".wav", base_dir=dataset._path
                ),
                "labels": torch.tensor(
                    [[int(c) for c in fileid.split("_")] for fileid in dataset._walker]
                ),
            }
        )

        return df


@deprecated("mk.get('yesno')")
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
