import logging
import os

from tqdm import tqdm

import meerkat as mk

from .data_utils import compute_slice_matrix, compute_file_tuples

logger = logging.getLogger(__name__)


def build_eeg_dp(
    dataset_dir: str,
    raw_dataset_dir: str,
    splits=["train", "dev"],
    clip_len: int = 60,
    step_size: int = 1,
    stride: int = 60,
):
    """
    Builds a `DataPanel` for accessing EEG data.

    Currently only supports TUH dataset for seq-seq prediction.
    Future TODO: integrating stanford dataset with weak seq-seq labels

    Args:
        dataset_dir (str): A local directory where the preprocessed
            (h5) EEG data are stored
        raw_dataset_dir (str): A local directory where the original
            (edf) EEG data are stored
        clip_len (int): Number of seconds in an EEG clip
        step_size (int): Number of seconds in a single 'step'
        stride (int):  Number of seconds in the stride when extracting
            clips from signals
    """

    # retrieve paths of all edf files in the raw_dataset_dir
    edf_files = []
    for path, subdirs, files in os.walk(raw_dataset_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    data = []
    for split in splits:
        file_tuples = compute_file_tuples(
            raw_dataset_dir, dataset_dir, split, clip_len, stride
        )

        for (edf_fn, clip_idx, _) in tqdm(file_tuples, total=len(file_tuples)):
            filepath = [file for file in edf_files if edf_fn in file]
            filepath = filepath[0]
            file_id = edf_fn.split(".edf")[0]

            eeg, sequence_sz, binary_sz = compute_slice_matrix(
                h5_fn=os.path.join(dataset_dir, edf_fn.split(".edf")[0] + ".h5"),
                edf_fn=filepath,
                clip_idx=int(clip_idx),
                time_step_size=step_size,
                clip_len=clip_len,
                stride=stride,
            )

            row_df = {
                "filepath": filepath,
                "file_id": file_id,
                "eeg": eeg,
                "sequence_sz": sequence_sz,
                "binary_sz": binary_sz,
                "split": split,
            }

            data.append(row_df)

    dp = mk.DataPanel(data)

    return dp


def download_tusz(download_dir, version="1.5.2"):
    """
    Downloads the EEG Seizure TUH dataset (TUSZ)

    REQUIRED:
        1. Need to first registed at https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml (very quick)
        2. run download_tusz from python script or simply run the provided rsync command below in your terminal
        3. enter the provided password sent to your email after step (1)

    Args:
        download_dir (str): The directory path to save to.
        version (str, optional): Which version to download
    """

    src_pth = f"nedc@www.isip.piconepress.com:data/tuh_eeg_seizure/v{version}/"
    rsync_command = f"rsync -auxvL {src_pth} {download_dir}"
    print("Executing rsync command")
    os.system(rsync_command)
