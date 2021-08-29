import logging
import os

from tqdm import tqdm

import meerkat as mk

from .data_utils import computeSliceMatrix

logger = logging.getLogger(__name__)


def build_eeg_dp(
    dataset_dir: str,
    raw_dataset_dir: str,
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
    dirname = os.path.dirname(__file__)
    for split in ["train", "dev", "test"]:
        fm_name = split + "_cliplen" + str(clip_len) + "_stride" + str(stride)
        if split == "train":
            fm_name += "_balanced"
        file_marker = os.path.join(
            os.path.join(dirname, "file_markers"),
            fm_name + ".txt",
        )

        with open(file_marker, "r") as f:
            file_str = f.readlines()
        file_tuples = [curr_str.strip("\n").split(",") for curr_str in file_str]

        for (edf_fn, clip_idx, _) in tqdm(file_tuples, total=len(file_tuples)):
            filepath = [file for file in edf_files if edf_fn in file]
            filepath = filepath[0]
            file_id = edf_fn.split(".edf")[0]

            eeg, sequence_sz, binary_sz = computeSliceMatrix(
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
