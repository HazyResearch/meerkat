import logging
import os
import pickle
from functools import partial

from tqdm import tqdm

import meerkat as mk

from .data_utils import (
    compute_file_tuples,
    compute_slice_matrix,
    compute_stanford_file_tuples,
    get_sz_labels,
    stanford_eeg_loader,
)

logger = logging.getLogger(__name__)


def build_eeg_df(
    dataset_dir: str,
    raw_dataset_dir: str,
    splits=["train", "dev"],
    clip_len: int = 60,
    step_size: int = 1,
    stride: int = 60,
):
    """Builds a `DataFrame` for accessing EEG data.

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

            sequence_sz, binary_sz = get_sz_labels(
                edf_fn=filepath,
                clip_idx=int(clip_idx),
                time_step_size=step_size,
                clip_len=clip_len,
                stride=stride,
            )

            row_df = {
                "filepath": filepath,
                "file_id": file_id,
                "sequence_sz": sequence_sz,
                "binary_sz": binary_sz,
                "clip_idx": int(clip_idx),
                "h5_fn": os.path.join(dataset_dir, edf_fn.split(".edf")[0] + ".h5"),
                "split": split,
            }

            data.append(row_df)

    df = mk.DataFrame(data)

    eeg_loader = partial(
        compute_slice_matrix, time_step_size=step_size, clip_len=clip_len, stride=stride
    )

    eeg_input_col = df[["clip_idx", "h5_fn"]].defer(fn=eeg_loader)

    df.add_column(
        "eeg_input",
        eeg_input_col,
        overwrite=True,
    )

    return df


def download_tusz(download_dir, version="1.5.2"):
    """Downloads the EEG Seizure TUH dataset (TUSZ)

    REQUIRED:
        1. Need to first registed at
            https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
        2. run download_tusz from python script or simply run the provided rsync
             command below in your terminal
        3. enter the provided password sent to your email after step (1)

    Args:
        download_dir (str): The directory path to save to.
        version (str, optional): Which version to download
    """

    src_pth = f"nedc@www.isip.piconepress.com:data/tuh_eeg_seizure/v{version}/"
    rsync_command = f"rsync -auxvL {src_pth} {download_dir}"
    print("Executing rsync command")
    os.system(rsync_command)


def build_stanford_eeg_df(
    stanford_dataset_dir: str,
    lpch_dataset_dir: str,
    file_marker_dir: str,
    splits=["train", "dev"],
    reports_pth=None,
    clip_len: int = 60,
):
    """Builds a `DataFrame` for accessing EEG data.

    This is for accessing private stanford data.
    The stanford data is limited to specific researchers on IRB.
    No public directions on how to download them yet.
    Contact ksaab@stanford.edu for more information.

    Args:
        stanford_dataset_dir (str): A local dir where stanford EEG are stored
        lpch_dataset_dir (str): A local dir where the lpch EEG are stored
        file_marker_dir (str): A local dir where file markers are stored
        splits (list[str]): List of splits to load
        reports_pth (str): if not None, will load reports
        clip_len (int): Number of seconds in an EEG clip
    """

    # retrieve file tuples which is a list of
    # (eeg filepath, location of sz or -1 if no sz, split)
    file_tuples = compute_stanford_file_tuples(
        stanford_dataset_dir, lpch_dataset_dir, file_marker_dir, splits
    )
    data = []

    for (filepath, sz_loc, split) in file_tuples:
        row_df = {
            "filepath": filepath,
            "file_id": filepath.split("/")[-1].split(".eeghdf")[0],
            "binary_sz": sz_loc != -1,
            "sz_start_index": sz_loc,
            "split": split,
        }
        data.append(row_df)

    df = mk.DataFrame(data)

    eeg_input_col = df[["sz_start_index", "filepath", "split"]].defer(
        fn=partial(stanford_eeg_loader, clip_len=clip_len)
    )

    df.add_column(
        "eeg_input",
        eeg_input_col,
        overwrite=True,
    )

    if reports_pth:
        raw_reports_pth = os.path.join(reports_pth, "reports_unique_for_hl_mm.txt")
        raw_reports_df = mk.DataFrame.from_csv(raw_reports_pth, sep="\t")

        parsed_reports_pth = os.path.join(reports_pth, "parsed_eeg_notes.dill")
        with open(parsed_reports_pth, "rb") as dill_f:
            parsed_reports = pickle.load(dill_f)

        doc_data = []
        for doc in parsed_reports:
            uuid = doc.doc_id
            mask_id = raw_reports_df["note_uuid"] == uuid
            if mask_id.sum() == 1 and "findings" in doc.sections:
                file_id = raw_reports_df[mask_id]["edf_file_name"][0].split(".edf")[0]
                findings = doc.sections["findings"]["text"]
                row_df = {"file_id": file_id, "findings": findings}
                doc_data.append(row_df)
        reports_df = mk.DataFrame(doc_data)

        df = df.merge(reports_df, how="left", on="file_id")

    return df
