import logging
import os
import pickle
from functools import partial

import terra
from tqdm import tqdm
import numpy as np


import meerkat as mk

from .data_utils import (
    compute_stanford_file_tuples,
    compute_streaming_file_tuples,
    eeg_age_loader,
    stanford_eeg_loader,
    streaming_eeg_loader,
    fft_eeg_loader,
    eeg_patientid_loader,
    split_dp,
    merge_in_split,
)

from .data_utils_tuh import (
    compute_file_tuples,
    tuh_eeg_loader,
    get_sz_labels,
    fft_tuh_eeg_loader,
    ss_tuh_eeg_loader,
)

logger = logging.getLogger(__name__)


@terra.Task
def build_tuh_eeg_dp(
    dataset_dir: str,
    raw_dataset_dir: str,
    splits=["train", "dev"],
    clip_len: int = 60,
    offset: int = 0,
    ss_clip_len: int = 0,
    ss_offset: int = 0,
    step_size: int = 1,
    stride: int = 60,
    train_frac: float = 0.9,
    valid_frac: float = 0.1,
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
            raw_dataset_dir,
            dataset_dir,
            split,
            clip_len,
            stride,
            ss_clip_len,
            ss_offset,
        )

        for (edf_fn, paitent_id, clip_idx, _) in tqdm(
            file_tuples, total=len(file_tuples)
        ):
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
                "id": file_id,
                "paitent_id": paitent_id,
                "sequence_sz": sequence_sz,
                "target": binary_sz,
                "clip_idx": int(clip_idx),
                "h5_fn": os.path.join(dataset_dir, edf_fn.split(".edf")[0] + ".h5"),
                "split": "test" if split == "dev" else "train",
                "age": -1,
            }

            data.append(row_df)

    dp = mk.DataPanel(data)

    train_mask = np.array(dp["split"] == "train")
    dp_train = dp.lz[train_mask]
    dp_test = dp.lz[~train_mask]

    dp_train_split = split_dp(
        dp_train,
        split_on="paitent_id",
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=0,
    )
    dp_train = merge_in_split(dp_train, dp_train_split)
    dp = dp_train.append(dp_test)

    eeg_loader = partial(
        tuh_eeg_loader,
        time_step_size=step_size,
        clip_len=clip_len,
        stride=stride,
        offset=offset,
    )

    eeg_input_col = dp[["clip_idx", "h5_fn", "split"]].to_lambda(fn=eeg_loader)

    dp.add_column(
        "input",
        eeg_input_col,
        overwrite=True,
    )

    eeg_fftinput_col = dp[["clip_idx", "h5_fn", "split"]].to_lambda(
        fn=partial(
            fft_tuh_eeg_loader,
            time_step=step_size,
            clip_len=clip_len,
            stride=stride,
            offset=offset,
        )
    )

    dp.add_column(
        "fft_input",
        eeg_fftinput_col,
        overwrite=True,
    )

    if ss_clip_len != 0:
        eeg_ss_output_col = dp[["clip_idx", "h5_fn", "split"]].to_lambda(
            fn=partial(
                ss_tuh_eeg_loader,
                time_step=step_size,
                clip_len=ss_clip_len,
                stride=stride,
                offset=-(clip_len + ss_offset),
            )
        )
        dp.add_column(
            "ss_output",
            eeg_ss_output_col,
            overwrite=True,
        )

    return dp


def download_tusz(download_dir, version="1.5.2"):
    """
    Downloads the EEG Seizure TUH dataset (TUSZ)

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


@terra.Task
def build_stanford_eeg_dp(
    stanford_dataset_dir: str,
    lpch_dataset_dir: str,
    file_marker_dir: str,
    reports_pth=None,
    restrict_to_reports=False,
    clip_len: int = 60,
    offset: int = 0,
    seed: int = 123,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """
    Builds a `DataPanel` for accessing EEG data.

    This is for accessing private stanford data.
    The stanford data is limited to specific researchers on IRB.
    No public directions on how to download them yet.
    Contact ksaab@stanford.edu for more information.

    Args:
        stanford_dataset_dir (str): A local dir where stanford EEG are stored
        lpch_dataset_dir (str): A local dir where the lpch EEG are stored
        file_marker_dir (str): A local dir where file markers are stored
        reports_pth (str): if not None, will load reports
        restrict_to_reports (bool): If true, only considers eegs with report
        clip_len (int): Number of seconds in an EEG clip
    """

    # retrieve file tuples which is a list of
    # (eeg filepath, location of sz or -1 if no sz, split)
    file_tuples = compute_stanford_file_tuples(
        stanford_dataset_dir, lpch_dataset_dir, file_marker_dir, ["train"]
    )
    data = []

    np.random.seed(seed)
    corrupt_files = []

    for (filepath, sz_loc, fm_split) in tqdm(file_tuples, total=len(file_tuples)):
        row_df = {
            "filepath": filepath,
            "file_id": filepath.split("/")[-1].split(".eeghdf")[0],
            "id": filepath.split("/")[-1].split(".eeghdf")[0] + f"_{sz_loc}",
            "target": sz_loc != -1,
            "sz_start_index": sz_loc,
            "fm_split": fm_split,
        }
        data.append(row_df)

        # check to see if can open file
        # try:
        #     row_df_ = row_df.copy()
        #     row_df_["split"] = "train"
        #     eeg_clip = stanford_eeg_loader(row_df_)
        # except:
        #     corrupt_files.append(filepath)

    # print(corrupt_files)
    # breakpoint()

    dp = mk.DataPanel(data)

    patientid_col = dp["filepath"].map(function=eeg_patientid_loader)
    dp.add_column(
        "patient_id",
        patientid_col,
        overwrite=True,
    )

    dp_split = split_dp(
        dp,
        split_on="patient_id",
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
    )
    dp = merge_in_split(dp, dp_split)

    eeg_input_col = dp[["sz_start_index", "filepath", "fm_split", "split"]].to_lambda(
        fn=partial(stanford_eeg_loader, clip_len=clip_len, offset=offset)
    )

    dp.add_column(
        "input",
        eeg_input_col,
        overwrite=True,
    )

    # eeg_fftinput_col = dp[
    #     ["sz_start_index", "filepath", "fm_split", "split"]
    # ].map(fn=partial(fft_eeg_loader, clip_len=clip_len, offset=offset),mmap=True,mmap_path="")
    eeg_fftinput_col = dp[
        ["sz_start_index", "filepath", "fm_split", "split"]
    ].to_lambda(fn=partial(fft_eeg_loader, clip_len=clip_len, offset=offset))

    dp.add_column(
        "fft_input",
        eeg_fftinput_col,
        overwrite=True,
    )

    if reports_pth:
        raw_reports_pth = os.path.join(reports_pth, "reports_unique_for_hl_mm.txt")
        raw_reports_dp = mk.DataPanel.from_csv(raw_reports_pth, sep="\t")

        parsed_reports_pth = os.path.join(reports_pth, "parsed_eeg_notes.dill")
        with open(parsed_reports_pth, "rb") as dill_f:
            parsed_reports = pickle.load(dill_f)

        doc_data = []
        for doc in parsed_reports:
            uuid = doc.doc_id
            mask_id = raw_reports_dp["note_uuid"] == uuid
            if mask_id.sum() == 1 and "findings" in doc.sections:
                file_id = raw_reports_dp[mask_id]["edf_file_name"][0].split(".edf")[0]
                findings = doc.sections["findings"]["text"]
                narrative = "none"
                if "narrative" in doc.sections:
                    narrative = doc.sections["narrative"]["text"]
                mrn = raw_reports_dp[mask_id]["mrn"][0]
                row_df = {
                    "file_id": file_id,
                    "findings": findings,
                    "narrative": narrative,
                    "patient_id": mrn,
                }
                doc_data.append(row_df)
        reports_dp = mk.DataPanel(doc_data)

        if restrict_to_reports:
            dp = dp.merge(reports_dp, how="inner", on="file_id")
        else:
            dp = dp.merge(reports_dp, how="left", on="file_id")

    # Add metadata
    age_col = dp["filepath"].map(function=eeg_age_loader)
    dp.add_column(
        "age",
        age_col,
        overwrite=True,
    )
    # logage_col = dp["filepath"].map(function=eeg_logage_loader)
    # dp.add_column(
    #     "logage", logage_col, overwrite=True,
    # )

    # male_col = dp["filepath"].map(function=eeg_male_loader)
    # dp.add_column(
    #     "male", male_col, overwrite=True,
    # )

    # remove duplicate ID rows
    dp = dp.lz[~dp["id"].duplicated()]

    return dp


@terra.Task
def build_streaming_stanford_eeg_dp(
    stanford_dataset_dir: str,
    lpch_dataset_dir: str,
    annotations_dir: str,
    clip_len: int = 60,
    stride: int = 60,
    sz_label_sensitivity: int = 20,
    seed: int = 123,
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
):
    """
    Builds a `DataPanel` for accessing EEG data in streaming setting.

    This is for accessing private stanford data.
    The stanford data is limited to specific researchers on IRB.
    No public directions on how to download them yet.
    Contact ksaab@stanford.edu for more information.

    Args:
        stanford_dataset_dir (str): A local dir where stanford EEG are stored
        lpch_dataset_dir (str): A local dir where the lpch EEG are stored
        annotations_dir (str): A local dir where the fine grained annotations are stores
        clip_len (int): length of eeg input in seconds
        stride (int): stride for moving window to define clips
        sz_label_sensitivity (int): how many seconds of seizure in the clip to be considered a seizure
    """

    # retrieve file tuples which is a list of
    # (eeg filepath, clip_st, sz_label)
    file_tuples = compute_streaming_file_tuples(
        stanford_dataset_dir,
        lpch_dataset_dir,
        annotations_dir,
        clip_len,
        stride,
        sz_label_sensitivity,
    )
    data = []

    np.random.seed(seed)

    for (filepath, clip_st, sz_label) in tqdm(file_tuples, total=len(file_tuples)):
        row_df = {
            "filepath": filepath,
            "file_id": filepath.split("/")[-1].split(".eeghdf")[0],
            "id": filepath.split("/")[-1].split(".eeghdf")[0] + f"_{clip_st}",
            "target": sz_label,
            "clip_start": clip_st,
        }
        data.append(row_df)

    dp = mk.DataPanel(data)

    patientid_col = dp["filepath"].map(function=eeg_patientid_loader)
    dp.add_column(
        "patient_id",
        patientid_col,
        overwrite=True,
    )

    dp_split = split_dp(
        dp,
        split_on="patient_id",
        train_frac=train_frac,
        valid_frac=valid_frac,
        test_frac=test_frac,
    )
    dp = merge_in_split(dp, dp_split)

    eeg_input_col = dp[["clip_start", "filepath"]].to_lambda(
        fn=partial(streaming_eeg_loader, clip_len=clip_len)
    )

    dp.add_column(
        "input",
        eeg_input_col,
        overwrite=True,
    )

    # Add metadata
    age_col = dp["filepath"].map(function=eeg_age_loader)
    dp.add_column(
        "age",
        age_col,
        overwrite=True,
    )

    # remove duplicate ID rows
    dp = dp.lz[~dp["id"].duplicated()]

    return dp
