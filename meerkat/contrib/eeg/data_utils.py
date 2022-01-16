import os
import meerkat as mk
import eeghdf
import numpy as np
import torch
from functools import partial
import hashlib
import math
import pandas as pd
import json
from scipy.fftpack import fft
import pickle
from scipy.sparse import linalg
import scipy.sparse as sp


FREQUENCY = 200

INCLUDED_CHANNELS = [
    "EEG Fp1",
    "EEG Fp2",
    "EEG F3",
    "EEG F4",
    "EEG C3",
    "EEG C4",
    "EEG P3",
    "EEG P4",
    "EEG O1",
    "EEG O2",
    "EEG F7",
    "EEG F8",
    "EEG T3",
    "EEG T4",
    "EEG T5",
    "EEG T6",
    "EEG Fz",
    "EEG Cz",
    "EEG Pz",
]

SEIZURE_STRINGS = ["sz", "seizure", "absence", "spasm"]
FILTER_SZ_STRINGS = ["@sz", "@seizure"]

EEG_MEANS = np.array(
    [
        -0.5984,
        -0.2472,
        -0.6126,
        -0.5510,
        -0.6367,
        -0.6069,
        -0.0117,
        -0.0703,
        -0.1345,
        -0.3937,
        -1.6548,
        -1.6683,
        -1.4646,
        -1.3756,
        -0.8673,
        -0.9610,
        -0.5203,
        -0.3437,
        0.0175,
    ]
)
EEG_STDS = np.array(
    [
        244.10924185,
        240.52975195,
        165.48532864,
        165.96979748,
        82.23935837,
        81.51823183,
        159.74918782,
        165.79969442,
        173.39542554,
        172.15047675,
        214.96174016,
        212.43146653,
        139.07166002,
        138.3382818,
        163.11562014,
        166.22150418,
        136.93211929,
        148.61830207,
        151.21653098,
    ]
)


def get_ordered_channels(
    file_name, labels_object, channel_names=INCLUDED_CHANNELS, verbose=False,
):
    """
    Reads channel names and returns consistent ordering
    Args:
        file_name (str): name of edf file
        labels_object: extracted from edf signal using f.getSignalLabels()
        channel_names (List(str)): list of channel names
        verbose (bool): whether to be verbose
    Returns:
        list of channel indices in ordered form
    """

    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except IndexError:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def compute_stanford_file_tuples(
    stanford_dataset_dir, lpch_dataset_dir, file_marker_dir, splits
):
    """
    Given the splits, processes file tuples form filemarkers
    file tuple: (eeg filename, location of sz or -1 if no sz, split)

    Args:
        stanford_dataset_dir (str): data dir for stanford EEG files
        lpch_dataset_dir (str): data dir for lpc EEG files
        file_marker_dir (str): dir where file markers are stored
        splits (List[str]): which splits to process
    """

    file_tuples = []
    for split in splits:
        for hospital in ["lpch", "stanford"]:
            data_dir = (
                stanford_dataset_dir if hospital == "stanford" else lpch_dataset_dir
            )
            for sz_type in ["non_sz", "sz"]:
                fm_dir = (
                    f"{file_marker_dir}/file_markers_{hospital}/{sz_type}_{split}.txt"
                )
                filemarker_contents = open(fm_dir, "r").readlines()
                for fm in filemarker_contents:
                    fm_tuple = fm.strip("\n").split(",")
                    filepath = os.path.join(data_dir, fm_tuple[0])
                    fm_tuple = (filepath, float(fm_tuple[1]), split)
                    file_tuples.append(fm_tuple)

    return file_tuples


def compute_streaming_file_tuples(
    stanford_dataset_dir,
    lpch_dataset_dir,
    annotations_dir,
    clip_len,
    stride,
    sz_label_sensitivity,
):

    file_tuples = []
    # read all json files in annotations_dir
    for root, dirs, files in os.walk(annotations_dir):
        for name in files:
            if name.endswith((".json")) and "checkpoint" not in name:
                # read json
                annot_path = os.path.join(root, name)
                annot_dict = json.load(open(annot_path))
                sz_times = annot_dict["time"]

                # read eeg file
                file_id = name.split("-annot")[0]
                if "lpch" in root:
                    filepath = os.path.join(lpch_dataset_dir, file_id + ".eeghdf")
                else:
                    filepath = os.path.join(stanford_dataset_dir, file_id + ".eeghdf")

                eegf = eeghdf.Eeghdf(filepath)
                phys_signals = eegf.phys_signals
                signal_len = phys_signals.shape[1] // FREQUENCY

                clip_start = 0
                while clip_start + clip_len <= signal_len:
                    sz_label = 0
                    # check if there is a seizure anywhere in [clip_start,clip_end]
                    # needs to overlap by at least sz_label_sensitivity

                    for (sz_start, sz_len) in sz_times:
                        sz_end = sz_start + sz_len
                        clip_end = clip_start + clip_len
                        sz_overlap = min(clip_end, sz_end) - max(clip_start, sz_start)
                        if sz_overlap > sz_label_sensitivity:
                            sz_label = 1

                    ftup = (filepath, clip_start, sz_label)
                    file_tuples.append(ftup)
                    clip_start += stride

    return file_tuples


def get_stanford_sz_times(eegf):
    df = eegf.edf_annotations_df
    seizure_df = df[df.text.str.contains("|".join(SEIZURE_STRINGS), case=False)]
    seizure_df = seizure_df[
        seizure_df.text.str.contains("|".join(FILTER_SZ_STRINGS), case=False)
        == False  # noqa: E712
    ]

    seizure_times = seizure_df["starts_sec"].tolist()
    return seizure_times


def is_increasing(channel_indices):
    """
    Check if a list of indices is sorted in ascending order.
    If not, we will have to convert it to a numpy array before slicing,
    which is a rather expensive operation
    Returns: bool
    """
    last = channel_indices[0]
    for i in range(1, len(channel_indices)):
        if channel_indices[i] < last:
            return False
        last = channel_indices[i]
    return True


def stanford_eeg_loader(
    input_dict, clip_len=60, augmentation=True, nomalize=True, offset=0
):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """
    filepath = input_dict["filepath"]
    sz_start_idx = input_dict["sz_start_index"]
    fm_split = input_dict["fm_split"]
    split = input_dict["split"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)
    ordered_channels = get_ordered_channels(filepath, eegf.electrode_labels)
    phys_signals = eegf.phys_signals

    # get seizure time
    if sz_start_idx == -1 or fm_split != "train":
        sz_start = sz_start_idx
    else:
        sz_times = get_stanford_sz_times(eegf)
        sz_start = sz_times[int(sz_start_idx)]

    # extract clip
    if sz_start == -1:
        max_start = max(phys_signals.shape[1] - FREQUENCY * clip_len, 0)
        # if split == "train":
        #     if max_start == 0:
        #         sz_start = 0
        #     else:
        #         sz_start = np.random.randint(0, max_start)
        # else:
        #     sz_start = int(max_start / 2)
        sz_start = int(max_start / 2)
        sz_start /= FREQUENCY

    sz_start -= offset

    start_time = int(FREQUENCY * max(0, sz_start))
    end_time = start_time + int(FREQUENCY * clip_len)

    if not is_increasing(ordered_channels):
        eeg_slice = phys_signals[:, start_time:end_time]
        eeg_slice = eeg_slice[ordered_channels, :]
    else:
        eeg_slice = (
            phys_signals.s2u[ordered_channels]
            * phys_signals.data[ordered_channels, start_time:end_time].T
        ).T

    diff = FREQUENCY * clip_len - eeg_slice.shape[1]
    # padding zeros
    if diff > 0:
        zeros = np.zeros((eeg_slice.shape[0], diff))
        eeg_slice = np.concatenate((eeg_slice, zeros), axis=1)
    eeg_slice = eeg_slice.T

    swapped_pairs = None
    if augmentation and split == "train":
        eeg_slice, swapped_pairs = random_augmentation(eeg_slice)
    gnn_support = get_gnn_support(swapped_pairs)

    if nomalize:
        eeg_slice = eeg_slice - EEG_MEANS
        eeg_slice = eeg_slice / EEG_STDS

    return torch.FloatTensor(eeg_slice), gnn_support


def streaming_eeg_loader(
    input_dict, clip_len=60, nomalize=True,
):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """
    filepath = input_dict["filepath"]
    clip_start = input_dict["clip_start"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)
    ordered_channels = get_ordered_channels(filepath, eegf.electrode_labels)
    phys_signals = eegf.phys_signals

    start_time = int(FREQUENCY * max(0, clip_start))
    end_time = start_time + int(FREQUENCY * clip_len)

    if not is_increasing(ordered_channels):
        eeg_slice = phys_signals[:, start_time:end_time]
        eeg_slice = eeg_slice[ordered_channels, :]
    else:
        eeg_slice = (
            phys_signals.s2u[ordered_channels]
            * phys_signals.data[ordered_channels, start_time:end_time].T
        ).T

    diff = FREQUENCY * clip_len - eeg_slice.shape[1]
    # padding zeros
    if diff > 0:
        zeros = np.zeros((eeg_slice.shape[0], diff))
        eeg_slice = np.concatenate((eeg_slice, zeros), axis=1)
    eeg_slice = eeg_slice.T

    if nomalize:
        eeg_slice = eeg_slice - EEG_MEANS
        eeg_slice = eeg_slice / EEG_STDS

    return torch.FloatTensor(eeg_slice)


def split_dp(
    dp: mk.DataPanel,
    split_on: str,
    train_frac: float = 0.7,
    valid_frac: float = 0.1,
    test_frac: float = 0.2,
    other_splits: dict = None,
    salt: str = "",
):
    dp = dp.view()
    other_splits = {} if other_splits is None else other_splits
    splits = {
        "train": train_frac,
        "valid": valid_frac,
        "test": test_frac,
        **other_splits,
    }

    if not math.isclose(sum(splits.values()), 1):
        raise ValueError("Split fractions must sum to 1.")

    dp["split_hash"] = dp[split_on].apply(partial(hash_for_split, salt=salt))
    start = 0
    split_column = np.array(["unassigned"] * len(dp))
    for split, frac in splits.items():
        end = start + frac
        split_column[
            np.array(((start < dp["split_hash"]) & (dp["split_hash"] <= end)).data)
        ] = split
        start = end

    # need to drop duplicates, since split_on might not be unique
    df = pd.DataFrame({split_on: dp[split_on], "split": split_column}).drop_duplicates()
    return mk.DataPanel.from_pandas(df)


def hash_for_split(example_id: str, salt=""):
    GRANULARITY = 100000
    hashed = hashlib.sha256((str(example_id) + salt).encode())
    hashed = int(hashed.hexdigest().encode(), 16) % GRANULARITY + 1
    return hashed / float(GRANULARITY)


def merge_in_split(dp: mk.DataPanel, split_dp: mk.DataPanel):
    split_dp.columns
    if "split" in dp:
        dp.remove_column("split")
    split_on = [col for col in split_dp.columns if (col != "split") and col != "index"]
    assert len(split_on) == 1
    split_on = split_on[0]

    if split_dp[split_on].duplicated().any():
        # convert the datapanel to one row per split_on id
        df = split_dp[[split_on, "split"]].to_pandas()
        gb = df.groupby(split_on)

        # cannot have multiple splits per `split_on` id
        assert (gb["split"].nunique() == 1).all()
        split_dp = mk.DataPanel.from_pandas(gb["split"].first().reset_index())

    return dp.merge(split_dp, on=split_on)


def computeFFT(signals, n):
    """
    Args:
        signals: EEG signals, (number of channels, number of data points)
        n: length of positive frequency terms of fourier transform
    Returns:
        FT: log amplitude of FFT of signals, (number of channels, number of data points)
        P: phase spectrum of FFT of signals, (number of channels, number of data points)
    """
    # # fourier transform
    # fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # # only take the positive freq part
    # idx_pos = int(np.floor(n / 2))
    # fourier_signal = fourier_signal[:, :idx_pos]
    # amp = np.abs(fourier_signal)
    # amp[amp == 0.0] = 1e-8  # avoid log of 0

    # FT = np.log(amp)
    # P = np.angle(fourier_signal)

    # fourier transform
    fourier_signal = torch.fft.rfft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = torch.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = torch.log(amp)
    # P = np.angle(fourier_signal)

    return FT


def fft_eeg_loader(
    input_dict, clip_len=60, augmentation=True, nomalize=True, offset=0, time_step=1
):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """
    eeg_slice, gnn_support = stanford_eeg_loader(
        input_dict, clip_len, augmentation, nomalize, offset
    )
    eeg_slice = eeg_slice.T
    fft_clips = []
    for st in np.arange(0, clip_len, time_step):
        curr_eeg_clip = eeg_slice[:, st * FREQUENCY : (st + time_step) * FREQUENCY]
        curr_eeg_clip, _ = computeFFT(curr_eeg_clip.numpy(), n=time_step * FREQUENCY)
        fft_clips.append(curr_eeg_clip)

    fft_slice = np.stack(fft_clips, axis=0)

    return torch.FloatTensor(fft_slice).view(clip_len, -1), gnn_support


def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Returns: list of tuples, each a pair of channel indices being swapped
    """
    f12 = (channels.index("EEG Fp1"), channels.index("EEG Fp2"))
    f34 = (channels.index("EEG F3"), channels.index("EEG F4"))
    f78 = (channels.index("EEG F7"), channels.index("EEG F8"))
    c34 = (channels.index("EEG C3"), channels.index("EEG C4"))
    t34 = (channels.index("EEG T3"), channels.index("EEG T4"))
    t56 = (channels.index("EEG T5"), channels.index("EEG T6"))
    o12 = (channels.index("EEG O1"), channels.index("EEG O2"))
    return [f12, f34, f78, c34, t34, t56, o12]


def random_augmentation(signals, included_channels=INCLUDED_CHANNELS):
    """
    Augment the data by randomly deciding whether to swap some channel pairs,
    and independently, whether to slightly shrink the amplitude of the signals
    Returns: the processed (augmented or not) signals
    """
    swapped_pairs = None
    if np.random.choice([True, False]):
        swapped_pairs = get_swap_pairs(included_channels)
        for pair in swapped_pairs:
            signals[:, [pair[0], pair[1]]] = signals[:, [pair[1], pair[0]]]
    if np.random.choice([True, False]):
        signals = signals * np.random.uniform(0.8, 1.2)
    return signals, swapped_pairs


def unit_string_to_days(timestring: str):
    """
    timestring : <number> <days|weeks|months|years>
    return age in days as float
    """
    (val, unit) = timestring.split()
    floatval = float(val)
    if unit in "days":
        rel = float
    elif unit in "weeks":
        rel = 7 * floatval
    elif unit in "months":
        rel = 40 * 7 + 30 * floatval  # average value
    elif unit in "years":
        rel = 365.25 * floatval

    return rel


# with open(
#     "/home/ksaab/Documents/meerkat/meerkat/contrib/eeg/eeg_maturation_ages.yaml"
# ) as fp:
#     PMA_AGES = yaml.safe_load(fp)["PMA ages"]
# PMA_AGES_DAYS = [unit_string_to_days(xx) for xx in PMA_AGES]
# PMA_AGES_DAYS_ARR = np.array(PMA_AGES_DAYS)
# NORMALIZED_AGE_TIMES = np.linspace(0.0, 1.0, num=len(PMA_AGES_DAYS_ARR))


# def eeg_logage_loader(filepath):
#     """
#     given filepath of an eeg, pulls relevant metadata
#     right now only supports pulling age
#     """
#     # filepath = input_dict["filepath"]

#     # load EEG signal
#     eegf = eeghdf.Eeghdf(filepath)
#     age_years = min(eegf.age_years, 119)
#     indx = np.searchsorted(PMA_AGES_DAYS_ARR, age_years * 365.25)
#     normalized_age = NORMALIZED_AGE_TIMES[indx]

#     return normalized_age


def eeg_age_loader(filepath):
    """
    given filepath of an eeg, pulls relevant metadata
    right now only supports pulling age
    """
    # filepath = input_dict["filepath"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)
    return eegf.age_years


def eeg_patientid_loader(filepath):
    """
    given filepath of an eeg, pulls relevant metadata
    right now only supports pulling age
    """
    # filepath = input_dict["filepath"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)

    return eegf.patient["patientcode"]


# def eeg_male_loader(filepath):
#     """
#     given filepath of an eeg, pulls relevant metadata
#     right now only supports pulling age
#     """
#     # filepath = input_dict["filepath"]

#     # load EEG signal
#     eegf = eeghdf.Eeghdf(filepath)
#     return eegf.patient["gender"] == "Male"


# def eeg_duration_loader(filepath):
#     """
#     given filepath of an eeg, pulls relevant metadata
#     right now only supports pulling age
#     """
#     # filepath = input_dict["filepath"]

#     # load EEG signal
#     eegf = eeghdf.Eeghdf(filepath)

#     return eegf.duration_seconds


def get_gnn_support(swap_pairs):
    # currently only for dist graph

    adj_mat = get_combined_graph(swap_pairs)
    support = calculate_scaled_laplacian(adj_mat, lambda_max=None)

    return torch.FloatTensor(support)


def get_combined_graph(
    swap_nodes=None,
    adj_mat_dir="/home/ksaab/Documents/meerkat/meerkat/contrib/eeg/adj_mx_3d.pkl",
):
    """
    Get adjacency matrix for pre-computed distance graph
    Returns:
        adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
    """
    with open(adj_mat_dir, "rb") as pf:
        adj_mat = pickle.load(pf)
        adj_mat = adj_mat[-1]

    adj_mat_new = adj_mat.copy()
    if swap_nodes is not None:
        for node_pair in swap_nodes:
            for i in range(adj_mat.shape[0]):
                adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                adj_mat_new[i, i] = 1
            adj_mat_new[node_pair[0], node_pair[1]] = adj_mat[
                node_pair[1], node_pair[0]
            ]
            adj_mat_new[node_pair[1], node_pair[0]] = adj_mat[
                node_pair[0], node_pair[1]
            ]

    return adj_mat_new


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    """
    # adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_laplacian = np.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt
    ).transpose().dot(
        d_mat_inv_sqrt
    )  # .tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Scaled Laplacian for ChebNet graph convolution
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = np.identity(M, dtype=L.dtype)  # np.identity(M, format="coo", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L  # .tocoo()

