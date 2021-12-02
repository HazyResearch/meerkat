import os
import meerkat as mk
import eeghdf
import h5py
import numpy as np
import pyedflib
import torch
from scipy.signal import resample
from tqdm import tqdm
from functools import partial
import hashlib
import math
import pandas as pd
import json
import yaml
from scipy.fftpack import fft

FREQUENCY = 200
INCLUDED_CHANNELS = [
    "EEG FP1",
    "EEG FP2",
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
    "EEG FZ",
    "EEG CZ",
    "EEG PZ",
]

STANFORD_INCLUDED_CHANNELS = [
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


def compute_file_tuples(raw_dataset_dir, dataset_dir, split, clip_len, stride):
    """
    Args:
        dataset_dir (str): location where resampled signals are
        split (str): whether train, dev, test
        clip_len(int): length of each clip in the input eeg segments
        stride (int): how to sample clips from eeg signal

    Returns (file_name, clip_idx, seizure_label) tuples
            for the given split, clip_len, and stride
    The clip_idx indicates which clip (i.e. segment of EEG signal with clip_len seconds)
    The stride determines how to sample clips from the eeg signal
            (e.g. if stride=clip_len we have no overlapping clips)
    """

    # retrieve paths of all edf files in the dataset_dir for given split
    edf_files = []
    edf_fullfiles = []
    for path, _, files in os.walk(os.path.join(raw_dataset_dir, split)):
        for name in files:
            if ".edf" in name:
                edf_fullfiles.append(os.path.join(path, name))
                edf_files.append(name)

    resampled_files = os.listdir(dataset_dir)
    file_tuples = []

    for h5_fn in resampled_files:
        edf_fn = h5_fn.split(".h5")[0] + ".edf"
        if edf_fn not in edf_files:
            continue
        edf_fn_full = [file for file in edf_fullfiles if edf_fn in file]
        if len(edf_fn_full) != 1:
            print(f"{edf_fn} found {len(edf_fn_full)} times!")
            print(edf_fn_full)

        edf_fn_full = edf_fn_full[0]
        seizure_times = get_seizure_times(edf_fn_full.split(".edf")[0])

        h5_fn_full = os.path.join(dataset_dir, h5_fn)
        with h5py.File(h5_fn_full, "r") as hf:
            resampled_sig = hf["resampled_signal"][()]

        num_clips = (resampled_sig.shape[-1] - clip_len * FREQUENCY) // (
            stride * FREQUENCY
        ) + 1

        for i in range(num_clips):
            start_window = i * FREQUENCY * stride
            end_window = np.minimum(
                start_window + FREQUENCY * clip_len, resampled_sig.shape[-1]
            )

            is_seizure = 0
            for t in seizure_times:
                start_t = int(t[0] * FREQUENCY)
                end_t = int(t[1] * FREQUENCY)
                if not ((end_window < start_t) or (start_window > end_t)):
                    is_seizure = 1
                    break
            file_tuples.append((edf_fn, i, is_seizure))

    return file_tuples


def get_sz_labels(
    edf_fn,
    clip_idx,
    time_step_size=1,
    clip_len=60,
    stride=60,
):
    """
    Convert entire EEG sequence into clips of length clip_len
    Args:
        edf_fn: edf/eeghdf file name, full path
        channel_names: list of channel names
        clip_idx: index of current clip/sliding window, int
        time_step_size: length of each time step, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        stride: stride size, by how many seconds the sliding window moves, int
    Returns:
        seizure_labels: per-time-step seizure labels
        is_seizure: overall label, 1 if at least one seizure in clip
    """

    physical_clip_len = int(FREQUENCY * clip_len)
    start_window = clip_idx * FREQUENCY * stride
    end_window = start_window + physical_clip_len

    # get seizure times, take min_sz_len into account
    if ".edf" in edf_fn:
        seizure_times = get_seizure_times(edf_fn.split(".edf")[0])
    else:
        raise NotImplementedError

    # get per-time-step seizure labels
    num_time_steps = int(clip_len / time_step_size)
    seizure_labels = np.zeros((num_time_steps)).astype(int)
    is_seizure = 0
    for t in seizure_times:
        start_t = int(t[0] * FREQUENCY)
        end_t = int(t[1] * FREQUENCY)
        if not ((end_window < start_t) or (start_window > end_t)):
            is_seizure = 1

            start_t_sec = int(t[0])  # start of seizure in int seconds
            end_t_sec = int(t[1])  # end of seizure in int seconds

            # shift start_t_sec and end_t_sec so that they start at current clip
            start_t_sec = np.maximum(0, start_t_sec - int(start_window / FREQUENCY))
            end_t_sec = np.minimum(clip_len, end_t_sec - int(start_window / FREQUENCY))
            # print("start_t_sec: {}; end_t_sec: {}".format(start_t_sec, end_t_sec))

            # time step size may not be 1-sec
            start_time_step = int(np.floor(start_t_sec / time_step_size))
            end_time_step = int(np.ceil(end_t_sec / time_step_size))

            seizure_labels[start_time_step:end_time_step] = 1

    return seizure_labels, is_seizure


def compute_slice_matrix(
    input_dict,
    time_step_size=1,
    clip_len=60,
    stride=60,
):
    """
    Convert entire EEG sequence into clips of length clip_len
    Args:
        channel_names: list of channel names
        clip_idx: index of current clip/sliding window, int
        h5_fn: file name of resampled signal h5 file (full path)
        time_step_size: length of each time step, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        stride: stride size, by how many seconds the sliding window moves, int
    Returns:
        eeg_clip: EEG clip
    """
    clip_idx = input_dict["clip_idx"]
    h5_fn = input_dict["h5_fn"]

    physical_clip_len = int(FREQUENCY * clip_len)

    start_window = clip_idx * FREQUENCY * stride

    with h5py.File(h5_fn, "r") as f:
        signal_array = f["resampled_signal"][()]
        resampled_freq = f["resample_freq"][()]
        assert resampled_freq == FREQUENCY

    # (num_channels, physical_clip_len)

    end_window = np.minimum(signal_array.shape[-1], start_window + physical_clip_len)
    curr_slc = signal_array[:, start_window:end_window]  # (num_channels, FREQ*clip_len)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    start_time_step = 0
    time_steps = []
    while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
        end_time_step = start_time_step + physical_time_step_size
        # (num_channels, physical_time_step_size)
        curr_time_step = curr_slc[:, start_time_step:end_time_step]

        time_steps.append(curr_time_step)
        start_time_step = end_time_step

    eeg_clip = np.stack(time_steps, axis=0)

    return eeg_clip


def get_seizure_times(file_name):
    """
    Args:
        file_name: file name of .edf file etc.
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_file = file_name.split(".edf")[0] + ".tse_bi"

    seizure_times = []
    with open(tse_file) as f:
        for line in f.readlines():
            if "seiz" in line:  # if seizure
                # seizure start and end time
                seizure_times.append(
                    [
                        float(line.strip().split(" ")[0]),
                        float(line.strip().split(" ")[1]),
                    ]
                )
    return seizure_times


def get_ordered_channels(
    file_name,
    labels_object,
    channel_names=INCLUDED_CHANNELS,
    verbose=False,
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


def get_edf_signals(edf):
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except IndexError:
            pass
    return signals


def resample_data(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled


def resample_files(raw_edf_dir, save_dir):
    """
    Resamples edf files to FREQUENCY and saves them in specified dir

    Args:
        raw_edf_dir (str): location where original edf files are located
        save_dir (str): location to save resampled signals
    """

    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    failed_files = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]

        save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)
        except BaseException:
            failed_files.append(edf_fn)

        orderedChannels = get_ordered_channels(edf_fn, f.getSignalLabels())
        signals = get_edf_signals(f)
        signal_array = np.array(signals[orderedChannels, :])
        sample_freq = f.getSampleFrequency(0)
        if sample_freq != FREQUENCY:
            signal_array = resample_data(
                signal_array,
                to_freq=FREQUENCY,
                window_size=int(signal_array.shape[1] / sample_freq),
            )

        with h5py.File(save_fn, "w") as hf:
            hf.create_dataset("resampled_signal", data=signal_array)
            hf.create_dataset("resample_freq", data=FREQUENCY)

    print("DONE. {} files failed.".format(len(failed_files)))


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
    ordered_channels = get_ordered_channels(
        filepath, eegf.electrode_labels, channel_names=STANFORD_INCLUDED_CHANNELS
    )
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

    if augmentation and split == "train":
        eeg_slice = random_augmentation(eeg_slice)

    if nomalize:
        eeg_slice = eeg_slice - EEG_MEANS
        eeg_slice = eeg_slice / EEG_STDS

    return torch.FloatTensor(eeg_slice)


def streaming_eeg_loader(
    input_dict,
    clip_len=60,
    nomalize=True,
):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """
    filepath = input_dict["filepath"]
    clip_start = input_dict["clip_start"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)
    ordered_channels = get_ordered_channels(
        filepath, eegf.electrode_labels, channel_names=STANFORD_INCLUDED_CHANNELS
    )
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
    # fourier transform
    fourier_signal = fft(signals, n=n, axis=-1)  # FFT on the last dimension

    # only take the positive freq part
    idx_pos = int(np.floor(n / 2))
    fourier_signal = fourier_signal[:, :idx_pos]
    amp = np.abs(fourier_signal)
    amp[amp == 0.0] = 1e-8  # avoid log of 0

    FT = np.log(amp)
    P = np.angle(fourier_signal)

    return FT, P


def fft_eeg_loader(
    input_dict, clip_len=60, augmentation=True, nomalize=True, offset=0, time_step=1
):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """
    eeg_slice = stanford_eeg_loader(
        input_dict, clip_len, augmentation, nomalize, offset
    ).T
    fft_clips = []
    for st in np.arange(0, clip_len, time_step):
        curr_eeg_clip = eeg_slice[:, st * FREQUENCY : (st + time_step) * FREQUENCY]
        curr_eeg_clip, _ = computeFFT(curr_eeg_clip.numpy(), n=time_step * FREQUENCY)
        fft_clips.append(curr_eeg_clip)

    fft_slice = np.stack(fft_clips, axis=0)

    return torch.FloatTensor(fft_slice).view(clip_len, -1)


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


def random_augmentation(signals):
    """
    Augment the data by randomly deciding whether to swap some channel pairs,
    and independently, whether to slightly shrink the amplitude of the signals
    Returns: the processed (augmented or not) signals
    """
    for pair in get_swap_pairs(STANFORD_INCLUDED_CHANNELS):
        if np.random.choice([True, False]):
            signals[:, [pair[0], pair[1]]] = signals[:, [pair[1], pair[0]]]
    if np.random.choice([True, False]):
        signals = signals * np.random.uniform(0.8, 1.2)
    return signals


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


def eeg_male_loader(filepath):
    """
    given filepath of an eeg, pulls relevant metadata
    right now only supports pulling age
    """
    # filepath = input_dict["filepath"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)
    return eegf.patient["gender"] == "Male"


def eeg_duration_loader(filepath):
    """
    given filepath of an eeg, pulls relevant metadata
    right now only supports pulling age
    """
    # filepath = input_dict["filepath"]

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)

    return eegf.duration_seconds
