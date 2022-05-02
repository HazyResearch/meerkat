import os
import h5py
import numpy as np
import pyedflib
import torch
from scipy.signal import resample
from tqdm import tqdm
from .data_utils import (
    get_ordered_channels,
    computeFFT,
    get_gnn_support,
)


SEIZURE_STRINGS = ["sz", "seizure", "absence", "spasm"]
FILTER_SZ_STRINGS = ["@sz", "@seizure"]

FREQUENCY = 200

TUH_INCLUDED_CHANNELS = [
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

TUH_EEG_MEANS = np.array(
    [
        -10.5187,
        -8.6126,
        -10.7719,
        -12.7746,
        -13.3394,
        -15.3364,
        -13.3333,
        -14.6708,
        -10.4807,
        -7.7543,
        -13.7053,
        -10.1221,
        -11.6965,
        -13.5834,
        -14.4518,
        -10.0639,
        -12.1372,
        -9.9625,
        -10.2223,
    ]
)

TUH_EEG_STDS = np.array(
    [
        293.5294,
        267.9896,
        291.9434,
        312.8902,
        274.3873,
        353.7668,
        342.4909,
        304.0491,
        312.6968,
        279.8322,
        286.1863,
        258.3522,
        282.8169,
        275.4195,
        273.7004,
        268.8392,
        300.3167,
        330.4046,
        395.5132,
    ]
)

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
TUH_FFT_MEANS = torch.FloatTensor(
    np.load(os.path.join(FILE_DIR, "data_stats/tuh_fft_mean.npy"))
)
TUH_FFT_STDS = torch.FloatTensor(
    np.load(os.path.join(FILE_DIR, "data_stats/tuh_fft_std.npy"))
)


def compute_file_tuples(dataset_dir, split, clip_len, stride):
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
    # edf_files = []
    # edf_fullfiles = []
    # for path, _, files in os.walk(os.path.join(raw_dataset_dir, split)):
    #     for name in files:
    #         if ".edf" in name:
    #             edf_fullfiles.append(os.path.join(path, name))
    #             edf_files.append(name)
    

    edf_fullfiles = np.loadtxt(os.path.join(FILE_DIR,f"edf_files_{split}.txt"),dtype=str)
    edf_files = [entry.split("/")[-1] for entry in edf_fullfiles]

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
        patient_id = edf_fn_full.split("/")[-3]
        edf_fn_full = os.path.join(dataset_dir.split("resampled_signal")[0],edf_fn_full.split("TUH_v1.5.2/")[-1])
        seizure_times = get_seizure_times(edf_fn_full.split(".edf")[0])

        h5_fn_full = os.path.join(dataset_dir, h5_fn)
        with h5py.File(h5_fn_full, "r") as hf:
            resampled_sig = hf["resampled_signal"][()]

        num_clips = (resampled_sig.shape[-1] - (clip_len) * FREQUENCY) // (
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
            file_tuples.append((edf_fn, patient_id, i, is_seizure))

    return file_tuples


def get_sz_labels(
    edf_fn, clip_idx, time_step_size=1, clip_len=60, stride=60,
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


def tuh_eeg_loader(
    input_dict,
    time_step_size=1,
    clip_len=60,
    stride=60,
    offset=0,
    normalize=True,
    augmentation=True,
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
    split = input_dict["split"]

    physical_clip_len = int(FREQUENCY * clip_len)

    start_window = max((clip_idx * FREQUENCY * stride) - offset * FREQUENCY, 0)

    with h5py.File(h5_fn, "r") as f:
        signal_array = f["resampled_signal"][()]
        resampled_freq = f["resample_freq"][()]
        assert resampled_freq == FREQUENCY

    # (num_channels, physical_clip_len)

    end_window = min(signal_array.shape[-1], start_window + physical_clip_len)
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

    eeg_clip = np.stack(time_steps, axis=0).transpose(0, 2, 1).reshape(-1, 19)

    swapped_pairs = None
    if augmentation and split == "train":  # and ss_clip_len == 0:
        eeg_clip, swapped_pairs = random_augmentation(
            eeg_clip, included_channels=TUH_INCLUDED_CHANNELS
        )
    gnn_support = get_gnn_support(swapped_pairs)

    if normalize:
        eeg_clip = eeg_clip - TUH_EEG_MEANS
        eeg_clip = eeg_clip / TUH_EEG_STDS

    eeg_clip = torch.FloatTensor(eeg_clip)

    return eeg_clip, gnn_support


def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Returns: list of tuples, each a pair of channel indices being swapped
    """
    f12 = (channels.index("EEG FP1"), channels.index("EEG FP2"))
    f34 = (channels.index("EEG F3"), channels.index("EEG F4"))
    f78 = (channels.index("EEG F7"), channels.index("EEG F8"))
    c34 = (channels.index("EEG C3"), channels.index("EEG C4"))
    t34 = (channels.index("EEG T3"), channels.index("EEG T4"))
    t56 = (channels.index("EEG T5"), channels.index("EEG T6"))
    o12 = (channels.index("EEG O1"), channels.index("EEG O2"))
    return [f12, f34, f78, c34, t34, t56, o12]


def random_augmentation(signals, included_channels=TUH_INCLUDED_CHANNELS):
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


def fft_tuh_eeg_loader(
    input_dict, time_step=1, clip_len=60, stride=60, offset=0, normalize=True
):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """
    eeg_slice, gnn_support = tuh_eeg_loader(
        input_dict, time_step, clip_len, stride, offset
    )
    eeg_slice = eeg_slice.T
    fft_clips = []
    for st in np.arange(0, clip_len, time_step):
        curr_eeg_clip = eeg_slice[:, st * FREQUENCY : (st + time_step) * FREQUENCY]
        curr_eeg_clip = computeFFT(curr_eeg_clip, n=time_step * FREQUENCY)
        fft_clips.append(curr_eeg_clip)

    fft_slice = torch.cat(fft_clips).view(clip_len, -1)

    if normalize:
        fft_slice = fft_slice - TUH_FFT_MEANS
        fft_slice = fft_slice / TUH_FFT_STDS

    # return {
    #     "fft_input": fft_slice.view(clip_len, -1).numpy(),
    #     "gnn_support": gnn_support.numpy(),
    # }
    return fft_slice, gnn_support  # .numpy()


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

        orderedChannels = get_ordered_channels(
            edf_fn, f.getSignalLabels(), channel_names=TUH_INCLUDED_CHANNELS
        )
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
