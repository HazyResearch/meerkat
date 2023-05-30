from typing import Sequence

import h5py
import pandas as pd
import torch
from meddlr.utils import env
from tqdm.auto import tqdm

import meerkat as mk
from meerkat.interactive.formatter.tensor import TensorFormatterGroup


def is_url(path):
    return (
        isinstance(path, str)
        and path.startswith("http://")
        or path.startswith("https://")
    )


def build_slice_df(
    paths: Sequence[str], defer: bool = True, pbar: bool = False, slice_dim: int = 0
):
    """Build a dataframe containing the slices to visualize.

    Args:
        paths: A list of paths to the HDF5 files containing the slices.
            Files should follow the meddlr dataset format with keys:
            * ``kspace``: The fully sampled kspace for the slice.
              Shape: (Z, Y, X, #coils)
            * ``maps``: The sensitivity maps for the slice.
              Shape: (Z, Y, X, #coils)
            * ``target`` (optional): The target image for the slice. Shape: (Z, Y, X).
        defer: Whether to defer loading the data.
    """

    def _load_data(row):
        path = row["path"]
        sl = row["sl"]
        with h5py.File(path, "r") as f:
            kspace = torch.as_tensor(f["kspace"][sl])
            maps = torch.as_tensor(f["maps"][sl])
            target = torch.as_tensor(f["target"][sl]) if "target" in f else None
        return {"kspace": kspace, "maps": maps, "target": target}

    pm = env.get_path_manager()

    records = []
    for path in tqdm(paths, disable=not pbar):
        path = pm.get_local_path(path)
        with h5py.File(path, "r") as f:
            num_slices = f["kspace"].shape[0]
        for sl in range(num_slices):
            records.append({"path": path, "sl": sl})

    df = pd.DataFrame.from_records(records)
    df = mk.DataFrame.from_pandas(df)
    if defer:
        df_load = mk.defer(df, _load_data)
    else:
        df_load = mk.map(df, _load_data)
    df = mk.concat([df, df_load], axis=1).drop("index")

    # Set formatters
    df["kspace"].formatters = TensorFormatterGroup().defer()
    df["maps"].formatters = TensorFormatterGroup().defer()
    if "target" in df:
        df["target"].formatters = TensorFormatterGroup().defer()
    return df
