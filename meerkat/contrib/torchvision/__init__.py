import random

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.cifar import CIFAR10

import meerkat as mk


def get_torchvision_dataset(dataset_name, download_dir, is_train):
    return NotImplemented
    # dataset = torchvision.datasets.__dict__[dataset_name.upper()](
    #     root=download_dir,
    #     train=is_train,
    #     download=True,
    # )


def get_all_cifar10_splits(
    download_dir,
    frac_val=0.0,
    transforms=None,
    seed=42,
):
    """
    Load a CIFAR10 split as a Meerkat DataPanel.

    Args:
        download_dir: download directory
        frac_val: fraction of training set to use for validation
        transforms: torchvision tranforms to apply to train, val and test
        seed: random seed

    Returns:
        DataPanels for train, val and test containing columns
            `raw_image`, `image` and `label`
    """
    # Validation fraction should  be between 0 and 1
    assert 0 <= frac_val <= 1, "frac_val must be between 0 and 1"

    # Load the train and test datasets
    train_dataset = CIFAR10(
        root=download_dir,
        train=True,
        download=True,
    )

    test_dataset = CIFAR10(
        root=download_dir,
        train=False,
        download=True,
    )

    # Build the train and test DataPanels
    train_dp = mk.DataPanel(
        {
            "raw_image": train_dataset.data,
            "label": mk.TensorColumn(train_dataset.targets),
        }
    )
    train_dp["split"] = np.array(["train"] * len(train_dp))

    test_dp = mk.DataPanel(
        {
            "raw_image": test_dataset.data,
            "label": mk.TensorColumn(test_dataset.targets),
        }
    )
    test_dp["split"] = np.array(["test"] * len(test_dp))

    if frac_val > 1e-4:
        # sample indices for splitting off val
        random.seed(seed)
        val_indices = set(
            random.sample(
                range(len(train_dp)),
                int(frac_val * len(train_dp)),
            )
        )

        train_dp["split"] = pd.Series(
            ["train" if i not in val_indices else "val" for i in range(len(train_dp))]
        )

    # Split up the train dataset into train and val
    val_dp = train_dp.lz[train_dp["split"] == "val"]
    train_dp = train_dp.lz[train_dp["split"] != "val"]

    def _transform(x):
        """Convert to PIL image and then apply the transforms."""
        x = Image.fromarray(x)
        if transforms:
            return transforms(x)
        return x

    train_dp["image"] = mk.LambdaColumn(train_dp["raw_image"], _transform)
    val_dp["image"] = mk.LambdaColumn(val_dp["raw_image"], _transform)
    test_dp["image"] = mk.LambdaColumn(test_dp["raw_image"], _transform)

    return {
        "train": train_dp,
        "val": val_dp,
        "test": test_dp,
    }


def get_cifar10(download_dir, is_train=True, frac_val=0.0, transforms=None):
    """Load CIFAR10 as a Meerkat DataPanel.

    Args:
        download_dir: download directory
        is_train: load train set
        frac_val: fraction of training set to use for validation
        transforms: torchvision tranforms to apply

    Returns:
        a DataPanel containing columns `raw_image`, `image` and `label`
    """
    dataset = CIFAR10(
        root=download_dir,
        train=is_train,
        download=True,
    )

    dp = mk.DataPanel(
        {
            "raw_image": dataset.data,
            "label": mk.TensorColumn(dataset.targets),
        }
    )

    def _transform(x):
        """Convert to PIL image and then apply the transforms."""
        x = Image.fromarray(x)
        if transforms:
            return transforms(x)
        return x

    dp["image"] = mk.LambdaColumn(dp["raw_image"], _transform)
    dp["split"] = np.array(["train"] * len(dp))

    if is_train and frac_val > 1e-4:
        # sample indices for splitting off val
        val_indices = set(
            random.sample(
                range(len(dp)),
                int(frac_val * len(dp)),
            )
        )

        dp["split"] = np.array(
            ["train" if i not in val_indices else "val" for i in range(len(dp))]
        )

    return dp
