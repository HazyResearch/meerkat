import random

import numpy as np
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
