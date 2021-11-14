import random

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


def get_cifar10(download_dir: str, frac_val: float = 0.0, download: bool = True):
    """Load CIFAR10 as a Meerkat DataPanel.

    Args:
        download_dir: download directory
        frac_val: fraction of training set to use for validation

    Returns:
        a DataPanel containing columns `raw_image`, `image` and `label`
    """
    dps = []
    for split in ["train", "test"]:
        dataset = CIFAR10(
            root=download_dir,
            train=split == "train",
            download=download,
        )

        dp = mk.DataPanel(
            {
                "raw_image": dataset.data,
                "label": mk.TensorColumn(dataset.targets),
            }
        )

        if split == "train" and frac_val > 1e-4:
            # sample indices for splitting off val
            val_indices = set(
                random.sample(
                    range(len(dp)),
                    int(frac_val * len(dp)),
                )
            )

            dp["split"] = [
                "train" if i not in val_indices else "val" for i in range(len(dp))
            ]
        else:
            dp["split"] = [split] * len(dataset)

        dps.append(dp)
    dp = mk.concat(dps)

    dp["image"] = mk.LambdaColumn(dp["raw_image"], Image.fromarray)

    return dp
