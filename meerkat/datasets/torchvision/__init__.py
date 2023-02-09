import random

from PIL import Image

import meerkat as mk


def get_torchvision_dataset(dataset_name, download_dir, is_train):
    return NotImplemented
    # dataset = torchvision.datasets.__dict__[dataset_name.upper()](
    #     root=download_dir,
    #     train=is_train,
    #     download=True,
    # )


def get_cifar10(download_dir: str, frac_val: float = 0.0, download: bool = True):
    """Load CIFAR10 as a Meerkat DataFrame.

    Args:
        download_dir: download directory
        frac_val: fraction of training set to use for validation

    Returns:
        a DataFrame containing columns `raw_image`, `image` and `label`
    """
    from torchvision.datasets.cifar import CIFAR10

    dfs = []
    for split in ["train", "test"]:
        dataset = CIFAR10(
            root=download_dir,
            train=split == "train",
            download=download,
        )

        df = mk.DataFrame(
            {
                "raw_image": dataset.data,
                "label": mk.TorchTensorColumn(dataset.targets),
            }
        )

        if split == "train" and frac_val > 1e-4:
            # sample indices for splitting off val
            val_indices = set(
                random.sample(
                    range(len(df)),
                    int(frac_val * len(df)),
                )
            )

            df["split"] = [
                "train" if i not in val_indices else "val" for i in range(len(df))
            ]
        else:
            df["split"] = [split] * len(dataset)

        dfs.append(df)
    df = mk.concat(dfs)

    df["image"] = mk.DeferredColumn(df["raw_image"], Image.fromarray)

    return df
