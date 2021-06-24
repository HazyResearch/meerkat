"""WILDS integration for Meerkat."""
from __future__ import annotations

from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
from datasets import DatasetInfo
from torch.utils.data._utils.collate import default_collate

from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.numpy_column import NumpyArrayColumn
from meerkat.datapanel import DataPanel
from meerkat.tools.identifier import Identifier

from .config import base_config, populate_defaults
from .transforms import initialize_transform

try:
    import wilds
    from wilds.datasets.wilds_dataset import WILDSSubset

    _wilds_available = True
except ImportError:
    _wilds_available = False


def get_wilds_datapanel(
    dataset_name: str,
    root_dir: str,
    version: str = None,
    identifier: Identifier = None,
    column_names: List[str] = None,
    info: DatasetInfo = None,
    split: str = None,
    use_transform: bool = True,
    include_raw_input: bool = True,
):
    """Get a DataPanel that holds a WildsInputColumn alongside NumpyColumns for
    targets and metadata.

    Example:
    Run inference on the dataset and store predictions alongside the data.
    .. code-block:: python

        dp = get_wilds_datapane("fmow", root_dir="/datasets/", split="test")
        model = ... # get the model
        model.to(0).eval()

        @torch.no_grad()
        def predict(batch: dict):

            out = torch.softmax(model(batch["input"].to(0)), axis=-1)
            return {"pred": out.cpu().numpy().argmax(axis=-1)}

        dp = dp.update(function=predict, batch_size=128, is_batched_fn=True)


    Args:

        dataset_name (str, optional): dataset name. Defaults to `"fmow"`.
        version (str, optional): dataset version number, e.g., '1.0'.
            Defaults to the latest version.
        root_dir (str): the directory where the WILDS dataset is downloaded.
            See https://wilds.stanford.edu/ for download instructions.
        split (str, optional): see . Defaults to None.
        use_transform (bool, optional): Whether to apply the transform from the
            WILDS example directory on load. Defaults to True.
        identifier (Identifier, optional): [description]. Defaults to None.
        column_names (List[str], optional): [description]. Defaults to None.
        info (DatasetInfo, optional): [description]. Defaults to None.
        use_transform (bool, optional): [description]. Defaults to True.
        include_raw_input (bool, optional): include a column for the input without
            the transform applied – useful for visualizing images. Defaults to True.
    """
    if not _wilds_available:
        raise ImportError(
            "The WILDS package is not installed. To use the RG WILDS module, please "
            "install by following instructions at "
            "https://wilds.stanford.edu/get_started/"
        )

    input_column = WILDSInputColumn(
        dataset_name=dataset_name,
        version=version,
        root_dir=root_dir,
        split=split,
        use_transform=use_transform,
    )
    output_column = input_column.get_y_column()
    metadata_columns = input_column.get_metadata_columns()

    data = {"input": input_column, "y": output_column, **metadata_columns}
    if include_raw_input:
        data["raw_input"] = input_column.copy()
        data["raw_input"].use_transform = False
        data["raw_input"]._data.transform = lambda x: x
        data["raw_input"]._collate_fn = lambda x: x

    return DataPanel(
        data,
        identifier=identifier,
        column_names=column_names,
        info=info,
        split=split,
    )


class WILDSInputColumn(AbstractColumn):
    def __init__(
        self,
        dataset_name: str = "fmow",
        version: str = None,
        root_dir: str = None,
        split: str = None,
        use_transform: bool = True,
        **kwargs,
    ):
        """A column wrapper around a WILDS dataset that can lazily load the
        inputs for each dataset.

        Args:
            dataset_name (str, optional): dataset name. Defaults to `"fmow"`.
            version (str, optional): dataset version number, e.g., '1.0'.
                Defaults to the latest version.
            root_dir (str, optional): the directory  . Defaults to None.
            split (str, optional): the split . Defaults to None.
            use_transform (bool, optional): Whether to apply the transform from the
                WILDS example directory on load. Defaults to True.
        """

        self.dataset_name = dataset_name
        self.version = version
        self.root_dir = root_dir
        self.split = split
        self.use_transform = use_transform

        dataset = wilds.get_dataset(
            dataset=dataset_name, version=version, root_dir=root_dir
        )
        self.root = dataset.root
        self.split = split

        self.metadata_columns = {}
        # get additional, dataset-specific metadata columns
        if dataset_name == "fmow":
            metadata_df = dataset.metadata
            metadata_df = metadata_df[metadata_df.split != "seq"]
            if self.split is not None:
                metadata_df = metadata_df[
                    dataset.split_array == dataset.split_dict[self.split]
                ]
            self.metadata_columns.update(
                {
                    field: NumpyArrayColumn(data=series.values)
                    for field, series in metadata_df.iteritems()
                }
            )

        if use_transform:
            # we need a WILDS config in order to initialize transform
            config = Namespace(dataset=dataset_name, **base_config)
            config = populate_defaults(config)
            transform_name = (
                config.train_transform if split == "train" else config.eval_transform
            )
            transform = initialize_transform(
                transform_name, config=config, dataset=dataset
            )
            # wilds defaults to torch `default_collate`
            collate = default_collate if dataset.collate is None else dataset.collate
        else:
            transform = collate = None

        if split is not None:
            dataset = dataset.get_subset(split, transform=transform, frac=1.0)
        elif transform is not None:
            # only WILDSSubset supports applying a transform, so we use it even if no
            # split is applied
            dataset = WILDSSubset(dataset, np.arange(len(dataset)), transform=transform)

        self.metadata_columns.update(
            {
                f"meta_{field}": NumpyArrayColumn(data=dataset.metadata_array[:, idx])
                for idx, field in enumerate(dataset.metadata_fields)
            }
        )
        super(WILDSInputColumn, self).__init__(data=dataset, collate_fn=collate)

    def get_y_column(self):
        """Get a NumpyArrayColumn holding the targets for the dataset.

        Warning: `WildsDataset`s may remap indexes in arbitrary ways so it's important
        not to directly try to access the underlying data structures, instead relying on
        the `y_array` and `metadata_array` properties which are universal across WILDS
        datasets.
        """
        return NumpyArrayColumn(data=self.data.y_array)

    def get_metadata_columns(self):
        return self.metadata_columns

    def _get_cell(self, index: int):
        # only get input (not y and meta)
        return self.data[index][0]

    def _repr_pandas_(
        self,
    ) -> pd.Series:
        series = pd.Series(np.arange(len(self._data)))
        return series.apply(
            lambda x: f"WildsInput(path={self.root}/images/rgb_img_{x}.png)"
        )

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object.

        Warning: this write
        is very lightweight, only the name of the dataset (`dataset_name`), the
        directory of the dataset `root_dir`, and the other args to `__init__` are
        written to disk. If the data at `root_dir` is modified, `read` will return a
        column with different data.
        """
        return {
            "dataset_name",
            "version",
            "root_dir",
            "split",
            "use_transform",
            "_visible_rows",
        }
