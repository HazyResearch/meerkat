from __future__ import annotations

from argparse import Namespace
from typing import List

import numpy as np
from datasets import DatasetInfo
from torch.utils.data._utils.collate import default_collate

from robustnessgym.core.identifier import Identifier
from robustnessgym.mosaic.columns.abstract import AbstractColumn
from robustnessgym.mosaic.columns.numpy_column import NumpyArrayColumn
from robustnessgym.mosaic.datapane import DataPane

from .config import base_config, populate_defaults
from .transforms import initialize_transform

try:
    import wilds
    from wilds.datasets.wilds_dataset import WILDSSubset
except ImportError:
    raise ImportError(
        "The WILDS package is not installed. To use the RG WILDS module, please "
        "install by following instructions at  https://wilds.stanford.edu/get_started/"
    )


class WildsDataPane(DataPane):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        version: str = None,
        identifier: Identifier = None,
        column_names: List[str] = None,
        info: DatasetInfo = None,
        split: str = None,
        use_transform: bool = True,
    ):
        """ A DataPane that hols a `WildsInputColumn` alongside `NumpyColumn`s for 
        targets and metadata. 


        Example: 
        Run inference on the dataset and store predictions alongside the data. 
        ```
            dp = WildsDataPane("fmow", root_dir="/datasets/", split="test")
            model = ... # get the model
            model.to(0).eval()

            @torch.no_grad()
            def predict(batch: dict):
                out = torch.softmax(model(batch["input"].to(0)), axis=-1)
                return {"pred": out.cpu().numpy().argmax(axis=-1)}

            dp = dp.update(function=predict, batch_size=128, batched=True)
        ```

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
        """
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        input_column = WildsInputColumn(
            dataset_name=dataset_name,
            version=version,
            root_dir=root_dir,
            split=split,
            use_transform=use_transform,
        )
        output_column = input_column.get_y_column()
        metadata_column = input_column.get_metadata_column()
        super(WildsDataPane, self).__init__(
            {"input": input_column, "y": output_column, "meta": metadata_column},
            identifier=identifier,
            column_names=column_names,
            info=info,
            split=split,
        )


class WildsInputColumn(AbstractColumn):
    def __init__(
        self,
        dataset_name: str = "fmow",
        version: str = None,
        root_dir: str = None,
        split: str = None,
        use_transform: bool = True,
        **dataset_kwargs,
    ):
        """ A column wrapper around a WILDS dataset that can lazily load the inputs 
        for each dataset.

        Args:
            dataset_name (str, optional): dataset name. Defaults to `"fmow"`. 
            version (str, optional): dataset version number, e.g., '1.0'.
                Defaults to the latest version.
            root_dir (str, optional): the directory  . Defaults to None.
            split (str, optional): the split . Defaults to None.
            use_transform (bool, optional): Whether to apply the transform from the 
                WILDS example directory on load. Defaults to True. 
        """
        self._state = {
            "dataset_name": dataset_name,
            "version": version,
            "root_dir": root_dir,
            **dataset_kwargs,
        }
        dataset = wilds.get_dataset(
            dataset=dataset_name, version=version, root_dir=root_dir, **dataset_kwargs
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
            dataset = dataset.get_subset(split, transform=transform)
        elif transform is not None:
            # only WILDSSubset supports applying a transform, so we use it even if no
            # split is applied
            dataset = WILDSSubset(dataset, np.arange(len(dataset)), transform=transform)

        super(WildsInputColumn, self).__init__(data=dataset, collate_fn=collate)

    def get_y_column(self):
        """ 
        Get a NumpyArrayColumn holding the targets for the dataset. 
        Warning: `WildsDataset`s may remap indexes in arbitrary ways so it's important
        not to directly try to access the underlying data structures, instead relying on
        the `y_array` and `metadata_array` properties which are universal across WILDS
        datasets. 
        """
        return NumpyArrayColumn(data=self.data.y_array)

    def get_metadata_column(self):
        return NumpyArrayColumn(data=self.data.metadata_array)

    def _get_cell(self, index: int):
        # only get input (not y and meta)
        return self.data[index][0]

    def write(
        self, path: str, write_together: bool = None, write_data: bool = None
    ) -> None:
        # TODO (Sabri): implement read and write – I think this requires significant 
        # changes to ColumnStorageMixin and StateDictMixin, so I'm punting to another PR
        raise NotImplementedError("Writing `WildsInputColumn` not supported.")

    def get_state(self):
        raise NotImplementedError("Writing `WildsInputColumn` not supported.")

    @classmethod
    def from_state(cls, state, *args, **kwargs) -> object:
        raise NotImplementedError("Reading `WildsInputColumn` not supported.")

    @classmethod
    def read(cls, path: str, *args, **kwargs) -> object:
        raise NotImplementedError("Reading `WildsInputColumn` not supported.")

