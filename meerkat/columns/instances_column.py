from __future__ import annotations

import itertools
from typing import Sequence

from tqdm import tqdm

from meerkat.columns.list_column import ListColumn
from meerkat.tools.lazy_loader import LazyLoader

ops = LazyLoader("torchvision.ops")


class InstancesColumn(ListColumn):
    def __init__(self, data: Sequence = None, *args, **kwargs):

        super(InstancesColumn, self).__init__(data=data, *args, **kwargs)

    def num_instances(
        self, batch_size: int = 32
    ) -> ListColumn:  # TODO(Priya): ListColumn or some other type
        # Returns a column with the number of instances in each Instances object

        data = []  # Holds the number of instances in each object
        for batch in tqdm(
            self.batch(batch_size),
            total=(len(self) // batch_size + int(len(self) % batch_size != 0)),
        ):
            batch_data = [len(instance) for instance in batch]
            data = list(itertools.chain(data, batch_data))

        data_col = ListColumn(data)

        return data_col

    def get_field(self, field: str, batch_size: int = 32) -> ListColumn:
        # Returns a column of the required field from each Instances object

        data = []
        for batch in tqdm(
            self.batch(batch_size),
            total=(len(self) // batch_size + int(len(self) % batch_size != 0)),
        ):
            batch_data = [instance.get_fields()[field] for instance in batch]
            data = list(itertools.chain(data, batch_data))

        data_col = ListColumn(data)

        return data_col

    def get_nms(
        self,
        iou_threshold: float,
        batch_size: int = 32,
    ) -> ListColumn:
        # Returns ListColumn of tensors of indices given by torchvision.ops.nms

        data = []
        for batch in tqdm(
            self.batch(batch_size),
            total=(len(self) // batch_size + int(len(self) % batch_size != 0)),
        ):
            batch_data = [
                ops.nms(
                    instance.get_fields()["pred_boxes"].tensor,
                    instance.get_fields()["scores"],
                    iou_threshold,
                )
                for instance in batch
            ]
            data = list(itertools.chain(data, batch_data))

        data_col = ListColumn(data)
        return data_col
