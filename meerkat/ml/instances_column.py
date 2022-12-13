from __future__ import annotations

import itertools
from typing import Sequence

from tqdm import tqdm

from meerkat.columns.object.base import ObjectColumn
from meerkat.columns.tensor.torch import TorchTensorColumn
from meerkat.tools.lazy_loader import LazyLoader

ops = LazyLoader("torchvision.ops")


class InstancesColumn(ObjectColumn):
    def __init__(self, data: Sequence = None, *args, **kwargs):

        super(InstancesColumn, self).__init__(data=data, *args, **kwargs)

    def num_instances(self) -> TorchTensorColumn:
        """Returns the number of instances for each image."""

        data_col = TorchTensorColumn([len(instance) for instance in self])

        return data_col

    def get_field(self, field: str, batch_size: int = 32) -> ObjectColumn:
        """Returns a speific field of the Instances object.

        Returns:
            ListColumn: List of the specified field from Instances object.
        """

        data = []
        for batch in tqdm(
            self.batch(batch_size),
            total=(len(self) // batch_size + int(len(self) % batch_size != 0)),
        ):
            batch_data = [instance.get_fields()[field] for instance in batch]
            data = list(itertools.chain(data, batch_data))

        data_col = ObjectColumn(data)

        return data_col

    def nms(
        self,
        iou_threshold: float,
        batch_size: int = 32,
    ) -> InstancesColumn:
        """Returns the instances retained after NMS for each image.

        Returns:
            ListColumn: Contains tensors of shape (N,4) representing the boxes retained
                        after NMS where N is the number of boxes.
        """

        data = []
        for batch in tqdm(
            self.batch(batch_size),
            total=(len(self) // batch_size + int(len(self) % batch_size != 0)),
        ):

            batch_data = [
                instance.get_fields()["pred_boxes"].tensor[
                    ops.batched_nms(
                        boxes=instance.get_fields()["pred_boxes"].tensor,
                        scores=instance.get_fields()["scores"],
                        idxs=instance.get_fields()["pred_classes"],
                        iou_threshold=iou_threshold,
                    )
                ]
                for instance in batch
            ]
            data = list(itertools.chain(data, batch_data))

        data_col = ObjectColumn(data)
        return data_col
