from __future__ import annotations

from typing import Dict, List

import torch

from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.ml.instances_column import InstancesColumn
from meerkat.ml.model import Model
from meerkat.ml.segmentation_column import SegmentationOutputColumn


class TensorModel(Model):
    def __init__(
        self,
        # identifier: str,
        model: torch.nn.Module,
        is_classifier: bool = None,
        task: str = None,
        device: str = None,
    ):

        if model is None:
            raise ValueError(
                f"A PyTorch model is required with {self.__class__.__name__}."
            )
        super(TensorModel, self).__init__(
            # identifier=identifier,
            model=model,
            is_classifier=is_classifier,
            task=task,
            device=device,
        )

        # Move the model to device
        self.to(self.device)

    def forward(self, input_batch: Dict) -> Dict:
        # Run the model on the input_batch
        with torch.no_grad():
            outputs = self.model(input_batch)

        if self.is_classifier:
            # probs and preds can be handled at ClassificationOutputColumn
            # TODO(Priya): See if there is any case where these are to be returned
            output_dict = {"logits": outputs.to("cpu")}

        elif self.task == "semantic_segmentation":
            # Output is a dict with key 'out'
            output_dict = {"logits": outputs["out"].to("cpu")}

        elif self.task == "timeseries":
            output_dict = {"preds": outputs.to("cpu")}

        elif self.task == "instance_segmentation":
            output_dict = {
                "preds": [output["instances"].to("cpu") for output in outputs]
            }

        return output_dict

    def process_batch(self, batch: DataPanel, input_columns: List[str]):

        # Convert the batch to torch.Tensor and move to device
        if self.task == "instance_segmentation":
            input_batch = batch[input_columns[0]].data
        else:
            input_batch = batch[input_columns[0]].data.to(self.device)

        return input_batch

    def semantic_segmentation(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        batch_size: int = 32,
        num_classes: int = None,
    ) -> DataPanel:

        # Handles outputs for semantic_segmentation tasks

        predictions = dataset.map(
            function=self._predict,
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=SegmentationOutputColumn,
            input_cols=input_columns,
        )

        # TODO(Priya): How to pass other args of SegmentationOutputColumn above?
        output_col = SegmentationOutputColumn(
            logits=predictions["logits"].data, num_classes=num_classes
        )

        output_dp = DataPanel(
            {
                "logits": output_col,
                "probs": SegmentationOutputColumn(output_col.probabilities().data),
                "preds": SegmentationOutputColumn(output_col.predictions().data),
            }
        )

        dataset.add_column("logits", output_col)
        dataset.add_column("probs", output_col.probabilities())
        dataset.add_column("preds", output_col.predictions())

        return output_dp

    def timeseries(
        self, dataset: DataPanel, input_columns: List[str], batch_size: int = 32
    ) -> DataPanel:

        # Handles outputs for timeseries

        output_dp = dataset.map(
            function=self._predict,
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=TensorColumn,
            input_cols=input_columns,
        )

        dataset.add_column("preds", output_dp["preds"])

        return output_dp

    def instance_segmentation(
        self, dataset: DataPanel, input_columns: List[str], batch_size: int = 32
    ) -> DataPanel:

        # Handles outputs for instance segmentation

        output_dp = dataset.map(
            function=self._predict,
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=InstancesColumn,
            input_cols=input_columns,
        )

        dataset.add_column("preds", output_dp["preds"])

        return output_dp

    def output(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        batch_size: int = 32,
        num_classes: int = None,
        multi_label: bool = False,
        one_hot: bool = None,
        threshold=0.5,
    ):

        if self.is_classifier:
            return self.classification(
                dataset,
                input_columns,
                batch_size,
                num_classes,
                multi_label,
                one_hot,
                threshold,
            )
        elif self.task == "semantic_segmentation":
            return self.semantic_segmentation(
                dataset, input_columns, batch_size, num_classes
            )
        elif self.task == "timeseries":
            return self.timeseries(dataset, input_columns, batch_size)
        elif self.task == "instance_segmentation":
            return self.instance_segmentation(dataset, input_columns, batch_size)
        else:
            raise NotImplementedError
