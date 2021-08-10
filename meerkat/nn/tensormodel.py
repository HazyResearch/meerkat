from __future__ import annotations

from functools import partial
from typing import Dict, List

import cytoolz as tz
import torch
from tqdm import tqdm

from meerkat.columns.tensor_column import TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.nn.activation import ActivationOp
from meerkat.nn.embedding_column import EmbeddingColumn
from meerkat.nn.instances_column import InstancesColumn
from meerkat.nn.model import Model
from meerkat.nn.segmentation_column import SegmentationOutputColumn


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

    def activation(
        self,
        dataset: DataPanel,
        target_module: str,  # TODO(Priya): Support multiple activation layers
        input_columns: List[str],
        batch_size=32,
    ) -> EmbeddingColumn:  # TODO(Priya): Disable return?

        # Get an activation operator
        activation_op = ActivationOp(self.model, target_module, self.device)
        activations = []

        for batch in tqdm(
            dataset.batch(batch_size),
            total=(len(dataset) // batch_size + int(len(dataset) % batch_size != 0)),
        ):
            # Process the batch
            input_batch = self.process_batch(batch, input_columns)

            # Forward pass
            with torch.no_grad():
                self.model(input_batch)

            # Get activations for the batch
            batch_activation = {
                f"activation ({target_module})": EmbeddingColumn(
                    activation_op.extractor.activation.cpu().detach()
                )
            }

            # Append the activations
            activations.append(batch_activation)

        activations = tz.merge_with(lambda v: torch.cat(v), *activations)
        activation_col = activations[f"activation ({target_module})"]

        # dataset.add_column(f"activation ({target_module})", activation_col)
        return activation_col

    def semantic_segmentation(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        batch_size: int = 32,
        num_classes: int = None,
    ) -> DataPanel:

        # Handles outputs for semantic_segmentation tasks

        predictions = dataset.map(
            function=partial(self._predict, input_columns=input_columns),
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=SegmentationOutputColumn,
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
            function=partial(self._predict, input_columns=input_columns),
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=TensorColumn,
        )

        dataset.add_column("preds", output_dp["preds"])

        return output_dp

    def instance_segmentation(
        self, dataset: DataPanel, input_columns: List[str], batch_size: int = 32
    ) -> DataPanel:

        # Handles outputs for instance segmentation

        output_dp = dataset.map(
            function=partial(self._predict, input_columns=input_columns),
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=InstancesColumn,
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
