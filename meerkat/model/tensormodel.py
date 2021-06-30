from __future__ import annotations

from typing import Dict, List

import cytoolz as tz
import torch
from tqdm import tqdm

from meerkat import DataPanel
from meerkat.columns.embedding_column import EmbeddingColumn
from meerkat.columns.prediction_column import ClassificationOutputColumn
from meerkat.columns.segmentation_column import SegmentationOutputColumn
from meerkat.columns.tensor_column import TensorColumn
from meerkat.model.activation import ActivationOp
from meerkat.model.model import Model


class TensorModel(Model):
    def __init__(
        self,
        identifier: str,
        model,
        device: str = None,
        is_classifier: bool = None,
        task: str = None,
    ):

        if model is None:
            raise ValueError(
                f"A PyTorch model is required with {self.__class__.__name__}."
            )
        super(TensorModel, self).__init__(
            identifier=identifier,
            model=model,
            device=device,
            is_classifier=is_classifier,
            task=task,
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

        elif self.task == "segmentation":
            # Output is a dict with key 'out'
            output_dict = {"logits": outputs["out"].to("cpu")}

        elif self.task == "timeseries":
            output_dict = {"preds": outputs.to("cpu")}

        return output_dict

    def process_batch(self, batch: DataPanel, input_columns: List[str]):

        # Convert the batch to torch.Tensor and move to device
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

        dataset.add_column(f"activation ({target_module})", activation_col)
        return activation_col

    def classification(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        batch_size: int = 32,
        num_classes: int = None,
        multi_label: bool = False,
        one_hot: bool = None,
        threshold=0.5,
    ) -> DataPanel:

        predictions = []

        for batch in tqdm(
            dataset.batch(batch_size),
            total=(len(dataset) // batch_size + int(len(dataset) % batch_size != 0)),
        ):

            # Process the batch to prepare input
            input_batch = self.process_batch(batch, input_columns)
            # Run forward pass
            prediction_dict = self.forward(input_batch)
            # Append the predictions
            predictions.append(prediction_dict)

        predictions = tz.merge_with(lambda v: torch.cat(v).to("cpu"), *predictions)

        logits = predictions["logits"]
        output_col = ClassificationOutputColumn(
            logits=logits,
            num_classes=num_classes,
            multi_label=multi_label,
            one_hot=one_hot,
            threshold=threshold,
        )

        output_dp = DataPanel(
            {
                "logits": output_col,
                "probs": output_col.probabilities(),
                "preds": output_col.predictions(),
            }
        )
        # TODO(Priya): Uncomment after append bug is resolved
        # dataset = dataset.append(classifier_dp, axis=1)
        return output_dp

    def segmentation(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        batch_size: int = 32,
        num_classes: int = None,
    ) -> DataPanel:

        predictions = []

        for batch in tqdm(
            dataset.batch(batch_size),
            total=(len(dataset) // batch_size + int(len(dataset) % batch_size != 0)),
        ):

            # Process the batch to prepare input
            input_batch = self.process_batch(batch, input_columns)
            # Run forward pass
            prediction_dict = self.forward(input_batch)
            # Append the predictions
            predictions.append(prediction_dict)

        predictions = tz.merge_with(lambda v: torch.cat(v).to("cpu"), *predictions)

        logits = predictions["logits"]
        output_col = SegmentationOutputColumn(logits=logits, num_classes=num_classes)

        output_dp = DataPanel(
            {
                "logits": output_col,
                "probs": SegmentationOutputColumn(output_col.probabilities().data),
                "preds": SegmentationOutputColumn(output_col.predictions().data),
            }
        )
        # TODO(Priya): Uncomment after append bug is resolved
        # dataset = dataset.append(classifier_dp, axis=1)
        return output_dp

    def timeseries(
        self, dataset: DataPanel, input_columns: List[str], batch_size: int = 32
    ) -> DataPanel:

        predictions = []

        for batch in tqdm(
            dataset.batch(batch_size),
            total=(len(dataset) // batch_size + int(len(dataset) % batch_size != 0)),
        ):

            # Process the batch to prepare input
            input_batch = self.process_batch(batch, input_columns)
            # Run forward pass
            prediction_dict = self.forward(input_batch)
            # Append the predictions
            predictions.append(prediction_dict)

        predictions = tz.merge_with(lambda v: torch.cat(v).to("cpu"), *predictions)

        output_col = TensorColumn(predictions["preds"])
        output_dp = DataPanel({"preds": output_col})

        # TODO(Priya): Uncomment after append bug is resolved
        # dataset = dataset.append(classifier_dp, axis=1)
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
        # TODO(Priya): The separate functions can be merged later
        # segmentation and classification models differ only in forward method
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
        elif self.task == "segmentation":
            return self.segmentation(dataset, input_columns, batch_size, num_classes)
        elif self.task == "timeseries":
            return self.timeseries(dataset, input_columns, batch_size)
        else:
            raise NotImplementedError
