from __future__ import annotations

from typing import Dict, List

import cytoolz as tz
import torch

from mosaic import DataPanel
from mosaic.columns.embedding_column import EmbeddingColumn
from mosaic.columns.prediction_column import ClassificationOutputColumn
from mosaic.columns.tensor_column import TensorColumn
from mosaic.model.activation import ActivationOp
from mosaic.model.model import Model


class TensorModel(Model):
    def __init__(
        self,
        identifier,
        # task: Task = None,
        model,
        device=None,
        # is_classifier=None,
    ):

        super(TensorModel, self).__init__(
            identifier=identifier,
            device=device,  # task = task, is_classifier=is_classifier
        )

        self.model = model
        if model is None:
            # TODO(Priya): See what to do if used without any model
            raise ValueError(
                f"A PyTorch model is required with {self.__class__.__name__}."
            )

        # Move the model to device
        self.to(self.device)

    def forward(self, input_batch: Dict) -> Dict:
        # Run the model on the input_batch
        with torch.no_grad():
            outputs = self.model(input_batch)

        # probs and preds can be handled at ClassificationOutputColumn
        # TODO(Priya): See if there is any case where these are to be returned
        return {"logits": outputs.to("cpu")}

    def predict_batch(self, batch: DataPanel, input_columns: List[str]):

        # Convert the batch to torch.Tensor
        # TODO(Priya): Generalize for multiple input columns
        input_batch: TensorColumn = batch[input_columns[0]]

        input = input_batch.data.to(device=self.device)

        # Apply the model to the batch
        return self.forward(input)

    def classifier(
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

        for idx in range(0, len(dataset), batch_size):
            # Create the batch
            print(f"Batch: {idx//batch_size}")
            batch = dataset[idx : idx + batch_size]

            # Predict on the batch
            prediction_dict = self.predict_batch(
                batch=batch, input_columns=input_columns
            )
            # Append the predictions
            predictions.append(prediction_dict)

        predictions = tz.merge_with(lambda v: torch.cat(v).to("cpu"), *predictions)

        logits = predictions["logits"]
        classifier_output = ClassificationOutputColumn(
            logits=logits,
            num_classes=num_classes,
            multi_label=multi_label,
            one_hot=one_hot,
            threshold=threshold,
        )

        classifier_dp = DataPanel(
            {
                "logits": classifier_output,
                "probs": classifier_output.probabilities(),
                "preds": classifier_output.predictions(),
            }
        )

        # TODO(Priya): Uncomment after append bug is resolved
        # dataset = dataset.append(classifier_dp, axis=1)

        return classifier_dp

    def get_activation(
        self,
        dataset: DataPanel,
        target_module: str,  # TODO(Priya): Support multiple activation layers
        input_columns: List[str],
        batch_size=32,
    ) -> EmbeddingColumn:  # TODO(Priya): Disable return?

        # Get an activation operator
        activation_op = ActivationOp(self.model, target_module, self.device)
        activations = []

        for idx in range(0, len(dataset), batch_size):
            # Create the batch
            batch = dataset[idx : idx + batch_size]

            # Get activations for the batch
            batch_activation = activation_op.process_batch(batch, input_columns)

            # Append the activations
            activations.append(batch_activation)

        activations = tz.merge_with(lambda v: torch.cat(v), *activations)
        activation_col = activations[f"activation ({target_module})"]

        dataset.add_column(f"activation ({target_module})", activation_col)
        return activation_col
