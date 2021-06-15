from __future__ import annotations

from typing import Dict, List

import cytoolz as tz
import torch
from tqdm import tqdm

from mosaic import DataPanel
from mosaic.columns.embedding_column import EmbeddingColumn
from mosaic.columns.prediction_column import ClassificationOutputColumn
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

    def process_batch(self, batch: DataPanel, input_columns: List[str]):

        # Convert the batch to torch.Tensor and move to device
        input_batch = batch[input_columns[0]].data.to(self.device)

        return input_batch

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
        # TODO (Priya): Include other arguments of batch method
        for batch in tqdm(dataset.batch(batch_size)):

            # Process the batch to prepare input
            input_batch = self.process_batch(batch, input_columns)
            # Run forward pass
            prediction_dict = self.forward(input_batch)
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

        for batch in tqdm(dataset.batch(batch_size)):
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
