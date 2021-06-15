from __future__ import annotations

from typing import Dict, List, Optional

import cytoolz as tz
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from mosaic import DataPanel
from mosaic.columns.embedding_column import EmbeddingColumn
from mosaic.columns.prediction_column import ClassificationOutputColumn
from mosaic.model.activation import ActivationOp
from mosaic.model.model import Model


# TODO(Priya): Need to test on NLP model
class HuggingfaceModel(Model):
    def __init__(
        self,
        identifier: str,
        # task: Task = None,
        model,
        tokenizer: Optional[AutoTokenizer] = None,
        device: str = None,
        # is_classifier=None,
    ):

        super(HuggingfaceModel, self).__init__(
            identifier=identifier,
            device=device,  # ,is_classifier=is_classifier, task=task
        )

        self.tokenizer = tokenizer
        if tokenizer is None:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.identifier)

        self.model = model
        if model is None:
            # TODO(Priya): See what to do if used without any model
            raise ValueError(
                f"A HuggingFace model is required with {self.__class__.__name__}."
            )

        """
            if self.task is None:
                if is_classifier is None:
                    raise ValueError("'is_classifier' required when task not specified")
            else:
                is_classifier = self.task.classification()
            if is_classifier:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.identifier
                )
            elif self.task.identifier == "ExtractiveQuestionAnswering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    self.identifier
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.identifier)

        self.task = task
        """

        # Move the model to device
        self.to(self.device)

    def forward(self, input_batch: Dict) -> Dict:

        """
        # Create the required outputs
        output_dict = {k: None for k in self.outputs}

        if self.task.classification():
            # Run the model on the input_batch
            # TODO(karan): allow outputs to generically contain side information (
            #  embeddings, attention, etc.)
            with torch.no_grad():
                outputs = self.model(**input_batch)

            # The logits are at the 0th index
            logits = outputs[0]

            # TODO(karan): these are still on GPU, do metric computation on GPU then
            #  move to CPU
            # TODO(karan): incrementally compute metrics?
            if "logits" in self.outputs:
                output_dict["logits"] = logits.to("cpu")

            if "probs" in self.outputs:
                output_dict["probs"] = torch.nn.functional.softmax(logits, dim=-1).to(
                    "cpu"
                )

            if "pred" in self.outputs:
                output_dict["pred"] = logits.argmax(dim=-1).to("cpu")
        else:
            with torch.no_grad():
                summary_token_ids = self.model.generate(**input_batch)
                summaries = [
                    self.tokenizer.decode(
                        token_id_list,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for token_id_list in summary_token_ids
                ]
                output_dict["pred"] = summaries

        return output_dict
        """
        # Run the model on the input_batch
        with torch.no_grad():
            outputs = self.model(**input_batch)

        # probs and preds can be handled at ClassificationOutputColumn
        # TODO(Priya): See if there is any case where these are to be returned
        # Logits are present at the 0th index
        return {"logits": outputs[0].to("cpu")}

    def encode_batch(self, batch: DataPanel, columns: List[str], **kwargs):
        # TODO(karan): Automatically writing this encoder for a variety of tasks
        return self.tokenizer(
            *[list(batch[key]) for key in columns],
            truncation=True,
            padding=True,
            **kwargs,
        )

    def process_batch(self, batch: DataPanel, input_columns: List[str]):

        # Tokenize the batch
        input_batch = self.encode_batch(batch=batch, columns=input_columns)

        # Convert the batch to torch.Tensor
        input_batch = tz.valmap(
            lambda v: torch.tensor(v).to(device=self.device), input_batch
        )

        # Return the converted batch
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

            # Process the batch
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
                self.model(**input_batch)

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

    """
    def evaluate(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        output_columns: List[str],
        batch_size: int = 32,
        metrics: List[str] = None,
        coerce_fn: Callable = None,
    ):

        # TODO(karan): generalize to TF2

        # Reset the dataset format
        dataset.reset_format()
        dataset.set_format(columns=input_columns + output_columns)

        # TODO(karan): check that the DataPanel conforms to the task definition
        # TODO(karan): figure out how the output_columns will be used by the metrics
        pass

        predictions = []
        targets = []

        # Loop and apply the prediction function
        # TODO(karan): not using .map() here in order to get more fine-grained
        #  control over devices
        for idx in range(0, len(dataset), batch_size):
            # Create the batch
            batch = dataset[idx : idx + batch_size]

            # Predict on the batch
            prediction_dict = self.predict_batch(
                batch=batch, input_columns=input_columns
            )

            # Coerce the predictions
            if coerce_fn:
                prediction_dict = coerce_fn(prediction_dict)

            # Grab the raw target key/values
            target_dict = tz.keyfilter(lambda k: k in output_columns, batch)

            # TODO(karan): general version for non-classification problems
            # TODO(karan): move this to the right device
            if self.task.classification():
                target_dict = tz.valmap(lambda v: torch.tensor(v), target_dict)

            # TODO(karan): incremental metric computation here
            # Append the predictions and targets
            predictions.append(prediction_dict)
            targets.append(target_dict)

        # Consolidate the predictions and targets
        if self.task.classification():
            # TODO(karan): Need to store predictions and outputs from the model
            predictions = tz.merge_with(lambda v: torch.cat(v).to("cpu"), *predictions)
            targets = tz.merge_with(lambda v: torch.cat(v).to("cpu"), *targets)
        else:
            predictions = tz.merge_with(
                lambda x: list(itertools.chain.from_iterable(x)), *predictions
            )
            targets = tz.merge_with(
                lambda x: list(itertools.chain.from_iterable(x)), *targets
            )

        # Compute the metrics
        # TODO(karan): generalize this code to support metric computation for any task

        # Assumes classification, so the output_columns contains a single key for the
        # label
        if self.task.classification():
            assert len(output_columns) == 1  # , "Only supports classification."
            num_classes = self.task.output_schema.features[
                list(self.task.output_schema.columns)[0]
            ].num_classes

        labels = targets[list(targets.keys())[0]]

        if metrics is None:
            if self.task is None:
                raise ValueError(
                    "Must specify metrics if model not associated with task"
                )
            metrics = self.task.metrics

        pred = predictions["pred"].to(self.device)
        target = labels.to(self.device)

        evaluation_dict = {
            metric: compute_metric(metric, pred, target, num_classes)
            for metric in metrics
        }

        # Reset the data format
        dataset.reset_format()

        return evaluation_dict
    """
