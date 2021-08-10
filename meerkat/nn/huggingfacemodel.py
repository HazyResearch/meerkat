from __future__ import annotations

from functools import partial
from typing import Dict, List, Optional

import cytoolz as tz
import torch
from tqdm import tqdm

from meerkat.columns.list_column import ListColumn
from meerkat.datapanel import DataPanel
from meerkat.nn.activation import ActivationOp
from meerkat.nn.embedding_column import EmbeddingColumn
from meerkat.nn.model import Model
from meerkat.tools.lazy_loader import LazyLoader

AutoTokenizer = LazyLoader("transformers.AutoTokenizer")


class HuggingfaceModel(Model):
    def __init__(
        self,
        identifier: str,
        model,
        tokenizer: Optional[AutoTokenizer] = None,
        is_classifier: bool = None,
        task: str = None,
        device: str = None,
    ):

        if model is None:
            raise ValueError(
                f"A HuggingFace model is required with {self.__class__.__name__}."
            )

        super(HuggingfaceModel, self).__init__(
            # identifier=identifier,
            model=model,
            is_classifier=is_classifier,
            task=task,
            device=device,
        )

        self.tokenizer = tokenizer
        if tokenizer is None:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.identifier)

        # Move the model to device
        self.to(self.device)

    def forward(self, input_batch: Dict) -> Dict:

        if self.is_classifier:
            # Run the model on the input_batch
            with torch.no_grad():
                outputs = self.model(**input_batch)

            # probs and preds can be handled at ClassificationOutputColumn
            # TODO(Priya): See if there is any case where these are to be returned
            # Logits are present at the 0th index
            output_dict = {"logits": outputs[0].to("cpu")}

        else:
            # TODO (Priya): Support for only summarization right now.
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
                output_dict = {"preds": summaries}

        return output_dict

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

        # dataset.add_column(f"activation ({target_module})", activation_col)
        return activation_col

    def summarization(
        self, dataset: DataPanel, input_columns: List[str], batch_size: int = 32
    ) -> DataPanel:

        output_dp = dataset.map(
            function=partial(self._predict, input_columns=input_columns),
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=ListColumn,
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
        # TODO(Priya): The separate functions can be merged later
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
        elif self.task == "summarization":
            return self.summarization(dataset, input_columns, batch_size)
        else:
            raise NotImplementedError

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return super()._state_keys() | {
            "identifier",
            "model",
            "tokenizer",
            "is_classifier",
            "task",
            "device",
        }
