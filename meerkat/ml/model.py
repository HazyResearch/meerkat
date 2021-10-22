from typing import Dict, List

import torch

from meerkat.datapanel import DataPanel
from meerkat.ml.embedding_column import EmbeddingColumn
from meerkat.ml.metrics import compute_metric
from meerkat.ml.prediction_column import ClassificationOutputColumn


# TODO(Priya): Move some general functions here
class Model(torch.nn.Module):
    def __init__(
        self,
        # identifier: str,
        model: torch.nn.Module,
        # evaluation_fn=None,
        is_classifier: bool = None,
        task: str = None,
        device: str = None,
    ):

        super(Model, self).__init__()

        self.model = model

        # if evaluation_fn is not None:
        #    self.evaluate = evaluation_fn

        if task is None:
            if is_classifier is None or not is_classifier:
                raise ValueError("Task is required for non-classification models.")
        else:
            is_classifier = False

        self.is_classifier = is_classifier
        self.task = task

        if not device:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"

    def forward(self, input_batch: Dict) -> Dict:
        raise NotImplementedError

    def _predict(self, batch, input_cols):
        # Use input_cols instead of input_columns to avoid naming conflict with map

        # Process the batch to prepare input
        input_batch = self.process_batch(batch, input_cols)
        # Run forward pass
        prediction_dict = self.forward(input_batch)
        return prediction_dict

    def _activation(self, batch, input_cols, activation_op):
        # Use input_cols instead of input_columns to avoid naming conflict with map

        # Process the batch
        input_batch = self.process_batch(batch, input_cols)
        # Run forward pass
        _ = self.forward(input_batch)

        activation_dict = {
            f"activation_{activation_op.target_module}": EmbeddingColumn(
                activation_op.extractor.activation.cpu().detach()
            )
        }

        return activation_dict

    def activation(
        self,
        dataset: DataPanel,
        target_module: str,  # TODO(Priya): Support multiple activation layers
        input_columns: List[str],
        batch_size=32,
    ) -> EmbeddingColumn:

        """An Operation that stores model activations in a new Embedding
        column.

        Args:
            dataset (DataPanel): the meerkat DataPanel containing the model inputs.
            target_module (str): the name of the submodule of `model` (i.e. an
                intermediate layer) that outputs the activations we'd like to extract.
                For nested submodules, specify a path separated by "." (e.g.
                `ActivationCachedOp(model, "block4.conv")`).
            input_columns (str): Column containing model inputs
        """

        # To avoid circular dependency for now.
        from meerkat.ml.activation import ActivationOp

        # Get an activation operator
        activation_op = ActivationOp(self.model, target_module, self.device)

        activations = dataset.map(
            function=self._activation,
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=EmbeddingColumn,
            input_cols=input_columns,
            activation_op=activation_op,
        )

        activation_col = activations[f"activation_{activation_op.target_module}"]

        # dataset.add_column(f"activation ({target_module})", activation_col)
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

        # Handles outputs for classification tasks

        predictions = dataset.map(
            function=self._predict,
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=ClassificationOutputColumn,
            input_cols=input_columns,
        )

        # TODO(Priya): How to pass other args of ClassificationOutputColumn above?
        output_col = ClassificationOutputColumn(
            logits=predictions["logits"].data,
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

        dataset.add_column("logits", output_col)
        dataset.add_column("probs", output_col.probabilities())
        dataset.add_column("preds", output_col.predictions())

        return output_dp

    def evaluate(
        self,
        dataset: DataPanel,
        target_column: List[str],  # str?
        pred_column: List[str],  # str?
        metrics: List[str],
        num_classes: int = None,
    ):
        preds = dataset[pred_column[0]]
        labels = dataset[target_column[0]]

        if num_classes is None:
            if isinstance(preds, ClassificationOutputColumn):
                num_classes = preds.num_classes
            elif self.is_classifier:
                raise ValueError(
                    "Must specify num_classes if column type \
                        is not ClassificationOutputColumn"
                )

        evaluation_dict = {
            metric: compute_metric(metric, preds.data, labels.data, num_classes)
            for metric in metrics
        }

        return evaluation_dict

    @staticmethod
    def remap_labels(output_dict: Dict, label_map: List[int]) -> Dict:
        """Map the output labels of the model.

        Example: 3-way classificaiton, with label_map = [1, 2, 0]
        => (model label 0 -> dataset label 1, model label 1 -> dataset label 2, ...).
        """

        # Check the number of classes
        num_classes = len(label_map)

        # Remap the columns of all outputs that have # columns = num_classes
        for key in output_dict:
            if output_dict[key].shape[-1] == num_classes:
                output_dict[key] = output_dict[key][..., label_map]

        # Remap the pred key
        inverse_label_map = [
            t[1] for t in sorted([(label, i) for i, label in enumerate(label_map)])
        ]
        output_dict["pred"] = torch.tensor(inverse_label_map)[output_dict["pred"]]

        return output_dict

    @classmethod
    def _state_keys(cls) -> set:
        """List of attributes that describe the state of the object."""
        return {"model", "is_classifier", "task", "device"}
