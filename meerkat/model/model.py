from typing import Dict, List

import torch

from meerkat import DataPanel
from meerkat.columns.prediction_column import ClassificationOutputColumn
from meerkat.model.metrics import compute_metric


# TODO(Priya): Move some general functions here
class Model:
    def __init__(
        self,
        identifier: str,
        model,
        evaluation_fn=None,
        device: str = None,
        is_classifier: bool = None,
        task: str = None,
    ):

        self.identifier = identifier
        # self.task = task
        self.model = model

        if evaluation_fn is not None:
            self.evaluate = evaluation_fn

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

    def to(self, device: str):
        self.device = device
        return self.model.to(device)

    """
    def __call__(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        output_columns: List[str],
        batch_size: int = 32,
        coerce_fn: Callable = None,
        *args,
        **kwargs
    ):

        return self.evaluate(
            dataset,
            input_columns,
            output_columns,
            batch_size,
            coerce_fn,
            *args,
            **kwargs
        )
    """

    def forward(self, input_batch: Dict) -> Dict:
        raise NotImplementedError

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
