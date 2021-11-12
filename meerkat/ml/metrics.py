import re
import statistics
from typing import Callable, Sequence, Union

import numpy as np
import torch

from meerkat.tools.lazy_loader import LazyLoader

nltk = LazyLoader("nltk")
rouge_score = LazyLoader("rouge_score")
torchmetrics = LazyLoader("torchmetrics")


def get_metric(name: str) -> Callable:
    """Get metrics from string names."""
    if name == "accuracy":
        return accuracy
    elif name == "f1":
        return f1
    elif name == "f1_micro":
        return f1_micro
    elif name == "f1_macro":
        return f1_macro
    else:
        raise NotImplementedError(f"Metric name {name} not recognized.")


def accuracy(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate accuracy."""
    return torchmetrics.functional.accuracy(preds=predictions, target=labels)


def f1(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate F1 score for binary classification."""
    return torchmetrics.functional.f1(preds=predictions, target=labels)


def f1_micro(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate micro F1 score for multi-class classification."""
    return torchmetrics.functional.f1(preds=predictions, target=labels, average="micro")


def f1_macro(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate macro F1 score for multi-class classification."""
    return torchmetrics.functional.f1(preds=predictions, target=labels, average="macro")


def class_distribution(
    labels: Union[list, np.array, torch.Tensor],
    num_classes: int = None,
    min_label: int = 0,
):
    """Calculate the aggregated class distribution."""
    if isinstance(labels, list):
        labels = np.array(labels)

    if len(labels.shape) == 1:
        # Find the unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Calculate the number of classes
        if num_classes is None:
            max_label = np.max(unique_labels)
            num_classes = max_label - min_label + 1

        # Fill out the distribution
        dist = np.zeros(num_classes)
        dist[(unique_labels - min_label).astype(int)] = counts / labels.shape[0]
        return dist
    elif len(labels.shape) == 2:
        return np.mean(labels, axis=0)
    else:
        raise ValueError("`labels` must be 1 or 2-dimensional.")


def iou_score(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
    num_classes: int = None,
):
    """Calculate IoU."""
    return torchmetrics.functional.iou(
        preds=predictions, target=labels, num_classes=num_classes
    ).item()


def dice(
    predictions: Union[list, np.array, torch.Tensor],
    labels: Union[list, np.array, torch.Tensor],
):
    """Calculate Dice Score."""
    return torchmetrics.functional.dice_score(preds=predictions, target=labels).item()


# TODO Refactor into separate class for each metric
# TODO change signature of compute_metric
def compute_metric(
    metric: str,
    predictions: Union[Sequence, torch.Tensor],
    labels: Union[Sequence, torch.Tensor],
    num_classes: int,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute metric given predictions and target labels.

    Args:
        metric (str): name of metric
        predictions (Union[Sequence, torch.Tensor]): a sequence of predictions
            (rouge metrics) or a torch Tensor (other metrics) containing predictions
        labels (Union[Sequence, torch.Tensor]): a sequence of labels
            (rouge metrics) or a torch Tensor (other metrics) containing target labels
        num_classes (int): number of classes

    Returns:
        the calculate metric value
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()

    # Classification metrics
    if metric == "accuracy":
        return accuracy(predictions=predictions, labels=labels)
    elif metric == "f1":
        return f1(predictions=predictions, labels=labels)
    elif metric == "f1_micro":
        return f1_micro(predictions=predictions, labels=labels)
    elif metric == "f1_macro":
        return f1_macro(predictions=predictions, labels=labels)
    elif metric in ("Rouge-1", "Rouge-2", "Rouge-L"):
        # Calculate rouge scores
        if metric == "Rouge-1":
            metric_id = "rouge1"
        elif metric == "Rouge-2":
            metric_id = "rouge2"
        else:
            metric_id = "rougeLsum"
        scorer = rouge_score.rouge_scorer.RougeScorer([metric_id], use_stemmer=True)
        # TODO Remove summarizaton-specific 'format_summary' call
        # TODO Don't call scorer.score separately for each metric
        score = statistics.mean(
            scorer.score(format_summary(reference), format_summary(pred))[
                metric
            ].fmeasure
            for reference, pred in zip(labels, predictions)
        )
    elif metric == "class_dist":
        # Calculate class distribution
        score = class_distribution(labels=labels, num_classes=num_classes)
    elif metric == "pred_dist":
        # Calculate predicted class distribution
        score = class_distribution(labels=predictions, num_classes=num_classes)
    elif metric == "iou":
        # Calculate IoU
        score = iou_score(
            predictions=predictions, labels=labels, num_classes=num_classes
        )
    elif metric == "dice_score":
        # Calculate Dice Score
        score = dice(predictions=predictions, labels=labels)
    else:
        raise NotImplementedError

    return score


def format_summary(x: str) -> str:
    """Format summary text for computing rouge."""
    re.sub("<n>", "", x)  # remove pegasus newline char
    return "\n".join(nltk.sent_tokenize(x))
