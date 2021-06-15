from typing import Callable, Dict, List

import torch

from mosaic import DataPanel

"""
from mosaic.tools.lazy_loader import LazyLoader

from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


from robustnessgym.core.metrics import compute_metric
from robustnessgym.core.slice import SliceDataPanel as DataPanel
from robustnessgym.tasks.task import Task

ludwig_api = LazyLoader("ludwig.api")
nltk = LazyLoader("nltk")
"""


# TODO(Priya): Move some general functions here
class Model:
    def __init__(
        self,
        identifier: str,
        # task: Task,
        model=None,
        evaluation_fn=None,
        device: str = None,
        # is_classifier: bool = None,
    ):

        self.identifier = identifier
        # self.task = task
        self.model = model

        if evaluation_fn is not None:
            self.evaluate = evaluation_fn

        # TODO(Priya): Implementation for non-classification tasks
        self.outputs = {"probs", "logits", "pred"}

        """
        if self.task is None:
            if is_classifier is None:
                raise ValueError("'is_classifier' required when task not passed")
        else:
            is_classifier = self.task.classification()


        if is_classifier:
            self.outputs = {
                "probs",
                "logits",
                "pred",
                # 'embeddings',
                # TODO(karan): other information from the model e.g. embeddings which
                #  aren't task related?
            }
        else:
            self.outputs = {
                "pred",
                # 'embeddings',
                # TODO(karan): other information from the model e.g. embeddings which
                #  aren't task related?
            }
        """
        if not device:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"

    def to(self, device: str):
        self.device = device
        return self.model.to(device)

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

    '''
    @classmethod
    def huggingface(
        cls,
        identifier: str,
        task: Task = None,
        model: Optional[AutoModel] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        is_classifier=None,
    ):
        """

        Args:
            identifier:
            task:
            model:
            tokenizer:

        Returns:

        Examples:
            >>> Model.huggingface(identifier='', task=TernaryNaturalLanguageInference())
            >>> Model.huggingface(identifier='', \
            model=AutoModelForSequenceClassification.from_pretrained(''),
            tokenizer=AutoTokenizer.from_pretrained(''))

        """

        return HuggingfaceModel(
            identifier=identifier,
            task=task,
            model=model,
            tokenizer=tokenizer,
            is_classifier=is_classifier,
        )
    '''

    def forward(self, input_batch: Dict) -> Dict:
        raise NotImplementedError

    def evaluate(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        output_columns: List[str],
        batch_size: int = 32,
        coerce_fn: Callable = None,
    ):
        raise NotImplementedError

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
