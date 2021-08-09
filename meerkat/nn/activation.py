from functools import reduce
from typing import List, Union

import torch

from meerkat.datapanel import DataPanel
from meerkat.nn.embedding_column import EmbeddingColumn
from meerkat.nn.model import Model


class ActivationExtractor:
    """Class for extracting activations of a targeted intermediate layer."""

    def __init__(self):
        self.activation = None

    def add_hook(self, module, input, output):
        self.activation = output


class ActivationOp:
    def __init__(
        self,
        model: Union[torch.nn.Module, Model],
        target_module: str,  # TODO(Priya): Support multiple extraction layers
        device: int = None,
    ):
        """Registers a forward hook on the target layer to facilitate extracting
        model activations

        Args:
            model (nn.Module): the torch model from which activations are extracted
            target_module (str): the name of the submodule of `model` (i.e. an
                intermediate layer) that outputs the activations we'd like to extract.
                For nested submodules, specify a path separated by "." (e.g.
                `ActivationCachedOp(model, "block4.conv")`).
            device (int, optional): the device for the forward pass. Defaults to None,
                in which case the CPU is used.
        """
        self.model = model
        self.device = device
        self.target_module = target_module

        try:
            target_module = _nested_getattr(
                model.model if isinstance(model, Model) else model, target_module
            )
        except torch.nn.modules.module.ModuleAttributeError:
            raise ValueError(f"`model` does not have a submodule {target_module}")

        self.extractor = ActivationExtractor()
        target_module.register_forward_hook(self.extractor.add_hook)

        if not device:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"
        self.model.to(self.device)

    def _activation(self, batch, input_cols, forward: bool = False):
        # Use input_cols instead of input_columns to avoid naming conflict with map

        if forward:
            # Process the batch
            input_batch = self.model.process_batch(batch, input_cols)
            # Run forward pass
            _ = self.model.forward(input_batch)

        activation_dict = {
            f"activation_{self.target_module}": EmbeddingColumn(
                self.extractor.activation.cpu().detach()
            )
        }

        return activation_dict

    def activation(
        self,
        dataset: DataPanel,
        input_columns: List[str],
        forward: bool = False,
        batch_size: int = 32,
    ) -> EmbeddingColumn:

        """An Operation that stores model activations in a new Embedding column.

        Args:
            model (nn.Module): the torch model from which activations are extracted.
            dataset (DataPanel): the meerkat DataPanel containing the model inputs.
            target_module (str): the name of the submodule of `model` (i.e. an
                intermediate layer) that outputs the activations we'd like to extract.
                For nested submodules, specify a path separated by "." (e.g.
                `ActivationCachedOp(model, "block4.conv")`).
            input_columns (str): Column containing model inputs
            forward (bool): If True, runs a forward pass on the model.
                model needs to be an Instance of Model class to use this.
            device (int, optional): the device for the forward pass. Defaults to None,
                in which case the CPU is used.

        """
        # Get an activation operator

        if forward:
            assert isinstance(
                self.model, Model
            ), "Model class object required to use forward method"

        activations = dataset.map(
            function=self._activation,
            is_batched_fn=True,
            batch_size=batch_size,
            output_type=EmbeddingColumn,
            input_cols=input_columns,
            forward=forward,
        )

        activation_col = activations[f"activation_{self.target_module}"]

        # dataset.add_column(f"activation ({target_module})", activation_col)
        return activation_col


def _nested_getattr(obj, attr, *args):
    """Get a nested property from an object.

    Example:
    ```
        model = ...
        weights = _nested_getattr(model, "layer4.weights")
    ```
    """

    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))
