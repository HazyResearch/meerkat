from functools import reduce
from typing import Union

import torch

from meerkat.ml.model import Model


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
        """Registers a forward hook on the target layer to facilitate
        extracting model activations.

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
        except AttributeError:
            raise ValueError(f"`model` does not have a submodule {target_module}")

        self.extractor = ActivationExtractor()
        target_module.register_forward_hook(self.extractor.add_hook)

        if not device:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda:0"
        self.model.to(self.device)


def _nested_getattr(obj, attr, *args):
    """Get a nested property from an object.

    Example:
    ```
        model = ...
        weights = _nested_getattr(model, "layer4.weights")
    ```
    """

    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))
