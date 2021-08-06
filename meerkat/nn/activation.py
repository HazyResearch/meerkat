from functools import reduce

import torch


def _nested_getattr(obj, attr, *args):
    """Get a nested property from an object.

    Example:
    ```
        model = ...
        weights = _nested_getattr(model, "layer4.weights")
    ```
    """
    return reduce(lambda o, a: getattr(o, a, *args), [obj] + attr.split("."))


class ActivationExtractor:
    """Class for extracting activations of a targeted intermediate layer."""

    def __init__(self):
        self.activation = None

    def add_hook(self, module, input, output):
        self.activation = output


# TODO: test on an NLP model
class ActivationOp:
    def __init__(
        self,
        model: torch.nn.Module,
        target_module: str,  # TODO(Priya): Support multiple extraction layers
        device: int = None,
    ):
        """An Operation that runs a forward pass over each example in the
        dataset and stores model activations in a new column.

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
            target_module = _nested_getattr(model, target_module)
        except torch.nn.modules.module.ModuleAttributeError:
            raise ValueError(f"`model` does not have a submodule {target_module}")

        self.extractor = ActivationExtractor()
        target_module.register_forward_hook(self.extractor.add_hook)

        if self.device is not None:
            self.model.to(self.device)

    # Functionality included in specific models
    '''
    def process_batch(
        self,
        dp: DataPanel,
        columns: List[str],
        **kwargs,
    ) -> Dict:
        """Compute example activations for a torch Model.

        Args:
            dp (DataPanel): DataPanel
            columns (list): list of columns
            **kwargs: optional keyword arguments

        Returns:
            Tuple with single output, a torch.Tensor.
        """

        inputs = dp[columns[0]]  # TODO(Priya): columns[0] --> columns?
        if self.device is not None:
            inputs = inputs.data.to(self.device)

        # TODO(Priya): Give an option for grad calculations?
        with torch.no_grad():
            self.model(inputs)
        return {
            f"activation ({self.target_module})": EmbeddingColumn(
                self.extractor.activation.cpu().detach()
            )
        }
    '''
