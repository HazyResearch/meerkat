from functools import partial

from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")


def _get_reduction_fn(reduction_name):
    if reduction_name == "max":
        reduction_fn = partial(torch.mean, dim=[-1, -2])
    elif reduction_name == "mean":
        reduction_fn = partial(torch.mean, dim=[-1, -2])
    else:
        raise ValueError(f"reduction_fn {reduction_name} not supported.")
    reduction_fn.__name__ = reduction_name
    return reduction_fn


class ActivationExtractor:
    """Class for extracting activations a targetted intermediate layer."""

    def __init__(self, reduction_fn: callable = None):
        self.activation = None
        self.reduction_fn = reduction_fn

    def add_hook(self, module, input, output):
        if self.reduction_fn is not None:
            output = self.reduction_fn(output)
        self.activation = output
