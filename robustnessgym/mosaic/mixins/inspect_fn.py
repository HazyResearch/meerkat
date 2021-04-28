from collections import Callable, Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import torch


class FunctionInspectorMixin:
    def __init__(self, *args, **kwargs):
        super(FunctionInspectorMixin, self).__init__(*args, **kwargs)

    def _inspect_function(
        self,
        function: Callable,
        with_indices: bool = False,
        batched: bool = False,
    ) -> SimpleNamespace:
        # Initialize variables to track
        no_output = dict_output = bool_output = list_output = False

        # Run the function to test it
        if batched:
            if with_indices:
                output = function(self[:2], range(2))
            else:
                output = function(self[:2])

        else:
            if with_indices:
                output = function(self[0], 0)
            else:
                output = function(self[0])

        if isinstance(output, Mapping):
            # `function` returns a dict output
            dict_output = True

        elif output is None:
            # `function` returns None
            no_output = True

        elif isinstance(output, bool):
            # `function` returns a bool
            bool_output = True

        elif isinstance(output, (Sequence, torch.Tensor, np.ndarray)):
            # `function` returns a list
            list_output = True
            if batched and (
                isinstance(output[0], bool)
                or (
                    hasattr(output[0], "dtype")
                    and output[0].dtype in (np.bool, torch.bool)
                )
            ):
                # `function` returns a bool per example
                bool_output = True

        return SimpleNamespace(
            dict_output=dict_output,
            no_output=no_output,
            bool_output=bool_output,
            list_output=list_output,
        )
