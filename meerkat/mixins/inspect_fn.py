from collections.abc import Callable, Mapping, Sequence
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
        is_batched_fn: bool = False,
        data=None,
        indices=None,
        materialize=True,
        **kwargs
    ) -> SimpleNamespace:

        # Initialize variables to track
        no_output = dict_output = bool_output = list_output = False

        # If dict_output = True and `function` is used for updating the `DataPanel`
        # useful to know if any existing column is modified
        updates_existing_column = True
        existing_columns_updated = []

        # Record the shape and dtype of the output
        output_shape = output_dtype = None

        # Run the function to test it
        if data is None:
            if is_batched_fn:
                data = self[:2] if materialize else self.lz[:2]
            else:
                data = self[0] if materialize else self.lz[0]

        if indices is None:
            if is_batched_fn:
                indices = range(2)
            else:
                indices = 0

        if with_indices and is_batched_fn:
            output = function(data, indices, **kwargs)
        elif with_indices and not is_batched_fn:
            output = function(data, indices, **kwargs)
        else:
            output = function(data, **kwargs)

        # lazy import to avoid circular dependency
        from meerkat.columns.abstract import AbstractColumn
        from meerkat.columns.numpy_column import NumpyArrayColumn
        from meerkat.columns.tensor_column import TensorColumn

        if isinstance(output, Mapping):
            # `function` returns a dict output
            dict_output = True

            # Check if `self` is a `DataPanel`
            if hasattr(self, "all_columns"):
                # Set of columns that are updated
                existing_columns_updated = set(self.all_columns).intersection(
                    set(output.keys())
                )

                # Check if `function` updates an existing column
                if len(existing_columns_updated) == 0:
                    updates_existing_column = False

        elif output is None:
            # `function` returns None
            no_output = True

        elif (
            isinstance(output, (bool, np.bool_))
            or (
                isinstance(output, (np.ndarray, NumpyArrayColumn))
                and output.dtype == bool
            )
            or (
                isinstance(output, (torch.Tensor, TensorColumn))
                and output.dtype == torch.bool
            )
        ):

            # `function` returns a bool
            bool_output = True

        elif isinstance(output, (Sequence, AbstractColumn, torch.Tensor, np.ndarray)):
            # `function` returns a list
            list_output = True
            if is_batched_fn and (
                isinstance(output[0], (bool, np.bool_))
                or (isinstance(output[0], np.ndarray) and (output[0].dtype == bool))
                or (
                    isinstance(output[0], torch.Tensor)
                    and (output[0].dtype == torch.bool)
                )
            ):
                # `function` returns a bool per example
                bool_output = True

        if not isinstance(output, Mapping) and not isinstance(output, bool):
            # Record the shape of the output
            if hasattr(output, "shape"):
                output_shape = output.shape
            elif len(output) > 0 and hasattr(output[0], "shape"):
                output_shape = output[0].shape

            # Record the dtype of the output
            if hasattr(output, "dtype"):
                output_dtype = output.dtype
            elif len(output) > 0 and hasattr(output[0], "dtype"):
                output_dtype = output[0].dtype

        return SimpleNamespace(
            output=output,
            dict_output=dict_output,
            no_output=no_output,
            bool_output=bool_output,
            list_output=list_output,
            updates_existing_column=updates_existing_column,
            existing_columns_updated=existing_columns_updated,
            output_shape=output_shape,
            output_dtype=output_dtype,
        )
