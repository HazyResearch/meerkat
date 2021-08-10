from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence


def convert_to_batch_column_fn(
    function: Callable, with_indices: bool, materialize: bool = True, **kwargs
):
    """Batch a function that applies to an example."""

    def _function(batch: Sequence, indices: Optional[List[int]], *args, **kwargs):
        # Pull out the batch size
        batch_size = len(batch)

        # Iterate and apply the function
        outputs = None
        for i in range(batch_size):

            # Apply the unbatched function
            if with_indices:
                output = function(
                    batch[i] if materialize else batch.lz[i],
                    indices[i],
                    *args,
                    **kwargs,
                )
            else:
                output = function(
                    batch[i] if materialize else batch.lz[i],
                    *args,
                    **kwargs,
                )

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, dict) else []

            # Append the output
            if isinstance(output, dict):
                for k in output.keys():
                    outputs[k].append(output[k])
            else:
                outputs.append(output)

        if isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    if with_indices:
        # Just return the function as is
        return _function
    else:
        # Wrap in a lambda to apply the indices argument
        return lambda batch, *args, **kwargs: _function(batch, None, *args, **kwargs)


def convert_to_batch_fn(
    function: Callable, with_indices: bool, materialize: bool = True, **kwargs
):
    """Batch a function that applies to an example."""

    def _function(
        batch: Dict[str, List], indices: Optional[List[int]], *args, **kwargs
    ):
        # Pull out the batch size
        batch_size = len(batch[list(batch.keys())[0]])

        # Iterate and apply the function
        outputs = None
        for i in range(batch_size):

            # Apply the unbatched function
            if with_indices:
                output = function(
                    {k: v[i] if materialize else v.lz[i] for k, v in batch.items()},
                    indices[i],
                    *args,
                    **kwargs,
                )
            else:
                output = function(
                    {k: v[i] if materialize else v.lz[i] for k, v in batch.items()},
                    *args,
                    **kwargs,
                )

            if i == 0:
                # Create an empty dict or list for the outputs
                outputs = defaultdict(list) if isinstance(output, dict) else []

            # Append the output
            if isinstance(output, dict):
                for k in output.keys():
                    outputs[k].append(output[k])
            else:
                outputs.append(output)

        if isinstance(outputs, dict):
            return dict(outputs)
        return outputs

    if with_indices:
        # Just return the function as is
        return _function
    else:
        # Wrap in a lambda to apply the indices argument
        return lambda batch, *args, **kwargs: _function(batch, None, *args, **kwargs)
