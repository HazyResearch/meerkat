import logging
from typing import Callable, Mapping, Optional

from tqdm.auto import tqdm

from meerkat.provenance import capture_provenance

from .cloneable import CloneableMixin

logger = logging.getLogger(__name__)


class MappableMixin:
    def __init__(self, *args, **kwargs):
        super(MappableMixin, self).__init__(*args, **kwargs)

    @capture_provenance()
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        is_batched_fn: bool = False,
        batch_size: Optional[int] = 1,
        drop_last_batch: bool = False,
        num_workers: Optional[int] = 0,
        output_type: type = None,
        mmap: bool = False,
        materialize: bool = True,
        pbar: bool = False,
        **kwargs,
    ):
        # TODO (sabri): add materialize?
        from meerkat.columns.abstract import AbstractColumn
        from meerkat.datapanel import DataPanel

        """Map a function over the elements of the column."""
        # Check if need to materialize:
        # TODO(karan): figure out if we need materialize=False

        # Just return if the function is None
        if function is None:
            logger.info("`function` None, returning None.")
            return None

        # Ensure that num_workers is not None
        if num_workers is None:
            num_workers = 0

        # Return if `self` has no examples
        if not len(self):
            logger.info("Dataset empty, returning None.")
            return None

        if not is_batched_fn:
            # Convert to a batch function
            function = self._convert_to_batch_fn(
                function, with_indices=with_indices, materialize=materialize
            )
            is_batched_fn = True
            logger.info(f"Converting `function` {function} to a batched function.")

        # Run the map
        logger.info("Running `map`, the dataset will be left unchanged.")
        for i, batch in tqdm(
            enumerate(
                self.batch(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    num_workers=num_workers,
                    materialize=materialize
                    # TODO: collate=batched was commented out in list_column
                )
            ),
            total=(len(self) // batch_size)
            + int(not drop_last_batch and len(self) % batch_size != 0),
            disable=not pbar,
        ):
            # Calculate the start and end indexes for the batch
            start_index = i * batch_size
            end_index = min(len(self), (i + 1) * batch_size)

            # Use the first batch for setup
            if i == 0:
                # Get some information about the function
                function_properties = self._inspect_function(
                    function,
                    with_indices,
                    is_batched_fn,
                    batch,
                    range(start_index, end_index),
                    materialize=materialize,
                )

                # Pull out information
                output = function_properties.output
                dtype = function_properties.output_dtype
                is_mapping = isinstance(output, Mapping)
                writers, output_types, templates, mmap_info = {}, {}, {}, {}
                for key, curr_output in output.items() if is_mapping else [(0, output)]:
                    curr_output_type = (
                        type(AbstractColumn.from_data(curr_output))
                        if output_type is None
                        else output_type
                    )
                    output_types[key] = curr_output_type
                    templates[key] = (
                        curr_output.copy()
                        if isinstance(curr_output, AbstractColumn)
                        and isinstance(curr_output, CloneableMixin)
                        else None
                    )
                    writer = curr_output_type.get_writer(mmap=mmap)

                    # Setup for writing to a certain output column
                    # TODO: support optionally memmapping only some columns
                    if mmap:
                        # Assumes first dimension of output is the batch dimension.
                        shape = (len(self), *curr_output.shape[1:])
                        # Construct the mmap file path
                        # TODO: how/where to store the files
                        path = self.logdir / f"{hash(function)}" / key
                        mmap_info[key] = {"path": path, "shape": shape, "dtype": dtype}
                        # Open the output writer
                        writer.open(str(path), dtype, shape=shape)
                    else:
                        # Create an empty dict or list for the outputs
                        writer.open()
                    writers[key] = writer

            else:
                # Run `function` on the batch
                output = (
                    function(
                        batch,
                        range(i * batch_size, min(len(self), (i + 1) * batch_size)),
                    )
                    if with_indices
                    else function(batch)
                )

            # Append the output
            if output is not None:
                if isinstance(output, Mapping):
                    if set(output.keys()) != set(writers.keys()):
                        raise ValueError(
                            "Map function must return same keys for each batch."
                        )
                    for k, writer in writers.items():
                        writer.write(output[k])
                else:
                    writers[0].write(output)

        # Check if we are returning a special output type
        outputs = {}
        for key, writer in writers.items():
            if mmap:
                writer.flush()
                # TODO: Writers should have correspondence to their own columns
                out = output_types[key].read(
                    path=str(mmap_info[key]["path"]),
                    mmap=mmap,
                    dtype=mmap_info[key]["dtype"],
                    shape=mmap_info[key]["shape"],
                )
                if templates[key] is not None:
                    out = templates[key]._clone(data=out.data)
                outputs[key] = out
            else:
                data = writer.flush()
                outputs[key] = (
                    templates[key]._clone(data=data)
                    if templates[key] is not None
                    else output_types[key].from_data(data)
                )

        if not is_mapping:
            outputs = outputs[0]
        else:
            # TODO (arjundd): This is duck type. We should probably make this
            # class signature explicit.
            outputs = (
                self._clone(data=outputs)
                if isinstance(self, DataPanel)
                else DataPanel.from_batch(outputs)
            )
        return outputs
